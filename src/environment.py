import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MicrogridEnv(gym.Env):
    """
    A simple residential microgrid environment.

    State:  [soc, hour_norm, solar_gen, demand, price]
    Actions: 0=charge, 1=discharge, 2=idle
    Reward: negative electricity cost, penalty for constraint violations
    """

    metadata = {"render_modes": []}

    BATTERY_CAPACITY     = 10.0
    MAX_CHARGE_RATE      = 3.0
    MAX_DISCHARGE_RATE   = 3.0
    CHARGE_EFFICIENCY    = 0.95
    DISCHARGE_EFFICIENCY = 0.95
    SOC_MIN              = 0.1
    SOC_MAX              = 0.9
    CONSTRAINT_PENALTY   = 5.0

    def __init__(self, seed: int = 42):
        super().__init__()
        self.np_random = np.random.default_rng(seed)

        low  = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 5.0, 5.0, 0.5], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space      = spaces.Discrete(3)

        self.soc             = 0.5
        self.hour            = 0
        self._solar_profile  = None
        self._demand_profile = None
        self._price_profile  = None

    def _generate_solar(self):
        hours = np.arange(24)
        base  = np.clip(3.5 * np.sin(np.pi * (hours - 6) / 12), 0, None)
        noise = self.np_random.normal(0, 0.2, 24)
        return np.clip(base + noise, 0, 5).astype(np.float32)

    def _generate_demand(self):
        hours   = np.arange(24)
        morning = 1.5 * np.exp(-0.5 * ((hours - 8) / 2) ** 2)
        evening = 2.5 * np.exp(-0.5 * ((hours - 19) / 2) ** 2)
        base    = 0.5 + morning + evening
        noise   = self.np_random.normal(0, 0.15, 24)
        return np.clip(base + noise, 0.2, 5).astype(np.float32)

    def _generate_price(self):
        hours = np.arange(24)
        base  = 0.08 + 0.22 * np.clip(np.sin(np.pi * (hours - 10) / 12), 0, None)
        noise = self.np_random.normal(0, 0.01, 24)
        return np.clip(base + noise, 0.03, 0.50).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.soc             = float(self.np_random.uniform(0.3, 0.7))
        self.hour            = 0
        self._solar_profile  = self._generate_solar()
        self._demand_profile = self._generate_demand()
        self._price_profile  = self._generate_price()

        return self._get_obs(), {}

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        solar  = self._solar_profile[self.hour]
        demand = self._demand_profile[self.hour]
        price  = self._price_profile[self.hour]
        penalty = 0.0

        if action == 0:  # Charge
            energy_in = self.MAX_CHARGE_RATE * self.CHARGE_EFFICIENCY
            new_soc   = self.soc + energy_in / self.BATTERY_CAPACITY
            if new_soc > self.SOC_MAX:
                penalty += self.CONSTRAINT_PENALTY
                new_soc  = self.SOC_MAX
            self.soc   = new_soc
            grid_power = demand - solar + self.MAX_CHARGE_RATE

        elif action == 1:  # Discharge
            energy_out = self.MAX_DISCHARGE_RATE * self.DISCHARGE_EFFICIENCY
            new_soc    = self.soc - energy_out / self.BATTERY_CAPACITY
            if new_soc < self.SOC_MIN:
                penalty += self.CONSTRAINT_PENALTY
                new_soc  = self.SOC_MIN
            self.soc   = new_soc
            grid_power = max(0.0, demand - solar - self.MAX_DISCHARGE_RATE)

        else:  # Idle
            grid_power = max(0.0, demand - solar)

        grid_power = max(0.0, grid_power)
        cost       = grid_power * price
        reward     = -cost - penalty

        self.hour     += 1
        terminated     = self.hour >= 24

        return self._get_obs(), reward, terminated, False, {
            "cost": cost, "grid_power": grid_power,
            "solar": solar, "demand": demand,
            "price": price, "soc": self.soc,
        }

    def _get_obs(self):
        hour = min(self.hour, 23)
        return np.array([
            self.soc,
            hour / 23.0,
            self._solar_profile[hour],
            self._demand_profile[hour],
            self._price_profile[hour],
        ], dtype=np.float32)

    def render(self):
        hour = min(self.hour, 23)
        print(
            f"Hour: {hour:02d} | SoC: {self.soc:.2f} | "
            f"Solar: {self._solar_profile[hour]:.2f} kW | "
            f"Demand: {self._demand_profile[hour]:.2f} kW | "
            f"Price: {self._price_profile[hour]:.3f} $/kWh"
        )