# 🥗 Personalized Nutrition Recommender using Reinforcement Learning

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Colab](https://img.shields.io/badge/Open%20in-Colab-yellow.svg)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A Reinforcement Learning agent that acts as a personalized daily nutrition coach, recommending culturally-appropriate meals across breakfast, lunch, dinner, and snack to help users meet their nutritional goals by end of day.

**Author:** Zahra El Ghourani | **Supervisor:** Dr. Abbas Rammal  
**Course:** Reinforcement Learning — Master of Science in Data Science, Lebanese American University

---

## 🎯 Project Overview

Most food recommendation systems treat each suggestion as a one-shot, independent decision. This project frames daily meal planning as a **Markov Decision Process (MDP)**, where each meal choice affects the remaining nutritional budget for the rest of the day — a fundamentally sequential decision problem.

Three agents are trained and compared:
- **Rule-Based Baseline** — greedy selection minimizing distance to remaining targets
- **Q-Learning** — tabular method with discretized state space
- **Deep Q-Network (DQN)** — neural network with experience replay and target network

---

## ✨ Features

- 🍽️ **17 clinically grounded diet protocols** — Ketogenic, DASH, Mediterranean, Vegan, High-Protein, Atkins, Zone, and more
- 🌍 **5 cuisine preferences** — Lebanese, Mediterranean, Western, Asian, Mexican
- ⚠️ **Allergen-based hard action masking** — gluten, dairy, nuts, eggs, shellfish, soy, fish
- 🩺 **Health condition penalties** — diabetes (sugar management) and hypertension (sodium management)
- 🎲 **Diversity bonus** — encourages varied meal recommendations
- 📊 **Real dataset** — Food.com Recipes Dataset (230,000+ recipes)

---

## 🧠 MDP Formulation

| Component | Description |
|-----------|-------------|
| **State S** | Calories consumed, protein, carbs, fat, sugar, sodium, meal slot, remaining budget, diet type (one-hot), cuisine (one-hot), health flags |
| **Actions A** | Select one food item from curated database (~300 recipes per cuisine) |
| **Reward R** | −λ₁\|cal_gap\| − λ₂\|protein_gap\| − λ₃\|carb_gap\| − λ₄\|fat_gap\| + diversity_bonus |
| **Discount γ** | 0.99 — encourages planning across the full day |

---

## 📁 Project Structure

```
nutrition-rl/
│
├── nutrition_rl.ipynb          # Main Colab notebook (all-in-one)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── src/
│   ├── environment.py          # NutritionEnv (Gymnasium-style)
│   ├── q_learning.py           # Tabular Q-Learning agent
│   ├── dqn.py                  # Deep Q-Network agent
│   ├── baseline.py             # Rule-based greedy baseline
│   └── evaluate.py             # Evaluation metrics and plots
│
├── data/
│   └── README.md               # Dataset download instructions
│
└── results/
    ├── learning_curves.png     # Training plots
    └── evaluation_results.png  # Comparison charts
```

---

## 🗃️ Dataset

This project uses the **Food.com Recipes Dataset** available on Kaggle:

👉 [Food.com Recipes and User Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)

Download `RAW_recipes.csv` and place it in your Google Drive under `MyDrive/Recipes/`.

---

## 🚀 How to Run

### Option 1 — Google Colab (recommended)

1. Open `nutrition_rl.ipynb` in Google Colab
2. Mount your Google Drive
3. Upload `RAW_recipes.csv` to `MyDrive/Recipes/`
4. Run all cells in order

### Option 2 — Local (requires GPU recommended)

```bash
# Clone the repo
git clone https://github.com/zahraghourani/nutrition-rl.git
cd nutrition-rl

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run training
python src/train.py
```

---

## 📊 Results

| Metric | Rule-Based | Q-Learning | DQN |
|--------|-----------|------------|-----|
| Avg Daily Reward | -- | -- | -- |
| Calorie Adherence (%) | -- | -- | -- |
| Avg Calorie Gap (kcal) | -- | -- | -- |
| Avg Protein Gap (g) | -- | -- | -- |
| Recommendation Diversity | -- | -- | -- |
| Allergen Violations (%) | 0.0 | 0.0 | 0.0 |

*Results will be updated after final training on GPU.*

---

## 🔧 Requirements

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
gymnasium>=0.29.0
```

---

## 📖 References

- Mnih et al. (2015). *Human-level control through deep reinforcement learning*. Nature.
- Watkins & Dayan (1992). *Q-Learning*. Machine Learning.
- Amiri et al. (2024). *Delighting palates with AI: Reinforcement learning's triumph in crafting personalized meal plans*. Nutrients.
- Tellechea et al. (2025). *Population-level analysis of personalized food recommendation using reinforcement learning*. Foods.
- Durrani (2015). *Types of diet and their nutritional impact on health*. Science & Technology Journal.

---

## 📄 License

MIT License — feel free to use and adapt this project.

---

*Lebanese American University — School of Arts and Sciences — April 2026*
