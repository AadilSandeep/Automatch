# AutoMatch: Intelligent Vehicle Recommendation System

<div align="center">

[![Python 3.x](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.8.0-F7931E.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent vehicle recommendation engine that translates user lifestyle habits and budget constraints into personalized vehicle suggestions using machine learning.

[Quick Start](#quick-start) • [How It Works](#how-it-works) • [Tech Stack](#tech-stack) • [Installation](#installation)

</div>

---

## 📋 Overview

**AutoMatch** is a machine learning-powered system that recommends the perfect vehicle by understanding user preferences through lifestyle and budget constraints. Instead of overwhelming users with thousands of options, it filters vehicles intelligently and ranks them by personal relevance.

### Key Features

✨ **Two-Stage Recommendation Engine**
- **Stage 1 (Classification)**: Filters vehicles into budget tiers using machine learning
- **Stage 2 (Ranking)**: Ranks candidates using cosine similarity for personalized matching

🎯 **User-Centric Design**
- Lifestyle-to-specs mapping: Converts lifestyle questions into technical specifications
- Interactive UI built with Streamlit for seamless user interaction
- Explainability: Each recommendation includes reasoning for the match

🚗 **Smart Matching**
- Considers engine specs, mileage, seating, and performance preferences
- Handles 120+ vehicle features intelligently
- Robust data preprocessing and imputation

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.x |
| **UI Framework** | Streamlit |
| **Machine Learning** | Scikit-Learn |
| **Data Processing** | Pandas, NumPy |
| **Classification** | Random Forest, CART (Decision Tree) |
| **Ranking** | Cosine Similarity |

---

## 🚀 Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Aadilsandeep/Automatch.git
   cd Automatch
