# 🚗 Intelligent Vehicle Recommendation System

An intelligent vehicle recommendation system that combines **Machine Learning-based filtering** with **Similarity-Based Ranking** to recommend the most suitable vehicles based on user preferences.

---

## 📌 Project Overview

Choosing the right vehicle can be challenging due to the large number of available models, specifications, and price ranges.

This project addresses that problem using a **two-stage recommendation architecture**:

### Stage 1: Constraint Filtering
Uses:
- CART (Decision Tree)
- Random Forest

to narrow down vehicles based on user requirements such as:
- Budget category
- Fuel type
- Body type
- Vehicle specifications

### Stage 2: Similarity Ranking
Uses:
- Cosine Similarity

to rank shortlisted vehicles according to how closely they match the user's ideal preference profile.

The system finally returns the **Top-N most relevant vehicles**.

---

## 🎯 Objectives

- Automate vehicle selection
- Reduce search complexity
- Provide personalized recommendations
- Combine machine learning and recommender system concepts
- Improve recommendation accuracy through hybrid filtering

---

## 🏗 System Architecture

```text
                    User Preferences
                            │
                            ▼
               ┌───────────────────────┐
               │ Data Preprocessing    │
               └───────────┬───────────┘
                           │
                           ▼
               ┌───────────────────────┐
               │ Stage 1 Filtering     │
               │ CART + Random Forest  │
               └───────────┬───────────┘
                           │
                           ▼
               Filtered Vehicle Set
                           │
                           ▼
               ┌───────────────────────┐
               │ Stage 2 Ranking       │
               │ Cosine Similarity     │
               └───────────┬───────────┘
                           │
                           ▼
                Top Recommended Vehicles
```

---

## 📊 Dataset

### Car Dataset

- Source: Kaggle
- Records: 1267+
- Features: 141

Contains:
- Make
- Model
- Variant
- Fuel Type
- Body Type
- Engine Specifications
- Power
- Torque
- Seating Capacity
- Ex-Showroom Price
- and many more

---

## 🧹 Data Preprocessing

### Feature Extraction

The following numerical features are extracted from textual specifications:

| Original Feature | Extracted Feature |
|-----------------|------------------|
| Displacement | Engine CC |
| Power | Horsepower |
| Torque | Torque Value |

### Cleaning Steps

- Remove irrelevant columns
- Handle missing values
- Convert prices to numeric values
- Extract numerical values using regex
- Encode categorical features
- Automated preprocessing using Scikit-Learn Pipelines

---

## 🤖 Machine Learning Models

### CART (Decision Tree)

Used for:

- Interpretable decision making
- Vehicle filtering
- Rule extraction

### Random Forest

Used for:

- Robust classification
- Improved filtering accuracy
- Ensemble learning

### Current Performance

| Model | Accuracy |
|---------|---------|
| CART | 86.22% |
| Random Forest | 90.94% |

---

## ⚙️ Technologies Used

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Scikit-Learn

### Algorithms

- Decision Tree (CART)
- Random Forest
- Cosine Similarity

---

## 📁 Project Structure

```text
CAR/
│
├── data/
│   └── cars.csv
│
├── src/
│   └── main.py
│
├── requirements.txt
│
└── README.md
```

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/your-username/vehicle-recommendation-system.git
```

### Move into Project Folder

```bash
cd vehicle-recommendation-system
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python src/main.py
```

---

## 📈 Current Progress

### Completed

- Dataset Collection
- Data Cleaning
- Feature Extraction
- Budget Segmentation
- CART Training
- Random Forest Training
- Performance Evaluation

### Upcoming

- Vehicle Shortlisting Module
- Cosine Similarity Ranking
- Top-N Recommendation Engine
- Explainability Layer
- Streamlit User Interface

---

## 🔬 Future Enhancements

- Integration of bike datasets
- Hybrid vehicle recommendations
- Explainable AI dashboard
- Streamlit web application
- Personalized recommendation profiles
- Feature importance visualization

---

## 👨‍💻 Team

Mini Project – B.Tech Computer Science

Developed as part of academic coursework on Machine Learning and Intelligent Recommendation Systems.

---

## 📜 License

This project is intended for educational and research purposes.
