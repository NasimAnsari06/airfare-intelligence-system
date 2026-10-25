# Airfare Intelligence System
### Historical Fare Analytics · Route Insights · Machine Learning

An interactive data science application that transforms historical flight data into actionable airfare insights through exploratory analytics, airline comparisons, route intelligence, and machine-learning-based fare estimation.

**[Live Demo](https://airfare-intelligence-system-4hw9s5imhilyg8uxqzrm3z.streamlit.app/)**

---

## Overview

Airfare Intelligence System is designed to help users explore historical flight fares, understand pricing patterns, compare airlines, and investigate routes through an interactive dashboard. It combines data analysis, visualisation, and a saved machine learning model in a single Streamlit application.

## Key Capabilities

- **Executive Dashboard** — Summarises flight records, airlines, routes, and historical fare statistics.
- **Fare Explorer** — Filters and compares historical fares using flight and travel attributes.
- **Fare Estimator** — Provides an interface for estimating fares using an integrated trained model.
- **Route Intelligence** — Explores historical pricing patterns across routes.
- **Dataset Explorer** — Enables interactive inspection and filtering of flight records.

## Dataset at a Glance

| Metric | Value |
|---|---:|
| Flight records | 300,153 |
| Airlines | 6 |
| Unique routes | 30 |
| Average historical fare | ₹20,890 |

*Values represent the statistics displayed by the application.*

## Technology Stack

**Language:** Python  
**Data Analysis:** Pandas, NumPy  
**Machine Learning:** Scikit-learn, Joblib  
**Visualisation:** Plotly  
**Web Application:** Streamlit  
**Development:** Jupyter Notebook, Git

## Project Structure

```text
airfare-intelligence-system/
├── Data/
│   └── airfare_data.csv
├── Model/
│   └── best_model.joblib
├── Notebook/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   └── 03_preprocessing_modeling.ipynb
├── app.py
├── requirements.txt
└── README.md
```

## Methodology

1. **Data Understanding** — Examine the dataset, columns, and data types.
2. **Exploratory Analysis** — Investigate fare distributions, airline differences, and route patterns.
3. **Preprocessing** — Prepare features and data for machine learning.
4. **Model Development** — Train and evaluate a fare-prediction model.
5. **Deployment** — Integrate the saved model into an interactive Streamlit dashboard.

## Run Locally

```bash
git clone https://github.com/NasimAnsari06/airfare-intelligence-system.git
python -m streamlit run app.py
```

## Important Note

This project analyses **historical airfare data**; it does not fetch live ticket prices or guarantee future fares. Prediction quality depends on the trained model and its evaluation results.

## Author

**Nasimuddin Ansari**  
B.Tech — Computer Science & Engineering  
---

*Developed to bring data analytics, machine learning, and interactive visualisation together in one application.*
