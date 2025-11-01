# 🚗 EVisionAI - EV Sales & Adoption Analytics Platform

## Project Overview

EVisionAI leverages the EV Sales and Adoption Dataset to deliver a comprehensive platform featuring:

- A GenAI-powered chatbot for answering technical and analytics-driven EV queries.
- An interactive dashboard providing powerful visual insights, forecasting, and price predictions—all powered by machine learning.

---

## Problem Statement

Build a conversational AI assistant and analytics dashboard that help users:

- Forecast future electric vehicle (EV) sales using historical sales and feature data.
- Predict the price of an EV based on its specifications (brand, model, battery, range, etc.).
- Visualize market trends, adoption, and pricing using rich, interactive analytics.
- Get actionable answers to EV-related sales, pricing, and technical questions through a GenAI chatbot.

All modules use only data from the "EV Sales and Adoption Dataset."

---

## Features

- Sales Forecasting:  
  Time series analysis and regression models to forecast future EV sales by region, brand, or model.

- Price Prediction:  
  ML models trained to estimate EV prices from technical features.

- Exploratory Data Analysis (EDA):  
  Interactive graphs revealing relationships between EV specs, sales, and pricing.

- GenAI Chatbot:  
  Answers natural language queries on EV market, price prediction, model comparisons, and more by leveraging insights from the dataset.

- Unified Web Dashboard:  
  Single interface—built with Streamlit—combining chatbot and analytics for end-user convenience.

---

## Dataset Used

EV Sales and Adoption Dataset

- Contains: Model, brand, price, region, sales volumes (over time), vehicle specs (battery, range, acceleration), and market segments.
- Use cases: Sales forecasting, adoption analytics, price prediction, market insight generation.
- Source: Kaggle - Electric Vehicle (EV) Sales and Adoption (https://www.kaggle.com/datasets/rameezmeerasahib/electric-vehicle-ev-sales-and-adoption)

---

## Project File Structure

```

EVisionAI/
│
├── data/ # Store the EV Sales and Adoption Dataset here
│ └── ev_sales_adoption.csv
│
├── notebooks/ # Jupyter notebooks for EDA, modeling, experimentation
│ └── 01_eda.ipynb
│ └── 02_model_training.ipynb
│
├── src/ # Source code for backend logic, ML, and chatbot
│ ├── init.py
│ ├── data_loader.py # Functions to load and preprocess dataset
│ ├── eda.py # Plotting and EDA functions
│ ├── model.py # ML model training, prediction code
│ ├── chatbot.py # Chatbot/GenAI integration and logic
│ └── utils.py # Utility functions
│
├── app.py # Main Streamlit app/dashboard code
├── requirements.txt # Python dependencies
├── README.md # Project overview and instructions
│
├── assets/ # Images, logos, static files for UI
│ └── img.png
│
├── .gitignore # Files/folders to ignore in git

```

---

## Usage Examples

- Chatbot:  
  “What is the average price of EVs sold in 2023?”  
  “Which model saw the highest sales growth last year?”  
  “Forecast EV sales for Hyundai in California for next quarter.”  
  “How does battery capacity relate to price?”

- Analytics Dashboard:
- Interactive sales/price trend graphs
- EDA plots by brand, region, year
- Model-wise price predictions

---

## Technologies

- Python (pandas, scikit-learn, numpy)
- Streamlit (for dashboard UI)
- OpenAI or Gemini API (for GenAI chatbot, if used)
- matplotlib, seaborn, plotly (data visualization)

---

## Week 1 Status

✅ Week 1 tasks are complete:

- Dataset selected and reviewed.
- Problem statement defined and documented.
- Initial project structure and objectives finalized.

---
