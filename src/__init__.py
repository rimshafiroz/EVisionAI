"""
EVisionAI - EV Sales & Adoption Analytics Platform
Source code package for data loading, EDA, modeling, and chatbot functionality.
"""

__version__ = "1.0.0"

from .data_loader import load_data, preprocess_data
from .eda import plot_correlation, plot_sales_by_brand, plot_price_distribution
from .model import train_price_model, forecast_sales
from .chatbot import chatbot

__all__ = [
    "load_data",
    "preprocess_data",
    "plot_correlation",
    "plot_sales_by_brand",
    "plot_price_distribution",
    "train_price_model",
    "forecast_sales",
    "chatbot",
]

