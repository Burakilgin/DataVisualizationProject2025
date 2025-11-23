from pathlib import Path
import pandas as pd
import streamlit as st

#Data Loader File

CATEGORICAL_COLS = [
    "Platform",
    "Genre",
    "Publisher",
]

NUMERIC_COLS = [
    "NA_Sales",
    "EU_Sales",
    "JP_Sales",
    "Other_Sales",
    "Global_Sales",
]

REGION_COLS = [
    "NA_Sales",
    "EU_Sales",
    "JP_Sales",
    "Other_Sales",
]


@st.cache_data
def load_data(csv_path: str = "vgsales.csv") -> pd.DataFrame:

    path = Path(csv_path)

    df = pd.read_csv(path)

    if "Year" in df.columns:
        df = df.dropna(subset=["Year"])
        df["Year"] = df["Year"].astype(int)

    return df