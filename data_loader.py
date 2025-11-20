# data_loader.py
from pathlib import Path
import pandas as pd
import streamlit as st

#Data Loader File

# Bu sabitleri buraya taşıyoruz
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
    """
    CSV dosyasını okuyup, Year kolonunu temizleyip döner.
    Mevcut main.py içindeki load_data fonksiyonunu buraya aynen taşıyabilirsin.
    """
    path = Path(csv_path)
    if not path.exists():
        st.error(f"CSV file does not find: {path.resolve()}")
        st.stop()

    df = pd.read_csv(path)

    if "Year" in df.columns:
        df = df.dropna(subset=["Year"])
        df["Year"] = df["Year"].astype(int)

    return df