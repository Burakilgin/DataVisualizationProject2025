import streamlit as st
from data_loader import load_data, CATEGORICAL_COLS, NUMERIC_COLS, REGION_COLS
import charts

#Run File

st.set_page_config(
    page_title="Video Game Sales Dashboard",
    layout="wide",
)

df = load_data()

st.title("Video Game Sales Dashboard")

st.markdown("-------")

col_year, col_sales, col_violin, col_height = st.columns([2, 1, 1, 1])

with col_year:
    if "Year" in df.columns and not df["Year"].isna().all():
        min_year = int(df["Year"].min())
        max_year = int(df["Year"].max())
        start_year, end_year = st.slider(
            "Year Range",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
            step=1,
        )
        base_df = df[
            (df["Year"] >= start_year) & (df["Year"] <= end_year)
        ]
    else:
        base_df = df.copy()

with col_sales:
    selected_sales_col = st.selectbox(
        "Sales Column",
        NUMERIC_COLS,
        index=4,
    )

with col_violin:
    selected_category_for_violin = st.selectbox(
        "Violin Category",
        CATEGORICAL_COLS,
        index=1,
    )

with col_height:
    chart_height = st.slider(
        "Chart Height (px)",
        min_value=300,
        max_value=800,
        value=450,
        step=50,
    )

st.markdown("---")

if base_df.empty:
    st.error("There is no result for selected year range. Please change filters.")
    st.stop()

tab_overview, tab_dist, tab_networks, tab_geo, tab_details = st.tabs(
    ["Overview", "Distributions", "Networks", "Geography", "Details"]
)

with tab_overview:
    charts.render_overview_tab(base_df, chart_height)

with tab_dist:
    charts.render_distributions_tab(
        base_df,
        selected_sales_col=selected_sales_col,
        selected_category_for_violin=selected_category_for_violin,
        chart_height=chart_height,
    )

with tab_networks:
    charts.render_networks_tab(base_df, chart_height)

with tab_geo:
    charts.render_geography_tab(base_df, chart_height)

with tab_details:
    charts.render_details_tab(base_df)