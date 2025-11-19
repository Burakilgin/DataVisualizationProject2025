import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from pathlib import Path


# Header Ayarları

st.set_page_config(
    page_title="Video Game Sales Dashboard",
    layout="wide",
)

# Verisetinin Yüklenmesi

@st.cache_data
def load_data(csv_path: str = "vgsales.csv") -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        st.error(f"CSV file does not find: {path.resolve()}")
        st.stop()

    df = pd.read_csv(path)

    if "Year" in df.columns:

        df = df.dropna(subset=["Year"])
        df["Year"] = df["Year"].astype(int)

    return df


df = load_data()

#Kolon Listesi

categorical_cols = [
    "Platform",
    "Genre",
    "Publisher",
]

numeric_cols = [
    "NA_Sales",
    "EU_Sales",
    "JP_Sales",
    "Other_Sales",
    "Global_Sales",
]

# Sidebar Design

st.sidebar.header("Filters")

filtered_df = df.copy()

# Yıl filtresi
if "Year" in df.columns and not df["Year"].isna().all():
    min_year = int(df["Year"].min())
    max_year = int(df["Year"].max())
    start_year, end_year = st.sidebar.slider(
        "Year Range:",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1,
    )
    filtered_df = filtered_df[
        (filtered_df["Year"] >= start_year) & (filtered_df["Year"] <= end_year)
    ]

# Multiple Filtering
for col in categorical_cols:
    if col in filtered_df.columns:
        unique_vals = sorted(filtered_df[col].dropna().unique())
        selected_vals = st.sidebar.multiselect(
            label=f"{col}",
            options=unique_vals,
            default=unique_vals,
        )
        filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]

# Sales Column Selection
selected_sales_col = st.sidebar.selectbox(
    "Sales Column (histogram):",
    numeric_cols,
    index=4,  #Default Value for Global Sales
)

# Choosing category for Violin Graph
selected_category_for_violin = st.sidebar.selectbox(
    "Categories (violin):",
    categorical_cols,
    index=1,  # Genre varsayılan
)

st.sidebar.markdown("---")

# Başlık ve Veriseti Hakkında Açıklama

st.title("Video Game Sales")

st.markdown("---")

if filtered_df.empty:
    st.error("There is no result.Please add new filters from the list.")
    st.stop()



st.subheader("Summary Statistics")

n_games = len(filtered_df)
avg_global = filtered_df["Global_Sales"].mean()
total_global = filtered_df["Global_Sales"].sum()
top_genre = (
    filtered_df["Genre"].value_counts().idxmax()
    if "Genre" in filtered_df.columns and not filtered_df["Genre"].empty
    else None
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Number of Game", n_games)

with col2:
    st.metric("Total Sales on Global (M)", f"{total_global:.1f}")

with col3:
    st.metric("Approximately Sales on Global (M)", f"{avg_global:.2f}")

with col4:
    if top_genre:
        st.metric("Most Popular Type", top_genre)
    else:
        st.metric("Most Popular Type", "-")


# Main Graphics

st.markdown("### Sales Distributions")

col_a, col_b = st.columns([2, 1])

# Histogram Graph
with col_a:
    st.markdown(f"# {selected_sales_col} Distrubition")

    fig_hist = px.histogram(
        filtered_df,
        x=selected_sales_col,
        nbins=30,
        marginal="box",
    )
    fig_hist.update_layout(height=400, xaxis_title=f"{selected_sales_col} (M)")
    st.plotly_chart(fig_hist, width="stretch")

# Korelasyon Heatmap (bölgesel satışlar arası)
with col_b:
    st.markdown("#### Sales Correlations")

    corr = filtered_df[numeric_cols].corr()
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
    )
    fig_corr.update_layout(height=400)
    st.plotly_chart(fig_corr, width="stretch")

# Middle Area: Violin and Treemap

st.markdown("### Distribution and Composition")

col_v, col_t = st.columns(2)

#Burak
# Violin Graph
with col_v:
    st.markdown(f"# {selected_sales_col} vs {selected_category_for_violin}")

    fig_violin = px.violin(
        filtered_df,
        x=selected_category_for_violin,
        y=selected_sales_col,
        box=True,
        points="outliers",
    )
    fig_violin.update_layout(
        height=450,
        xaxis_title=selected_category_for_violin,
        yaxis_title=f"{selected_sales_col} (M)",
    )
    st.plotly_chart(fig_violin, width="stretch")

#Burak
# Treemap Graph
with col_t:
    st.markdown("# Global Sales")

    treemap_df = (
        filtered_df
        .groupby(["Genre", "Platform"])["Global_Sales"]
        .sum()
        .reset_index()
        .rename(columns={"Global_Sales": "Total_Global_Sales"})
    )

    fig_tree = px.treemap(
        treemap_df,
        path=["Genre", "Platform"],
        values="Total_Global_Sales",
    )
    fig_tree.update_layout(height=450)
    st.plotly_chart(fig_tree, width="stretch")

#Burak
# Paralell Coordinates Graph

st.markdown("### Parallel Coordinates – Sales Across Regions")

parallel_df = filtered_df.copy()

max_rows_for_parallel = 300
if len(parallel_df) > max_rows_for_parallel:
    parallel_df = parallel_df.sample(max_rows_for_parallel, random_state=42)


fig_parallel = px.parallel_coordinates(
    parallel_df,
    dimensions=numeric_cols,
)

fig_parallel.update_layout(height=450)
st.plotly_chart(fig_parallel, width="stretch")


# Bu kısma kodlarınızı yukarıdaki grafiklerin kodları gibi ekleyebilrsiniz.

# Meltem
# Correlation Network Graph
st.markdown("### Correlation Network Graph")

corr = filtered_df[numeric_cols].corr()

edges = []
threshold = 0.5  # Hata burada çözüldü: Değişken tanımlandı.
for col1 in corr.columns:
    for col2 in corr.columns:
        if col1 < col2:
            value = corr.loc[col1, col2]
            if abs(value) >= threshold:
                edges.append((col1, col2, value))

# Network graph oluşturma
if edges:
    G = nx.Graph()
    for col1, col2, weight in edges:
        G.add_edge(col1, col2, weight=weight)

    pos = nx.spring_layout(G, seed=42)
    nx_fig = px.scatter(
        x=[pos[node][0] for node in G.nodes()],
        y=[pos[node][1] for node in G.nodes()],
        text=list(G.nodes()),
        labels={"x": "", "y": ""},
    )

    # Kenarları çiz
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        nx_fig.add_shape(
            type="line",
            x0=x0, y0=y0,
            x1=x1, y1=y1,
            line=dict(width=2),
        )

    nx_fig.update_layout(height=450)
    st.plotly_chart(nx_fig, use_container_width=True)
else:
    st.info("No strong correlations (>|0.5|).")

# Meltem
# Genre–Platform Network Graph
st.markdown("### Genre–Platform Network Graph")

network_df = (
    filtered_df.groupby(["Genre", "Platform"])["Global_Sales"]
    .sum()
    .reset_index()
)

G2 = nx.Graph()

# Node ekleme
for _, row in network_df.iterrows():
    genre = row["Genre"]
    platform = row["Platform"]
    weight = row["Global_Sales"]

    G2.add_node(genre, type="Genre")
    G2.add_node(platform, type="Platform")
    G2.add_edge(genre, platform, weight=weight)

pos2 = nx.spring_layout(G2, seed=42, k=1.5)

nx_fig2 = px.scatter(
    x=[pos2[node][0] for node in G2.nodes()],
    y=[pos2[node][1] for node in G2.nodes()],
    text=list(G2.nodes()),
    labels={"x": "", "y": ""},
)

# Kenarları çiz
for edge in G2.edges():
    x0, y0 = pos2[edge[0]]
    x1, y1 = pos2[edge[1]]
    nx_fig2.add_shape(
        type="line",
        x0=x0, y0=y0,
        x1=x1, y1=y1,
        line=dict(width=1),
    )

nx_fig2.update_layout(height=500)
st.plotly_chart(nx_fig2, use_container_width=True)

# Meltem
# Sankey Diagram - DÜZELTİLMİŞ VERSİYON (px.sankey yerine go.Figure)
st.markdown("### Sankey Diagram – Regional Sales Flow")

sankey_df = filtered_df.groupby("Genre")[numeric_cols].sum().reset_index()

# Sankey nodes
nodes = list(sankey_df["Genre"]) + numeric_cols
node_indices = {name: i for i, name in enumerate(nodes)}

# Sankey links
source = []
target = []
value = []

for _, row in sankey_df.iterrows():
    genre = row["Genre"]
    for sales_col in numeric_cols:
        source.append(node_indices[genre])
        target.append(node_indices[sales_col])
        value.append(row[sales_col])

# Plotly Graph Objects ile çiziyoruz
fig_sankey = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=nodes,
        color="blue"
    ),
    link=dict(
        source=source,
        target=target,
        value=value
    )
)])

fig_sankey.update_layout(title_text="Genre to Regional Sales Flow", font_size=10, height=500)
st.plotly_chart(fig_sankey, use_container_width=True)

#MELISA
# Bar Chart – Global Sales by Genre
with col_t:
    st.markdown("### Global Sales by Genre")

    bar_df = (
        filtered_df
        .groupby("Genre")["Global_Sales"]
        .sum()
        .reset_index()
        .rename(columns={"Global_Sales": "Total_Global_Sales"})
        .sort_values("Total_Global_Sales", ascending=False)
    )

    fig_bar = px.bar(
        bar_df,
        x="Genre",
        y="Total_Global_Sales",
        text="Total_Global_Sales",
    )

    fig_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_bar.update_layout(height=450, xaxis_tickangle=45)
    st.plotly_chart(fig_bar, use_container_width=True)


#MELISA
# Geographic Chart – Global Sales by Region
st.markdown("### Global Sales by Region (Continents)")

na_sales_val = filtered_df["NA_Sales"].sum()
eu_sales_val = filtered_df["EU_Sales"].sum()
jp_sales_val = filtered_df["JP_Sales"].sum()
other_sales_val = filtered_df["Other_Sales"].sum()

na_countries = ["United States", "Canada", "Mexico", "Greenland"]
eu_countries = [
    "France", "Germany", "United Kingdom", "Italy", "Spain", "Poland", "Sweden", "Netherlands",
    "Norway", "Finland", "Belgium", "Austria", "Switzerland", "Portugal", "Greece", "Ireland",
    "Denmark", "Czech Republic", "Hungary", "Romania", "Russia", "Ukraine", "Turkey"
]
other_countries = [
    "China", "India", "Australia", "Brazil", "Argentina", "Chile", "South Africa", "Egypt",
    "Saudi Arabia", "South Korea", "Indonesia", "Thailand", "New Zealand", "Colombia"
]

map_data = []

# Kuzey Amerika'yı Boya
for c in na_countries:
    map_data.append({'Country': c, 'Sales': na_sales_val, 'Region': 'North America'})

# Avrupa'yı Boya
for c in eu_countries:
    map_data.append({'Country': c, 'Sales': eu_sales_val, 'Region': 'Europe'})

# Japonya'yı Boya
map_data.append({'Country': 'Japan', 'Sales': jp_sales_val, 'Region': 'Japan'})

# Diğer Kıtaları Boya
for c in other_countries:
    map_data.append({'Country': c, 'Sales': other_sales_val, 'Region': 'Other Regions'})

geo_df_continents = pd.DataFrame(map_data)

fig_geo = px.choropleth(
    geo_df_continents,
    locations="Country",
    locationmode="country names",
    color="Sales",
    hover_name="Region",
    color_continuous_scale="Plasma",
    title="Regional Sales (Continents Colored)",
)

fig_geo.update_layout(height=450)
st.plotly_chart(fig_geo, use_container_width=True)


# 3. Glyph-Based Chart – Critic Score YERİNE YEAR Kullanıldı
st.markdown("### Glyph-Based Scatter (Year vs Global Sales)")

glyph_df = filtered_df.copy()

fig_glyph = px.scatter(
    glyph_df,
    x="Year",
    y="Global_Sales",
    color="Genre",
    symbol="Platform",
    size="Global_Sales",
    hover_name="Name",
    opacity=0.7,
    title="Sales Distribution Over Years by Genre & Platform"
)

fig_glyph.update_layout(height=450)
st.plotly_chart(fig_glyph, use_container_width=True)








# Main Dataset Table

st.markdown("### Main Dataset Table")

st.dataframe(
    filtered_df,
    width="stretch",
    height=500,
)
