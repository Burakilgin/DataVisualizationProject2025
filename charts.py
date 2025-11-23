import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

from data_loader import CATEGORICAL_COLS, NUMERIC_COLS, REGION_COLS

# Functions File

def apply_chart_filters(
    base_df: pd.DataFrame,
    prefix: str,
    allowed_cols: list[str],
    allowed_regions: list[str] | None = None,
    label: str = "Filters",
) -> pd.DataFrame:
    filtered = base_df.copy()
    any_selection = False

    with st.expander(label):
        # Categorical Filters
        for col in allowed_cols:
            if col in filtered.columns:
                unique_vals = sorted(filtered[col].dropna().unique())
                selected_vals = st.multiselect(
                    label=f"{col}",
                    options=unique_vals,
                    key=f"{prefix}_{col}",
                )
                if selected_vals:
                    any_selection = True
                    filtered = filtered[filtered[col].isin(selected_vals)]

        # Region Filters
        if allowed_regions:
            region_selected = st.multiselect(
                label="Regions (NA/EU/JP/Other)",
                options=allowed_regions,
                key=f"{prefix}_regions",
            )
            if region_selected:
                any_selection = True
                mask = None
                for rcol in region_selected:
                    if rcol in filtered.columns:
                        cond = filtered[rcol] > 0
                        if mask is None:
                            mask = cond
                        else:
                            mask = mask | cond
                if mask is not None:
                    filtered = filtered[mask]

    # No selections
    if not any_selection:
        filtered = filtered.iloc[0:0]

    return filtered


# OVERVIEW TAB


def render_overview_tab(base_df: pd.DataFrame, chart_height: int):
    st.subheader("Summary & High-Level View")

    n_games = len(base_df)
    avg_global = base_df["Global_Sales"].mean()
    total_global = base_df["Global_Sales"].sum()
    top_Genre = (
        base_df["Genre"].value_counts().idxmax()
        if "Genre" in base_df.columns and not base_df["Genre"].empty
        else None
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Number of Games", n_games)

    with col2:
        st.metric("Total Global Sales", f"{total_global:.1f}")

    with col3:
        st.metric("Avg Global Sales/Game", f"{avg_global:.2f}")

    with col4:
        if top_Genre:
            st.metric("Top Genre", top_Genre)
        else:
            st.metric("Top Genre", "-")

    st.markdown("# Glyph-Based Scatter Plot")

    glyph_df = apply_chart_filters(
        base_df,
        prefix="glyph",
        allowed_cols=CATEGORICAL_COLS,
        allowed_regions=REGION_COLS,
        label="Filters – Glyph Scatter",
    )

    if glyph_df.empty:
        st.info("No data for selected filters (Glyph Scatter Graph).")
    else:
        fig_glyph = px.scatter(
            glyph_df,
            x="Year",
            y="Global_Sales",
            color="Genre",
            symbol="Platform",
            size="Global_Sales",
            hover_name="Name",
            opacity=0.7,
            title="Sales Distribution Over Years by Genre & Platform",
        )
        fig_glyph.update_layout(
            height=chart_height,
            xaxis_title="Year",
            yaxis_title="Global Sales",
        )
        st.plotly_chart(fig_glyph, use_container_width=True)


# DISTRIBUTIONS TAB

def render_distributions_tab(
    base_df: pd.DataFrame,
    selected_sales_col: str,
    selected_category_for_violin: str,
    chart_height: int,
):
    st.subheader("Distributions & Composition")

    # Histogram + Correlation Heatmap
    col_a, col_b = st.columns([2, 1])

    # Histogram
    with col_a:
        st.markdown(f"#### {selected_sales_col} Distribution")

        hist_df = apply_chart_filters(
            base_df,
            prefix="hist",
            allowed_cols=CATEGORICAL_COLS,
            allowed_regions=REGION_COLS,
            label="Filters – Histogram",
        )

        if hist_df.empty:
            st.info("No data for selected filters (Histogram Graph).")
        else:
            fig_hist = px.histogram(
                hist_df,
                x=selected_sales_col,
                nbins=30,
                marginal="box",
                title=f"Distribution of {selected_sales_col} (Histogram)",
                opacity=0.7,    # Hafif şeffaflık
                color_discrete_sequence=['#3366CC'] # Şık bir mavi ton
            )

            # Barların arasına ince çizgi
            fig_hist.update_traces(marker_line_width=1, marker_line_color="white")

            fig_hist.update_layout(
                height=chart_height,
                xaxis_title=f"{selected_sales_col}",
                yaxis_title="Frequency (Count)",
                bargap=0.2 # Sütunlar arasına hafif boşluk
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    # Correlation Heatmap
    with col_b:
        st.markdown("#### Sales Correlations")

        corr_df = apply_chart_filters(
            base_df,
            prefix="corr_heatmap",
            allowed_cols=CATEGORICAL_COLS,
            allowed_regions=REGION_COLS,
            label="Filters – Correlation Heatmap",
        )

        if corr_df.empty:
            st.info("No data for selected filters (Correlation Heatmap Graph).")
        else:
            corr = corr_df[NUMERIC_COLS].corr()
            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
            )
            fig_corr.update_layout(
                height=chart_height,
                xaxis_title="Sales Metrics",
                yaxis_title="Sales Metrics",
            )
            st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")

    # Violin + Treemap/Bar
    col_v, col_t = st.columns(2)

    # Violin
    with col_v:
        st.markdown(f"#### {selected_sales_col} vs {selected_category_for_violin}")

        violin_df = apply_chart_filters(
            base_df,
            prefix="violin",
            allowed_cols=[selected_category_for_violin],
            allowed_regions=REGION_COLS,
            label=f"Filters – Violin ({selected_category_for_violin})",
        )

        if violin_df.empty:
            st.info("No data for selected filters (Violin Plot Graph).")
        else:
            fig_violin = px.violin(
                violin_df,
                x=selected_category_for_violin,
                y=selected_sales_col,
                box=True,
                points="outliers",
            )
            fig_violin.update_layout(
                height=chart_height,
                xaxis_title=selected_category_for_violin,
                yaxis_title=f"{selected_sales_col} (M)",
            )
            st.plotly_chart(fig_violin, use_container_width=True)

    # Treemap + Bar
    with col_t:
        st.markdown("#### Global Sales (Treemap & Genre Bar)")

        treemap_bar_df = apply_chart_filters(
            base_df,
            prefix="treemap_bar",
            allowed_cols=["Genre", "Platform"],
            allowed_regions=REGION_COLS,
            label="Filters – Treemap & Genre Bar",
        )

        if treemap_bar_df.empty:
            st.info("No data for selected filters (Treemap & Bar Graph).")
        else:
            # Treemap
            treemap_df = (
                treemap_bar_df
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
            fig_tree.update_layout(height=int(chart_height * 0.9))
            st.plotly_chart(fig_tree, use_container_width=True)

            # Bar
            st.markdown("##### Global Sales by Genre")

            bar_df = (
                treemap_bar_df
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

            fig_bar.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig_bar.update_layout(
                height=int(chart_height * 0.9),
                xaxis_tickangle=45,
                xaxis_title="Genre",
                yaxis_title="Total Global Sales",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # Parallel Coordinates
    st.markdown("#### Parallel Coordinates – Sales Across Regions")

    parallel_df = apply_chart_filters(
        base_df,
        prefix="parallel",
        allowed_cols=CATEGORICAL_COLS,
        allowed_regions=REGION_COLS,
        label="Filters – Parallel Coordinates",
    )

    if parallel_df.empty:
        st.info("No data for selected filters (Parallel Coordinates Graph).")
    else:
        max_rows_for_parallel = 300
        if len(parallel_df) > max_rows_for_parallel:
            parallel_df = parallel_df.sample(max_rows_for_parallel, random_state=42)

        fig_parallel = px.parallel_coordinates(
            parallel_df,
            dimensions=NUMERIC_COLS,
        )
        fig_parallel.update_layout(
            height=chart_height,
        )
        st.plotly_chart(fig_parallel, use_container_width=True)


# NETWORKS TAB


def render_networks_tab(base_df: pd.DataFrame, chart_height: int):
    st.subheader("Network Views & Flows")

    # Correlation Network
    st.markdown("#### Correlation Network Graph")

    corr_net_df = apply_chart_filters(
        base_df,
        prefix="corr_network",
        allowed_cols=CATEGORICAL_COLS,
        allowed_regions=REGION_COLS,
        label="Filters – Correlation Network",
    )

    if corr_net_df.empty:
        st.info("No data for selected filters (Correlation Network Graph).")
    else:
        corr_net = corr_net_df[NUMERIC_COLS].corr()

        edges = []
        threshold = 0.5
        for col1 in corr_net.columns:
            for col2 in corr_net.columns:
                if col1 < col2:
                    value = corr_net.loc[col1, col2]
                    if abs(value) >= threshold:
                        edges.append((col1, col2, value))

        if edges:
            G = nx.Graph()
            for col1, col2, weight in edges:
                G.add_edge(col1, col2, weight=weight)

            pos = nx.spring_layout(G, seed=42)

            nx_fig = px.scatter(
                x=[pos[node][0] for node in G.nodes()],
                y=[pos[node][1] for node in G.nodes()],
                text=list(G.nodes()),
                labels={
                    "x": "Node position X",
                    "y": "Node position Y",
                },
            )

            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                nx_fig.add_shape(
                    type="line",
                    x0=x0, y0=y0,
                    x1=x1, y1=y1,
                    line=dict(width=2),
                )

            nx_fig.update_layout(
                height=chart_height,
                xaxis_title="Node position X",
                yaxis_title="Node position Y",
            )
            st.plotly_chart(nx_fig, use_container_width=True)
        else:
            st.info("No strong correlations for selected filters.")

    st.markdown("---")

    # Sankey
    st.markdown("### Sankey Diagram – Genre → Regional Sales")

    sankey_src_df = apply_chart_filters(
        base_df,
        prefix="sankey",
        allowed_cols=["Genre"],
        allowed_regions=REGION_COLS,
        label="Filters – Sankey Diagram",
    )

    selected_regions = st.session_state.get("sankey_regions", [])

    if sankey_src_df.empty:
        st.info("No data for selected filters (Sankey Diagram Graph).")
    else:
        base_region_cols = selected_regions or []
        sankey_numeric_cols = ["Global_Sales"] + base_region_cols

        sankey_df = (
            sankey_src_df
            .groupby("Genre")[sankey_numeric_cols]
            .sum()
            .reset_index()
        )

        nodes = list(sankey_df["Genre"]) + sankey_numeric_cols
        node_indices = {name: i for i, name in enumerate(nodes)}

        source = []
        target = []
        value = []

        for _, row in sankey_df.iterrows():
            cat = row["Genre"]
            for sales_col in sankey_numeric_cols:
                source.append(node_indices[cat])
                target.append(node_indices[sales_col])
                value.append(row[sales_col])

        fig_sankey = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="green", width=1),
                        label=nodes,
                        color="blue",
                    ),
                    link=dict(
                        source=source,
                        target=target,
                        value=value,
                    ),
                )
            ]
        )

        fig_sankey.update_layout(
            title_text="Genre to Regional Sales Flow",
            font_size=20,
            height=chart_height,
        )
        st.plotly_chart(fig_sankey, use_container_width=True)


# GEOGRAPHY TAB

def render_geography_tab(base_df: pd.DataFrame, chart_height: int):
    st.subheader("Geographical View")

    geo_src_df = apply_chart_filters(
        base_df,
        prefix="geo",
        allowed_cols=CATEGORICAL_COLS,
        allowed_regions=REGION_COLS,
        label="Filters – Geographic Chart",
    )

    if geo_src_df.empty:
        st.info("No data for selected filters (Geographic Chart Graph)")
    else:
        na_sales_val = geo_src_df["NA_Sales"].sum()
        eu_sales_val = geo_src_df["EU_Sales"].sum()
        jp_sales_val = geo_src_df["JP_Sales"].sum()
        other_sales_val = geo_src_df["Other_Sales"].sum()

        na_countries = ["United States", "Canada", "Mexico", "Greenland"]
        eu_countries = [
            "France", "Germany", "United Kingdom", "Italy", "Spain", "Poland",
            "Sweden", "Netherlands", "Norway", "Finland", "Belgium", "Austria",
            "Switzerland", "Portugal", "Greece", "Ireland", "Denmark",
            "Czech Republic", "Hungary", "Romania", "Russia", "Ukraine", "Turkey",
        ]
        other_countries = [
            "China", "India", "Australia", "Brazil", "Argentina", "Chile",
            "South Africa", "Egypt", "Saudi Arabia", "South Korea", "Indonesia",
            "Thailand", "New Zealand", "Colombia",
        ]

        map_data = []

        for c in na_countries:
            map_data.append({"Country": c, "Sales": na_sales_val, "Region": "North America"})

        for c in eu_countries:
            map_data.append({"Country": c, "Sales": eu_sales_val, "Region": "Europe"})

        map_data.append({"Country": "Japan", "Sales": jp_sales_val, "Region": "Japan"})

        for c in other_countries:
            map_data.append({"Country": c, "Sales": other_sales_val, "Region": "Other Regions"})

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

        fig_geo.update_layout(height=chart_height)
        st.plotly_chart(fig_geo, use_container_width=True)


# DETAILS TAB

def render_details_tab(base_df: pd.DataFrame):
    st.subheader("Detailed Data View")

    table_df = apply_chart_filters(
        base_df,
        prefix="table",
        allowed_cols=CATEGORICAL_COLS,
        allowed_regions=REGION_COLS,
        label="Filters – Table",
    )

    if table_df.empty:
        st.info("No data for selected filters.")
    else:
        st.dataframe(
            table_df,
            use_container_width=True,
            height=500,
        )
