
# Introduction to Data Visualization Project

## Video Game Sales Dashboard

Interactive Streamlit dashboard for exploring the **Video Game Sales** dataset (Dataset file is in the main folder as a "vgsales.csv").
The app provides multiple coordinated views (Summary&Overview, Distributions, Networks, Geography, and Details) with reusable filters for genres, platforms, publishers, and sales regions.

---

## 1. Presentation of Dataset

This dataeset includes the video games sales in the world between 1980-2020. It has more than 20+ Publisher and Game Categories. Dataset downloaded from Kaggle Datasets. 
Dataset Link = https://www.kaggle.com/datasets/gregorut/videogamesales

The dataset includes: 
- 16.598 rows 
- 578 Publisher
- 12 Genre (Categories) 


The goal of this project is to:

- To understanding the data visualization
- Provide multiple complementary visualizations (scatter, histograms, treemaps, networks, map, etc.).
- Allow the user to interactively filtering dataset on the dataset

The dashboard is implemented as a **modular Streamlit app**:

- `app.py` handles the layout, global controls, and tab structure.
- `charts.py` contains all visualization logic and tab render functions.
- `data_loader.py` is responsible for loading the dataset and defining column groups.

---

## 2. Tech Stack

- **Python 3.13**
- **Streamlit** – Web UI Framework
- **Pandas** – Data Manipulation
- **Plotly Express / Graph Objects** – Interactive Charts
- **NetworkX** – Graph/Network Layouts

To see the WebUI on the local computer:

- Clone the repository: 

```
git clone https://github.com/Burakilgin/DataVisualizationProject2025
```

- Install all dependencies with:

```
pip install -r requirements.txt
```

- Run the app system with the code below: 

```
streamlit run app.py
```

---

## 3. Chart Creation Structure
1. Melisa Akyol
    - Histogram Graph
    - Geopgraphic Graph
    - Glyph Based Graph

2. Meltem Yılmaz
    - Heat Graph
    - Sankey Graph
    - Network Graph

3. Burak Ilgın
    - Violin Graph
    - Tree Map Graph
    - Paralel Coordinates Graph


---

## Functions of Charts

- Every chart function created from the owner of chart part as like as Chart Creation structure. In the project, charts can be seem mixed because to make them more understandable we combined more than one chart like Glyph chart with Scatter Plot Graphs. 

- The based code system was created by Burak Ilgın. Function controlls, creating and designing system created from Meltem Yılmaz and Melisa Akyol. 

- End section of the Streamlit app you can see the **Raw Data**. We did that can be filterable because when someone who wants to compare the graphics with the raw data, when the select the same filters from the filter section, will be able to see the raw data. 

