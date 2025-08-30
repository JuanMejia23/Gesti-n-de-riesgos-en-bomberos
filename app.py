streamlit
pandas
numpy
plotly
pydeck
networkx

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pydeck as pdk
import networkx as nx

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Panel de Control de Riesgos",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Estilos CSS Personalizados para un look más profesional ---
st.markdown("""
<style>
    .stDeployButton {
        visibility: hidden;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stMetric {
        border: 1px solid #444;
        border-radius: 10px;
        padding: 1rem;
        background-color: #1E1E1E;
    }
    .stPlotlyChart {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# --- Funciones para Generar Datos Simulados ---

def generate_mock_data():
    """Genera datos simulados para el mapa y las métricas."""
    # Coordenadas base (Bogotá, Colombia)
    lat_base, lon_base = 4.60971, -74.08175

    # Generar puntos de riesgo aleatorios alrededor de la base
    data = pd.DataFrame({
        'lat': lat_base + np.random.randn(500) * 0.05,
        'lon': lon_base + np.random.randn(500) * 0.05,
        'risk_level': np.random.randint(1, 100, 500)
    })
    
    # Simular ubicaciones de vehículos de emergencia y un incidente
    vehicles = pd.DataFrame({
        'lat': lat_base + np.random.uniform(-0.1, 0.1, 3),
        'lon': lon_base + np.random.uniform(-0.1, 0.1, 3),
        'tooltip': ['Unidad 101', 'Ambulancia 52', 'Bomberos 08']
    })
    
    incident = pd.DataFrame({
        'lat': [lat_base + np.random.uniform(-0.05, 0.05)],
        'lon': [lon_base + np.random.uniform(-0.05, 0.05)],
        'tooltip': ['Incidente Activo']
    })
    
    return data, vehicles, incident

def create_gauge(value, title, max_val=100, color="green"):
    """Crea un gráfico de medidor (gauge) con Plotly."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': title, 'font': {'size': 18}},
        gauge = {
            'axis': {'range': [None, max_val], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#444",
            'steps' : [
                {'range': [0, max_val * 0.5], 'color': 'green'},
                {'range': [max_val * 0.5, max_val * 0.8], 'color': 'orange'},
                {'range': [max_val * 0.8, max_val], 'color': 'red'}],
        }))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor='rgba(0,0,0,0)', 
        font={'color': 'white'},
        height=250
    )
    return fig

def create_routing_graph():
    """Crea un grafo y calcula la ruta más corta para la visualización."""
    G = nx.erdos_renyi_graph(15, 0.3, seed=42)
    pos = nx.spring_layout(G, seed=42)
    
    # Asignar pesos aleatorios a las aristas
    for (u, v) in G.edges():
        G.edges[u,v]['weight'] = np.random.randint(1, 10)

    source, target = 0, 10
    shortest_path = nx.shortest_path(G, source=source, target=target, weight='weight')
    
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    
    path_x = [pos[node][0] for node in shortest_path]
    path_y = [pos[node][1] for node in shortest_path]

    # Crear la figura Plotly
    fig = go.Figure()
    # Aristas del grafo
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines'))
    # Nodos del grafo
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', marker=dict(
        showscale=True, colorscale='YlGnBu', reversescale=True, color=[], size=12,
        colorbar=dict(thickness=15, title='Node Connections', xanchor='left', titleside='right'),
        line_width=2)))
    # Ruta más corta
    fig.add_trace(go.Scatter(x=path_x, y=path_y, line=dict(width=4, color='red'), name='Ruta Óptima', mode='lines+markers'))
    
    fig.update_layout(
        title={'text': "Algoritmo de Ruta Más Corta (Teoría de Grafos)", 'font': {'color': 'white'}},
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor='#1E1E1E',
        font={'color': 'white'}
    )
    return fig

# --- Carga de Datos ---
risk_data, vehicle_locations, incident_location = generate_mock_data()

# --- Interfaz de Usuario ---
st.markdown("<h1 class='main-title'>🚨 Panel de Control de Riesgos Urbanos</h1>", unsafe_allow_html=True)

# --- Fila 1: Monitoreo en vivo y predicciones ---
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("🛰️ Monitoreo Aéreo (Dron)")
    # Placeholder para la imagen del dron
    st.image(
        "https://placehold.co/600x400/000000/FFFFFF?text=Vista+de+Dron",
        caption="Simulación de imagen de dron con superposición de riesgo.",
        use_column_width=True
    )
    st.metric(label="Estado del Dron", value="Activo")
    st.metric(label="Batería", value="82%")


with col2:
    st.subheader("🔮 Predicciones de Probabilidad de Riesgo")
    sub_col1, sub_col2, sub_col3 = st.columns(3)
    with sub_col1:
        st.plotly_chart(create_gauge(np.random.randint(10, 30), "Riesgo de Tráfico", color="orange"), use_container_width=True)
    with sub_col2:
        st.plotly_chart(create_gauge(np.random.randint(5, 20), "Riesgo de Incendio", color="red"), use_container_width=True)
    with sub_col3:
        st.plotly_chart(create_gauge(np.random.randint(20, 50), "Riesgo de Seguridad", color="yellow"), use_container_width=True)

st.divider()

# --- Fila 2: Mapa Principal y Métricas Agregadas ---
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("🗺️ Mapa de Riesgos GIS en Tiempo Real")
    
    view_state = pdk.ViewState(
        latitude=risk_data["lat"].mean(),
        longitude=risk_data["lon"].mean(),
        zoom=11,
        pitch=50,
    )

    heatmap_layer = pdk.Layer(
        'HeatmapLayer',
        data=risk_data,
        get_position='[lon, lat]',
        opacity=0.4,
        get_weight='risk_level'
    )
    
    icon_layer_vehicles = pdk.Layer(
        "IconLayer",
        data=vehicle_locations,
        get_icon="icon_data",
        get_position='[lon, lat]',
        get_size=4,
        size_scale=15,
        icon_atlas="https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/icon-atlas.png",
        icon_mapping={"marker": {"x": 0, "y": 0, "width": 128, "height": 128, "mask": True}}
    )
    # Iconos tomados de la librería de deck.gl
    vehicle_locations['icon_data'] = [{"url": "https://img.icons8.com/plasticine/100/ambulance.png", "width": 128, "height": 128, "anchorY": 128}] * len(vehicle_locations)


    incident_layer = pdk.Layer(
        'IconLayer',
        data=incident_location,
        get_position='[lon, lat]',
        get_icon=lambda r: {'url': 'https://img.icons8.com/fluency/48/siren.png', 'width': 240, 'height': 240, 'anchorY': 240},
        get_size=4,
        size_scale=20,
        pickable=True
    )
    
    r = pdk.Deck(
        layers=[heatmap_layer, incident_layer], # Desactivado temporalmente el icono de vehículos
        initial_view_state=view_state,
        map_style='mapbox://styles/mapbox/dark-v9',
        tooltip={"text": "{tooltip}"}
    )
    st.pydeck_chart(r)

with col2:
    st.subheader("📈 Nivel de Riesgo General")
    
    overall_risk = int(risk_data['risk_level'].mean())
    st.plotly_chart(create_gauge(overall_risk, "Riesgo Total", color="red"), use_container_width=True)

    st.subheader("Tendencias de Riesgo (Última Hora)")
    trend_data = pd.DataFrame({
        'Minuto': range(-60, 0, 5),
        'Nivel de Riesgo': np.random.randint(40, 80, 12)
    })
    st.line_chart(trend_data.set_index('Minuto'), color="#FF4B4B")
    

st.divider()

# --- Fila 3: Ruteo y Tendencias a Largo Plazo ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🚗 Ruteo de Vehículos de Emergencia")
    st.plotly_chart(create_routing_graph(), use_container_width=True)
    st.metric(label="Tiempo Estimado de Llegada", value="12 min 34 seg")


with col2:
    st.subheader("📊 Tendencia de Incidentes (Últimas 24h)")
    incident_trend = pd.DataFrame(
        np.random.randint(0, 15, size=(24, 3)),
        columns=['Tráfico', 'Incendios', 'Seguridad']
    )
    st.bar_chart(incident_trend)
