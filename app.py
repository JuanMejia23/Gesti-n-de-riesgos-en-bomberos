import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pydeck as pdk
import networkx as nx
from datetime import datetime

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Panel de Control de Riesgos",
    page_icon="🚒",
    layout="wide",
    initial_sidebar_state="auto" # Mostrar la barra lateral por defecto
)

# --- Estilos CSS Personalizados ---
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
    h5 {
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# --- Funciones para Generar Datos Simulados (Modificadas) ---

def generate_mock_data(num_points=500, risk_intensity=60):
    """Genera datos simulados basados en parámetros de entrada."""
    # Coordenadas base (Bogotá, Colombia)
    lat_base, lon_base = 4.60971, -74.08175

    # Generar puntos de riesgo aleatorios alrededor de la base
    data = pd.DataFrame({
        'lat': lat_base + np.random.randn(num_points) * 0.05,
        'lon': lon_base + np.random.randn(num_points) * 0.05,
        # La intensidad de riesgo ahora influye en la distribución
        'risk_level': np.clip(np.random.normal(risk_intensity, 20, num_points), 1, 100).astype(int)
    })
    
    return data

def create_gauge(value, title=None, max_val=100, color="green"):
    """Crea un gráfico de medidor (gauge) con Plotly."""
    indicator = go.Indicator(
        mode = "gauge+number",
        value = value,
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
        })
    
    if title:
        indicator.title = {'text': title, 'font': {'size': 18}}

    fig = go.Figure(indicator)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor='rgba(0,0,0,0)', 
        font={'color': 'white'},
        height=250,
        margin=dict(t=40, b=40) # Ajustar margen para que encaje mejor
    )
    return fig

def create_routing_graph(num_nodes=15, edge_prob=0.3):
    """Crea un grafo y calcula la ruta más corta para la visualización."""
    if num_nodes < 2:
        return go.Figure().update_layout(title_text="Se necesitan al menos 2 nodos.")

    G = nx.erdos_renyi_graph(num_nodes, edge_prob, seed=42)
    
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    
    if len(G.nodes) < 2:
        fig = go.Figure()
        fig.update_layout(title={'text': "Ruta no encontrada (grafo no conectado)", 'font': {'color': 'white'}}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor='#1E1E1E')
        return fig
    
    pos = nx.spring_layout(G, seed=42)
    
    for (u, v) in G.edges():
        G.edges[u,v]['weight'] = np.random.randint(1, 10)
    
    node_list = list(G.nodes)
    source, target = node_list[0], node_list[-1]
    
    try:
        shortest_path = nx.shortest_path(G, source=source, target=target, weight='weight')
        path_x = [pos[node][0] for node in shortest_path]
        path_y = [pos[node][1] for node in shortest_path]
    except nx.NetworkXNoPath:
        shortest_path = []
        path_x, path_y = [], []

    fig = go.Figure()
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines'))
    
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', marker=dict(color='lightblue', size=15, line_width=2)))
    
    if shortest_path:
        fig.add_trace(go.Scatter(x=path_x, y=path_y, line=dict(width=5, color='red'), name='Ruta Óptima', mode='lines+markers', marker=dict(size=18, color='red')))
    
    fig.update_layout(title={'text': "Algoritmo de Ruta Más Corta (Teoría de Grafos)", 'font': {'color': 'white'}}, showlegend=False, hovermode='closest', margin=dict(b=20,l=5,r=5,t=40), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor='#1E1E1E', font={'color': 'white'})
    return fig

# --- Barra Lateral con Controles de Simulación ---
st.sidebar.header("🔧 Controles de Simulación")
st.sidebar.subheader("Parámetros del Mapa de Riesgo")
num_points = st.sidebar.slider("Número de Puntos de Riesgo", 100, 2000, 500, 100)
risk_intensity = st.sidebar.slider("Intensidad de Riesgo Promedio (1-100)", 1, 100, 60)
st.sidebar.subheader("Parámetros de Ruteo")
num_nodes = st.sidebar.slider("Nº de Nodos (intersecciones)", 5, 50, 15)
edge_prob = st.sidebar.slider("Complejidad de la Red (calles)", 0.1, 1.0, 0.4, 0.1)
if st.sidebar.button("🔄 Generar Nuevo Escenario"):
    st.session_state.clear()
    st.rerun()

# --- Gestión de Estado para el Incidente ---
if 'incident_location' not in st.session_state:
    lat_base, lon_base = 4.60971, -74.08175
    st.session_state.incident_location = pd.DataFrame({
        'lat': [lat_base + np.random.uniform(-0.05, 0.05)],
        'lon': [lon_base + np.random.uniform(-0.05, 0.05)],
        'tooltip': ['Incidente Activo']
    })

# --- Carga de Datos ---
risk_data = generate_mock_data(num_points, risk_intensity)

# --- Interfaz de Usuario ---
st.markdown("<h1 class='main-title'>🚒 Panel de Control de Riesgos Operativos</h1>", unsafe_allow_html=True)
# --- Fila 1: Monitoreo en vivo y predicciones ---
col1, col2 = st.columns([2, 3])
with col1:
    st.subheader("🛰️ Monitoreo Aéreo (Dron)")
    st.image("https://placehold.co/600x400/000000/FFFFFF?text=Vista+de+Dron", caption="Simulación de imagen de dron con superposición de riesgo.", use_column_width=True)
    st.metric(label="Estado del Dron", value="Activo")
    st.metric(label="Batería", value=f"{np.random.randint(70, 95)}%")

with col2:
    st.subheader("🔮 Predicciones de Riesgos Operativos (Bomberos)")
    sub_col1, sub_col2, sub_col3, sub_col4 = st.columns(4) # Cambiado a 4 columnas
    overall_risk_from_data = risk_data['risk_level'].mean()
    
    # Simulación de riesgos operativos específicos
    traffic_accident_risk = np.clip(overall_risk_from_data * np.random.uniform(0.3, 0.6), 5, 100)
    dehydration_risk = np.clip(overall_risk_from_data * np.random.uniform(0.2, 0.7), 10, 100)
    disorientation_risk = np.clip(overall_risk_from_data * np.random.uniform(0.1, 0.5), 5, 100)
    personnel_accident_risk = np.clip(overall_risk_from_data * np.random.uniform(0.2, 0.4), 10, 100)

    with sub_col1:
        st.markdown("<h5>Accidente de Tránsito</h5>", unsafe_allow_html=True)
        st.plotly_chart(create_gauge(int(traffic_accident_risk), color="orange"), use_container_width=True)
    with sub_col2:
        st.markdown("<h5>Deshidratación</h5>", unsafe_allow_html=True)
        st.plotly_chart(create_gauge(int(dehydration_risk), color="#ADD8E6"), use_container_width=True) # Light Blue
    with sub_col3:
        st.markdown("<h5>Desorientación</h5>", unsafe_allow_html=True)
        st.plotly_chart(create_gauge(int(disorientation_risk), color="yellow"), use_container_width=True)
    with sub_col4:
        st.markdown("<h5>Accidente de Personal</h5>", unsafe_allow_html=True)
        st.plotly_chart(create_gauge(int(personnel_accident_risk), color="red"), use_container_width=True)


st.divider()

# --- Fila 2: Mapa Principal y Métricas Agregadas ---
col1, col2 = st.columns([3, 2])
with col1:
    st.subheader("🗺️ Mapa de Riesgos GIS en Tiempo Real")
    st.caption(f"Última actualización de datos: {datetime.now().strftime('%H:%M:%S')}")
    if st.button("Simular Actualización del Mapa en Vivo"):
        lat_base, lon_base = 4.60971, -74.08175
        st.session_state.incident_location = pd.DataFrame({
            'lat': [lat_base + np.random.uniform(-0.05, 0.05)],
            'lon': [lon_base + np.random.uniform(-0.05, 0.05)],
            'tooltip': ['Incidente Activo']
        })
        st.rerun()

    view_state = pdk.ViewState(latitude=risk_data["lat"].mean(), longitude=risk_data["lon"].mean(), zoom=11, pitch=50)
    heatmap_layer = pdk.Layer('HeatmapLayer', data=risk_data, get_position='[lon, lat]', opacity=0.5, get_weight='risk_level')
    incident_layer = pdk.Layer('IconLayer', data=st.session_state.incident_location, get_position='[lon, lat]', get_icon=lambda r: {'url': 'https://img.icons8.com/fluency/48/siren.png', 'width': 240, 'height': 240, 'anchorY': 240}, get_size=4, size_scale=20, pickable=True)
    
    r = pdk.Deck(layers=[heatmap_layer, incident_layer], initial_view_state=view_state, map_style='mapbox://styles/mapbox/dark-v9', tooltip={"text": "{tooltip}"})
    st.pydeck_chart(r)

with col2:
    st.subheader("📈 Nivel de Riesgo General")
    overall_risk = int(risk_data['risk_level'].mean())
    st.plotly_chart(create_gauge(overall_risk, title="Riesgo Total de la Operación", color="red"), use_container_width=True)
    st.subheader("Tendencias de Riesgo (Última Hora)")
    trend_data = pd.DataFrame({'Minuto': range(-60, 0, 5), 'Nivel de Riesgo': np.clip(np.random.normal(overall_risk, 10, 12), 1, 100)})
    st.line_chart(trend_data.set_index('Minuto'), color="#FF4B4B")
    
st.divider()

# --- Fila 3: Ruteo y Tendencias a Largo Plazo ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("🚗 Ruteo de Vehículos de Emergencia")
    st.plotly_chart(create_routing_graph(num_nodes, edge_prob), use_container_width=True)
    st.metric(label="Tiempo Estimado de Llegada", value=f"{np.random.randint(8, 15)} min {np.random.randint(0, 59)} seg")
with col2:
    st.subheader("📊 Tendencia de Alertas Operativas (Últimas 24h)")
    incident_trend = pd.DataFrame(
        np.random.randint(0, int(max(1, risk_intensity/10))), 
        size=(24, 4), 
        columns=['Acc. Tránsito', 'Deshidratación', 'Desorientación', 'Acc. Personal']
    )
    st.bar_chart(incident_trend)

