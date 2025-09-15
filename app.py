import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pydeck as pdk
import networkx as nx
from datetime import datetime
from sklearn.neighbors import KDTree

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Panel de Control de Riesgos Operativos",
    page_icon="🚒",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- ESTILOS CSS ---
st.markdown("""
<style>
    /* Ocultar el botón de despliegue de Streamlit */
    .stDeployButton {
        visibility: hidden;
    }
    /* Ajustar el padding del contenedor principal */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* Estilo para el título principal */
    .main-title {
        font-size: 2.8rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    /* Estilo para los títulos de las secciones */
    h5 {
        text-align: center;
        font-weight: bold;
        margin-bottom: -10px; /* Reducir espacio con el gráfico */
    }
</style>
""", unsafe_allow_html=True)


# --- FUNCIONES DE SIMULACIÓN Y GRÁFICOS ---

def create_gauge(value, max_val=100, color="red"):
    """Crea un gráfico de medidor (gauge) con Plotly."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        gauge={
            'axis': {'range': [None, max_val], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#444",
            'steps': [
                {'range': [0, max_val * 0.5], 'color': '#28a745'}, # Verde
                {'range': [max_val * 0.5, max_val * 0.8], 'color': '#ffc107'}, # Amarillo
                {'range': [max_val * 0.8, max_val], 'color': '#dc3545'}], # Rojo
        }))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        height=180, # Altura reducida para acomodar más gráficos
        margin=dict(t=20, b=20, l=30, r=30)
    )
    return fig

# --- DATOS Y ESTADO DE SESIÓN ---

# Coordenadas de la estación de bomberos de Guadalajara de Buga
FIRE_STATION_COORDS = {"lat": 3.9002, "lon": -76.3020, "tooltip": "Estación de Bomberos Buga"}

# Inicializar estado de la sesión para la simulación
if 'incident_location' not in st.session_state:
    st.session_state.incident_location = {
        "lat": FIRE_STATION_COORDS["lat"] + np.random.uniform(-0.05, 0.05),
        "lon": FIRE_STATION_COORDS["lon"] + np.random.uniform(-0.05, 0.05),
        "tooltip": "Incidente Activo"
    }
    st.session_state.vehicle_locations = pd.DataFrame({
        'lat': FIRE_STATION_COORDS["lat"] + np.random.uniform(-0.01, 0.01, 3),
        'lon': FIRE_STATION_COORDS["lon"] + np.random.uniform(-0.01, 0.01, 3),
        'tooltip': ['Unidad M-5', 'Ambulancia B-2', 'Logística T-1']
    })

# --- BARRA LATERAL (CONTROLES) ---
st.sidebar.header("🔧 Controles de Simulación")
risk_intensity = st.sidebar.slider("Intensidad General del Incidente (1-100)", 1, 100, 65)
num_intersections = st.sidebar.slider("Nº de Intersecciones (Nodos)", 50, 500, 150)
traffic_level = st.sidebar.slider("Nivel de Tráfico Simulado", 1, 10, 5)

if st.sidebar.button("🔄 Generar Nuevo Escenario"):
    st.session_state.clear()
    st.rerun()


# --- INTERFAZ PRINCIPAL ---
st.markdown("<h1 class='main-title'>🚒 Panel de Control de Riesgos Operacionales</h1>", unsafe_allow_html=True)

# --- FILA 1: PREDICCIONES DE RIESGOS OPERATIVOS ---
st.subheader("🔮 Predicciones de Riesgos Operativos en Tiempo Real")

# Riesgos extraídos del documento
operational_risks = {
    "Desorientación": {"color": "#ffc107"}, # Amarillo
    "Colapso Estructural": {"color": "#dc3545"}, # Rojo
    "Exposición a Hazmat": {"color": "#fd7e14"}, # Naranja
    "Estrés Térmico": {"color": "#0dcaf0"}, # Cian
    "Fallas SCBA": {"color": "#6f42c1"}, # Indigo
    "Accidentes de Personal": {"color": "#dc3545"}, # Rojo
    "Comunicaciones deficientes": {"color": "#ffc107"}, # Amarillo
    "Riesgo Eléctrico": {"color": "#fd7e14"}, # Naranja
    "Flashovers": {"color": "#dc3545"}, # Rojo
    "Atrapamiento": {"color": "#6f42c1"}  # Indigo
}

# Generar valores simulados para cada riesgo
risk_values = {name: np.clip(risk_intensity * np.random.uniform(0.5, 1.2), 10, 100) for name in operational_risks}

# Mostrar los 10 medidores en dos filas
row1_cols = st.columns(5)
row2_cols = st.columns(5)

for i, (risk_name, properties) in enumerate(operational_risks.items()):
    col = row1_cols[i] if i < 5 else row2_cols[i - 5]
    with col:
        st.markdown(f"<h5>{risk_name}</h5>", unsafe_allow_html=True)
        st.plotly_chart(create_gauge(int(risk_values[risk_name]), color=properties["color"]), use_container_width=True)

st.divider()

# --- FILA 2: MAPA GIS Y RUTEO ---
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("🗺️ Sistema de Información Geográfica (GIS) y Ruteo")
    st.caption(f"Última actualización: {datetime.now().strftime('%H:%M:%S')}")

    # --- Lógica de Ruteo Geoespacial ---
    # 1. Crear un 'bounding box' alrededor de la estación y el incidente
    min_lat = min(FIRE_STATION_COORDS["lat"], st.session_state.incident_location["lat"]) - 0.02
    max_lat = max(FIRE_STATION_COORDS["lat"], st.session_state.incident_location["lat"]) + 0.02
    min_lon = min(FIRE_STATION_COORDS["lon"], st.session_state.incident_location["lon"]) - 0.02
    max_lon = max(FIRE_STATION_COORDS["lon"], st.session_state.incident_location["lon"]) + 0.02

    # 2. Generar nodos (intersecciones) dentro del bounding box
    nodes_df = pd.DataFrame({
        'lat': np.random.uniform(min_lat, max_lat, num_intersections),
        'lon': np.random.uniform(min_lon, max_lon, num_intersections)
    })

    # 3. Construir el grafo y encontrar los nodos más cercanos al origen y destino
    G = nx.Graph()
    node_coords = np.array(nodes_df[['lat', 'lon']])
    kdtree = KDTree(node_coords)

    # Añadir nodos al grafo
    for i, row in nodes_df.iterrows():
        G.add_node(i, pos=(row['lon'], row['lat']))

    # Conectar nodos cercanos para simular calles
    for i in range(len(node_coords)):
        distances, indices = kdtree.query([node_coords[i]], k=5)
        for j in indices[0][1:]: # Conectar con los 4 vecinos más cercanos
            dist = distances[0][list(indices[0]).index(j)]
            # Añadir "tráfico" como peso
            weight = dist * (1 + np.random.uniform(0, traffic_level))
            G.add_edge(i, j, weight=weight)
    
    # Encontrar los nodos del grafo más cercanos a la estación y al incidente
    start_node = kdtree.query([[FIRE_STATION_COORDS['lat'], FIRE_STATION_COORDS['lon']]], k=1)[1][0][0]
    end_node = kdtree.query([[st.session_state.incident_location['lat'], st.session_state.incident_location['lon']]], k=1)[1][0][0]
    
    # 4. Calcular la ruta más corta
    try:
        path = nx.shortest_path(G, source=start_node, target=end_node, weight='weight')
        path_coords = [G.nodes[i]['pos'] for i in path]
        route_df = pd.DataFrame(path_coords, columns=['lon', 'lat'])
    except nx.NetworkXNoPath:
        route_df = pd.DataFrame(columns=['lon', 'lat']) # Ruta vacía si no se encuentra
    
    # --- Visualización del Mapa con Pydeck ---
    # Capas del mapa
    fire_station_layer = pdk.Layer('IconLayer', data=pd.DataFrame([FIRE_STATION_COORDS]), get_position='[lon, lat]', get_icon=lambda r: {'url': 'https://img.icons8.com/color/48/000000/fire-station.png', 'width': 200, 'height': 200, 'anchorY': 200}, get_size=4, size_scale=15, pickable=True)
    incident_layer = pdk.Layer('IconLayer', data=pd.DataFrame([st.session_state.incident_location]), get_position='[lon, lat]', get_icon=lambda r: {'url': 'https://img.icons8.com/fluency/48/siren.png', 'width': 240, 'height': 240, 'anchorY': 240}, get_size=4, size_scale=20, pickable=True)
    vehicle_layer = pdk.Layer('IconLayer', data=st.session_state.vehicle_locations, get_position='[lon, lat]', get_icon=lambda r: {'url': 'https://img.icons8.com/ultraviolet/40/000000/fire-truck.png', 'width': 200, 'height': 200, 'anchorY': 200}, get_size=4, size_scale=12, pickable=True)
    
    route_layer = pdk.Layer('PathLayer', data=route_df, get_path='[lon, lat]', get_width=8, width_scale=1, get_color=[255, 0, 0, 200], pickable=True)

    view_state = pdk.ViewState(latitude=np.mean([min_lat, max_lat]), longitude=np.mean([min_lon, max_lon]), zoom=13, pitch=45)
    
    st.pydeck_chart(pdk.Deck(
        layers=[route_layer, fire_station_layer, incident_layer, vehicle_layer],
        initial_view_state=view_state,
        map_style='mapbox://styles/mapbox/dark-v9',
        tooltip={"text": "{tooltip}"}
    ))

with col2:
    st.subheader("📈 Nivel de Riesgo General")
    overall_risk = int(np.mean(list(risk_values.values())))
    st.plotly_chart(create_gauge(overall_risk, max_val=100, color="#dc3545"), use_container_width=True)

    st.subheader("📊 Tendencia de Alertas (Últimas 24h)")
    # El gráfico de barras ahora refleja los 10 riesgos
    incident_trend_df = pd.DataFrame(
        np.random.randint(0, int(max(1, risk_intensity/10)), size=(24, len(operational_risks))),
        columns=list(operational_risks.keys())
    )
    st.bar_chart(incident_trend_df)

