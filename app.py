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
    page_title="Panel de Control de Riesgos Operacionales",
    page_icon="🚒",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- ESTILOS CSS ---
st.markdown("""
<style>
    .stDeployButton { visibility: hidden; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .main-title { font-size: 2.8rem; font-weight: bold; text-align: center; margin-bottom: 2rem; }
    h5 { text-align: center; font-weight: bold; margin-bottom: -10px; }
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
                {'range': [0, max_val * 0.4], 'color': '#28a745'}, # Verde
                {'range': [max_val * 0.4, max_val * 0.7], 'color': '#ffc107'}, # Amarillo
                {'range': [max_val * 0.7, max_val], 'color': '#dc3545'}], # Rojo
        }))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        height=200,
        margin=dict(t=30, b=20, l=30, r=30)
    )
    return fig

def calculate_dynamic_risks(temp, smoke_level, visibility):
    """Calcula los valores de riesgo basados en condiciones de exposición simuladas."""
    risks = {}
    
    # Mapeo de visibilidad a un factor numérico (peor visibilidad = mayor riesgo)
    visibility_factor = {'Buena': 0.2, 'Regular': 0.6, 'Mala': 1.0}[visibility]

    # 1. Deshidratación: Aumenta exponencialmente con la temperatura
    risks['Deshidratación'] = np.clip((temp - 20) * 3.5 + np.random.randint(-5, 5), 5, 100)

    # 2. Exposición Química: Directamente proporcional al nivel de humo/químicos
    risks['Exposición Química'] = np.clip(smoke_level / 5 + np.random.randint(-5, 5), 5, 100)
    
    # 3. Desorientación y Desubicación: Afectado por la visibilidad y el humo
    risks['Desorientación y Desubicación'] = np.clip(visibility_factor * 60 + smoke_level / 20 + np.random.randint(-10, 10), 5, 100)

    # 4. Accidente de Personal: Es un riesgo general influenciado por todas las condiciones adversas
    risks['Accidente de Personal'] = np.clip((risks['Deshidratación']/4 + risks['Desorientación y Desubicación']/3 + risks['Exposición Química']/5), 10, 100)
    
    # 5. Contagio Biológico: Un riesgo base, podría ser más complejo en un modelo real
    risks['Contagio Biológico'] = np.clip(np.random.randint(5, 25) + (smoke_level / 50), 5, 100)
    
    return {k: int(v) for k, v in risks.items()}


# --- DATOS Y ESTADO DE SESIÓN ---
FIRE_STATION_COORDS = {"lat": 3.9002, "lon": -76.3020, "tooltip": "Estación de Bomberos Buga"}

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

# --- BARRA LATERAL (CONTROLES DE EXPOSICIÓN) ---
st.sidebar.header("🔧 Simulación de Exposición")
st.sidebar.write("Ajusta las condiciones del entorno para ver cómo afectan los riesgos del personal.")

temp_ambiente = st.sidebar.slider("Temperatura en el Incidente (°C)", 15, 60, 35)
smoke_level = st.sidebar.slider("Nivel de Humo/Químicos (PPM)", 0, 500, 150)
visibility = st.sidebar.select_slider("Visibilidad en la Zona", options=['Buena', 'Regular', 'Mala'], value='Regular')

st.sidebar.header("⚙️ Simulación del Mapa")
num_intersections = st.sidebar.slider("Nº de Intersecciones (Nodos)", 50, 500, 150)
traffic_level = st.sidebar.slider("Nivel de Tráfico Simulado", 1, 10, 5)

if st.sidebar.button("🔄 Generar Nuevo Incidente"):
    st.session_state.clear()
    st.rerun()


# --- INTERFAZ PRINCIPAL ---
st.markdown("<h1 class='main-title'>🚒 Panel de Control de Riesgos Operacionales</h1>", unsafe_allow_html=True)

# --- FILA 1: PREDICCIONES DE RIESGOS DINÁMICOS ---
st.subheader("🔮 Predicciones de Riesgos (Basado en Exposición)")

operational_risks = {
    "Accidente de Personal": {"color": "#dc3545"},
    "Deshidratación": {"color": "#0dcaf0"},
    "Desorientación y Desubicación": {"color": "#ffc107"},
    "Contagio Biológico": {"color": "#28a745"},
    "Exposición Química": {"color": "#fd7e14"},
}

risk_values = calculate_dynamic_risks(temp_ambiente, smoke_level, visibility)

risk_cols = st.columns(5)
for i, (risk_name, properties) in enumerate(operational_risks.items()):
    with risk_cols[i]:
        st.markdown(f"<h5>{risk_name}</h5>", unsafe_allow_html=True)
        st.plotly_chart(create_gauge(risk_values[risk_name], color=properties["color"]), use_container_width=True)

st.divider()

# --- FILA 2: MAPA GIS Y RUTEO ---
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("🗺️ Sistema de Información Geográfica (GIS) - Simulación")
    st.caption(f"Simulación basada en mapa virtual. Última actualización: {datetime.now().strftime('%H:%M:%S')}")
    
    # Lógica de ruteo no ha cambiado, sigue siendo una simulación geoespacial.
    min_lat, max_lat = min(FIRE_STATION_COORDS["lat"], st.session_state.incident_location["lat"]) - 0.02, max(FIRE_STATION_COORDS["lat"], st.session_state.incident_location["lat"]) + 0.02
    min_lon, max_lon = min(FIRE_STATION_COORDS["lon"], st.session_state.incident_location["lon"]) - 0.02, max(FIRE_STATION_COORDS["lon"], st.session_state.incident_location["lon"]) + 0.02
    nodes_df = pd.DataFrame({'lat': np.random.uniform(min_lat, max_lat, num_intersections), 'lon': np.random.uniform(min_lon, max_lon, num_intersections)})
    G, node_coords = nx.Graph(), np.array(nodes_df[['lat', 'lon']])
    kdtree = KDTree(node_coords)
    for i, row in nodes_df.iterrows(): G.add_node(i, pos=(row['lon'], row['lat']))
    for i in range(len(node_coords)):
        distances, indices = kdtree.query([node_coords[i]], k=5)
        for j_idx, j in enumerate(indices[0][1:]):
            weight = distances[0][j_idx+1] * (1 + np.random.uniform(0, traffic_level))
            G.add_edge(i, j, weight=weight)
    start_node, end_node = kdtree.query([[FIRE_STATION_COORDS['lat'], FIRE_STATION_COORDS['lon']]], k=1)[1][0][0], kdtree.query([[st.session_state.incident_location['lat'], st.session_state.incident_location['lon']]], k=1)[1][0][0]
    try:
        path = nx.shortest_path(G, source=start_node, target=end_node, weight='weight')
        route_df = pd.DataFrame([G.nodes[i]['pos'] for i in path], columns=['lon', 'lat'])
    except nx.NetworkXNoPath:
        route_df = pd.DataFrame(columns=['lon', 'lat'])
    
    # Capas y visualización del mapa
    layers = [
        pdk.Layer('PathLayer', data=route_df, get_path='[lon, lat]', get_width=8, width_scale=1, get_color=[255, 0, 0, 200]),
        pdk.Layer('IconLayer', data=pd.DataFrame([FIRE_STATION_COORDS]), get_position='[lon, lat]', get_icon=lambda r: {'url': 'https://img.icons8.com/color/48/000000/fire-station.png', 'width': 200, 'height': 200, 'anchorY': 200}, get_size=4, size_scale=15, pickable=True),
        pdk.Layer('IconLayer', data=pd.DataFrame([st.session_state.incident_location]), get_position='[lon, lat]', get_icon=lambda r: {'url': 'https://img.icons8.com/fluency/48/siren.png', 'width': 240, 'height': 240, 'anchorY': 240}, get_size=4, size_scale=20, pickable=True),
        pdk.Layer('IconLayer', data=st.session_state.vehicle_locations, get_position='[lon, lat]', get_icon=lambda r: {'url': 'https://img.icons8.com/ultraviolet/40/000000/fire-truck.png', 'width': 200, 'height': 200, 'anchorY': 200}, get_size=4, size_scale=12, pickable=True)
    ]
    view_state = pdk.ViewState(latitude=np.mean([min_lat, max_lat]), longitude=np.mean([min_lon, max_lon]), zoom=13, pitch=45)
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view_state, map_style='mapbox://styles/mapbox/dark-v9', tooltip={"text": "{tooltip}"}))

with col2:
    st.subheader("📈 Nivel de Riesgo General")
    overall_risk = int(np.mean(list(risk_values.values())))
    st.plotly_chart(create_gauge(overall_risk, max_val=100, color="#dc3545"), use_container_width=True)

    st.subheader("📊 Tendencia de Alertas (Últimas 24h)")
    incident_trend_df = pd.DataFrame(np.random.randint(0, int(max(1, overall_risk/5)), size=(24, len(operational_risks))), columns=list(operational_risks.keys()))
    st.bar_chart(incident_trend_df)

