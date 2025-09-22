import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pydeck as pdk
from datetime import datetime

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Panel de Control de Riesgos Operacionales",
    page_icon="🧑‍🚒",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- ESTILOS CSS ---
st.markdown("""
<style>
    .stDeployButton { visibility: hidden; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .main-title { font-size: 2.8rem; font-weight: bold; text-align: center; margin-bottom: 1.5rem; }
    h5 { text-align: center; font-weight: bold; margin-bottom: -10px; }
    .stMetric { 
        border: 1px solid #444;
        border-radius: 10px;
        padding: 1rem;
        background-color: #1E1E1E;
        text-align: center;
    }
    .recommendation-box {
        border: 1px solid #ffc107;
        border-radius: 10px;
        padding: 1rem;
        background-color: #2c2c2c;
        height: 100%;
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
                {'range': [0, max_val * 0.4], 'color': '#28a745'},
                {'range': [max_val * 0.4, max_val * 0.7], 'color': '#ffc107'},
                {'range': [max_val * 0.7, max_val], 'color': '#dc3545'}],
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
    visibility_factor = {'Buena': 0.2, 'Regular': 0.6, 'Mala': 1.0}[visibility]
    risks['Deshidratación'] = np.clip((temp - 20) * 3.5 + np.random.randint(-5, 5), 5, 100)
    risks['Exposición Química'] = np.clip(smoke_level / 5 + np.random.randint(-5, 5), 5, 100)
    risks['Desorientación y Desubicación'] = np.clip(visibility_factor * 60 + smoke_level / 20 + np.random.randint(-10, 10), 5, 100)
    risks['Accidente de Personal'] = np.clip((risks['Deshidratación']/4 + risks['Desorientación y Desubicación']/3 + risks['Exposición Química']/5), 10, 100)
    risks['Contagio Biológico'] = np.clip(np.random.randint(5, 25) + (smoke_level / 50), 5, 100)
    return {k: int(v) for k, v in risks.items()}

def get_recommendations(risk_values):
    """Genera recomendaciones basadas en el riesgo más alto."""
    max_risk_name = max(risk_values, key=risk_values.get)
    max_risk_value = risk_values[max_risk_name]
    
    if max_risk_value < 50:
        return "✅ **Condiciones Estables:** Monitorear signos vitales y mantener comunicación constante."

    if max_risk_name == 'Deshidratación' and max_risk_value > 70:
        return "💧 **ALERTA DE DESHIDRATACIÓN:** Ordenar rotación inmediata y rehidratación del personal expuesto."
    elif max_risk_name == 'Exposición Química' and max_risk_value > 70:
        return "☣️ **PELIGRO QUÍMICO:** Verificar EPP. Considerar evacuación del personal en la zona de mayor concentración."
    elif max_risk_name == 'Desorientación y Desubicación' and max_risk_value > 75:
        return "🧭 **RIESGO DE DESORIENTACIÓN:** Verificar comunicación y ubicación del equipo. Usar línea de vida y reportar cada 5 minutos."
    elif max_risk_name == 'Accidente de Personal' and max_risk_value > 80:
        return "⚠️ **ALTO RIESGO DE ACCIDENTE:** Condiciones muy adversas. Reevaluar estrategia y considerar retirada parcial del equipo."
    else:
        return f"🔎 **ATENCIÓN EN {max_risk_name.upper()}:** Aumentar precauciones y monitorear al personal relacionado con este riesgo."


# --- DATOS Y ESTADO DE SESIÓN ---
FIRE_STATION_COORDS = {"lat": 3.9002, "lon": -76.3020, "tooltip": "Estación de Bomberos Buga"}
if 'incident_location' not in st.session_state:
    st.session_state.incident_location = {
        "lat": FIRE_STATION_COORDS["lat"] + np.random.uniform(0.01, 0.05),
        "lon": FIRE_STATION_COORDS["lon"] + np.random.uniform(0.01, 0.05),
        "tooltip": "Incidente Activo"
    }

# --- BARRA LATERAL (CONTROLES DE EXPOSICIÓN) ---
st.sidebar.header("🔧 Simulación de Exposición")
st.sidebar.write("Ajusta las condiciones del entorno para ver cómo afectan los riesgos del personal.")
temp_ambiente = st.sidebar.slider("Temperatura en el Incidente (°C)", 15, 60, 35)
smoke_level = st.sidebar.slider("Nivel de Humo/Químicos (PPM)", 0, 500, 150)
visibility = st.sidebar.select_slider("Visibilidad en la Zona", options=['Buena', 'Regular', 'Mala'], value='Regular')
if st.sidebar.button("🔄 Generar Nuevo Incidente"):
    st.session_state.clear()
    st.rerun()


# --- INTERFAZ PRINCIPAL ---
st.markdown("<h1 class='main-title'>🧑‍🚒 Panel de Monitoreo de Personal</h1>", unsafe_allow_html=True)

# --- FILA 1: PREDICCIONES DE RIESGOS DINÁMICOS ---
st.subheader("🔮 Predicciones de Riesgos (Basado en Exposición)")
operational_risks = {
    "Accidente de Personal": {"color": "#dc3545"}, "Deshidratación": {"color": "#0dcaf0"},
    "Desorientación y Desubicación": {"color": "#ffc107"}, "Contagio Biológico": {"color": "#28a745"},
    "Exposición Química": {"color": "#fd7e14"},
}
risk_values = calculate_dynamic_risks(temp_ambiente, smoke_level, visibility)
risk_cols = st.columns(5)
for i, (risk_name, properties) in enumerate(operational_risks.items()):
    with risk_cols[i]:
        st.markdown(f"<h5>{risk_name}</h5>", unsafe_allow_html=True)
        st.plotly_chart(create_gauge(risk_values[risk_name], color=properties["color"]), use_container_width=True)

st.divider()

# --- FILA 2: MONITORIZACIÓN DE PERSONAL INDIVIDUAL ---
st.subheader("📟 Estado del Personal en Campo")
personnel_cols = st.columns(3)
personnel_data = {
    "Jefe de Unidad (L-1)": {"temp_factor": 0.8, "hr_factor": 1.1},
    "Bombero (B-1)": {"temp_factor": 1.2, "hr_factor": 1.5},
    "Bombero (B-2)": {"temp_factor": 1.1, "hr_factor": 1.3},
}
for i, (name, factors) in enumerate(personnel_data.items()):
    with personnel_cols[i]:
        st.metric(label=f"**{name}** - Ritmo Cardíaco", value=f"{int(80 + risk_values['Accidente de Personal']/2 * factors['hr_factor'])} BPM")
        st.metric(label=f"**{name}** - Temp. Corporal", value=f"{37.5 + (risk_values['Deshidratación']/100) * factors['temp_factor']:.1f} °C")
        st.metric(label=f"**{name}** - Tiempo de Exposición", value=f"{np.random.randint(5,25)} min")

st.divider()

# --- FILA 3: MAPA GIS Y ACCIONES ---
col1, col2 = st.columns([3, 2])
with col1:
    st.subheader("🗺️ Mapa de Situación (GIS)")
    # Simulación de nube de humo/químicos
    plume_data = pd.DataFrame({
        'lat': st.session_state.incident_location['lat'] + np.random.normal(0, 0.0015, 200),
        'lon': st.session_state.incident_location['lon'] + np.random.normal(0, 0.0015, 200),
        'intensity': np.clip(np.random.normal(smoke_level, 50, 200), 1, 500)
    })
    smoke_layer = pdk.Layer(
        'HeatmapLayer', data=plume_data, get_position='[lon, lat]',
        opacity=0.5, get_weight='intensity',
        color_range=[[255, 255, 178, 20], [254, 204, 92, 80], [253, 141, 60, 150]]
    )
    # Capas de iconos
    incident_layer = pdk.Layer('IconLayer', data=pd.DataFrame([st.session_state.incident_location]), get_position='[lon, lat]', get_icon=lambda r: {'url': 'https://img.icons8.com/fluency/48/siren.png', 'width': 240, 'height': 240}, get_size=4, size_scale=20)
    station_layer = pdk.Layer('IconLayer', data=pd.DataFrame([FIRE_STATION_COORDS]), get_position='[lon, lat]', get_icon=lambda r: {'url': 'https://img.icons8.com/color/48/000000/fire-station.png', 'width': 200, 'height': 200}, get_size=4, size_scale=15)
    
    view_state = pdk.ViewState(latitude=st.session_state.incident_location['lat'], longitude=st.session_state.incident_location['lon'], zoom=14, pitch=50)
    st.pydeck_chart(pdk.Deck(layers=[smoke_layer, incident_layer, station_layer], initial_view_state=view_state, map_style='mapbox://styles/mapbox/dark-v9'))

with col2:
    st.subheader("📈 Nivel de Riesgo General")
    overall_risk = int(np.mean(list(risk_values.values())))
    st.plotly_chart(create_gauge(overall_risk, max_val=100, color="#dc3545"), use_container_width=True)
    
    st.subheader("💡 Acciones Recomendadas")
    recommendation = get_recommendations(risk_values)
    st.markdown(f"<div class='recommendation-box'>{recommendation}</div>", unsafe_allow_html=True)

