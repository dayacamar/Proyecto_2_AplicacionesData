import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =====================================================
# CONFIGURACIÓN
# =====================================================

st.set_page_config(
    page_title="Educación Colombia 2017-2021",
    layout="wide"
)

# =====================================================
# CARGA DE DATOS
# =====================================================

@st.cache_data
def cargar_datos():

    base = pd.read_csv(
        r"C:\Users\juana\OneDrive\Desktop\Proyecto_2_AplicacionesData\Data\Processed\analisis_municipal.csv"
    )

    maestro = pd.read_csv(
        r"C:\Users\juana\OneDrive\Desktop\Proyecto_2_AplicacionesData\Data\Processed\maestro_municipal_2017_2021.csv"
    )

    return base, maestro


base, maestro = cargar_datos()

# =====================================================
# TÍTULO
# =====================================================

st.title(
    "Educación en Colombia antes y después de la pandemia"
)

st.markdown("""

### Pregunta de investigación

¿Las condiciones educativas y los resultados académicos
antes de la pandemia se relacionaron con los cambios
en reprobación, deserción y educación superior
entre 2020 y 2021?

""")

# =====================================================
# KPIs
# =====================================================

st.subheader("Indicadores principales")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Municipios analizados",
    len(base)
)

c2.metric(
    "Cambio reprobación",
    f"{base['d_reprobacion_19_20'].median():.2f} pp"
)

c3.metric(
    "Cambio deserción",
    f"{base['d_desercion_19_21'].median():.2f} pp"
)

c4.metric(
    "Cambio matrícula superior",
    f"{base['cambio_matricula_19_20_pct'].median():.2f}%"
)

st.divider()

# =====================================================
# EVOLUCIÓN TEMPORAL
# =====================================================

st.subheader(
    "Evolución de los indicadores educativos"
)

serie = (
    maestro
    .groupby("anio")
    .agg({
        "aprobación_media": "median",
        "reprobación_media": "median",
        "deserción_media": "median"
    })
    .reset_index()
)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=serie["anio"],
        y=serie["aprobación_media"],
        mode="lines+markers",
        name="Aprobación",
        line=dict(width=3)
    )
)

fig.add_trace(
    go.Scatter(
        x=serie["anio"],
        y=serie["reprobación_media"],
        mode="lines+markers",
        name="Reprobación",
        line=dict(width=3)
    )
)

fig.add_trace(
    go.Scatter(
        x=serie["anio"],
        y=serie["deserción_media"],
        mode="lines+markers",
        name="Deserción",
        line=dict(width=3)
    )
)

fig.add_vline(
    x=2020,
    line_dash="dash"
)

fig.update_layout(
    height=350,
    xaxis_title="Año",
    yaxis_title="Porcentaje"
)

fig.update_xaxes(
    tickmode="array",
    tickvals=[2017, 2018, 2019, 2020, 2021]
)

c1, c2, c3 = st.columns([1, 4, 1])

with c2:
    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.divider()

# =====================================================
# FACTORES ASOCIADOS AL RENDIMIENTO
# =====================================================

st.subheader(
    "Factores asociados al puntaje de lectura"
)

corr_df = pd.DataFrame({
    "Variable": [
        "Colegios no oficiales",
        "Cobertura educativa",
        "Estrato promedio",
        "Ruralidad"
    ],
    "Correlacion": [
        0.59,
        0.58,
        0.57,
        -0.32
    ]
})

fig_corr = px.bar(
    corr_df,
    x="Correlacion",
    y="Variable",
    orientation="h",
    text="Correlacion"
)

fig_corr.update_traces(
    texttemplate="%{text:.2f}",
    textposition="outside"
)

fig_corr.update_layout(
    height=420,
    showlegend=False,
    xaxis_title="Correlación con el puntaje de lectura",
    yaxis_title=""
)

st.plotly_chart(
    fig_corr,
    use_container_width=True
)

st.divider()

# =====================================================
# CUARTILES
# =====================================================

st.subheader(
    "Comparación por niveles de desempeño previo"
)

base["cuartil_lectura"] = pd.qcut(
    base["punt_lectura"],
    4,
    labels=[
        "Q1 Bajo",
        "Q2",
        "Q3",
        "Q4 Alto"
    ]
)

agr = (
    base.groupby(
        "cuartil_lectura",
        observed=True
    )
    .agg({
        "d_reprobacion_19_20": "median",
        "d_desercion_19_21": "median"
    })
    .reset_index()
)

c1, c2 = st.columns(2)

with c1:

    fig_rep = px.bar(
        agr,
        x="cuartil_lectura",
        y="d_reprobacion_19_20",
        color="cuartil_lectura",
        title="Cambio en reprobación"
    )

    fig_rep.update_layout(
        showlegend=False,
        xaxis_title="Nivel de lectura previo",
        yaxis_title="Cambio 2019-2020"
    )

    st.plotly_chart(
        fig_rep,
        use_container_width=True
    )

with c2:

    fig_des = px.bar(
        agr,
        x="cuartil_lectura",
        y="d_desercion_19_21",
        color="cuartil_lectura",
        title="Cambio en deserción"
    )

    fig_des.update_layout(
        showlegend=False,
        xaxis_title="Nivel de lectura previo",
        yaxis_title="Cambio 2019-2021"
    )

    st.plotly_chart(
        fig_des,
        use_container_width=True
    )

st.divider()
st.subheader(
    "Acceso a educación superior durante la pandemia"
)
agr_sup = (
    base.groupby(
        "cuartil_lectura",
        observed=True
    )
    .agg({
        "cambio_matricula_19_20_pct": "median"
    })
    .reset_index()
)

fig_sup = px.bar(
    agr_sup,
    x="cuartil_lectura",
    y="cambio_matricula_19_20_pct",
    color="cuartil_lectura",
    title="Cambio en matrícula de educación superior"
)

fig_sup.update_layout(
    showlegend=False,
    xaxis_title="Nivel de lectura previo",
    yaxis_title="Cambio matrícula (%)"
)

st.plotly_chart(
    fig_sup,
    use_container_width=True
)
# =====================================================
# HALLAZGOS
# =====================================================

st.subheader(
    "Principales hallazgos"
)

st.info(
    """
    • Los mejores resultados de lectura se asociaron con una mayor cobertura educativa, un mayor estrato promedio y una mayor presencia de colegios no oficiales.

    • La ruralidad mostró una relación negativa con el desempeño académico.

    • Durante la pandemia aumentaron los niveles de reprobación y posteriormente los niveles de deserción.

    • Los municipios con peores resultados previos presentaron mayores incrementos de reprobación.

    • No se encontró evidencia clara de que los municipios con mejores resultados académicos antes de la pandemia resistieran significativamente mejor los efectos educativos de la pandemia.
    """
)
st.divider()

# =====================================================
# RESPUESTA FINAL
# =====================================================

st.subheader("Respuesta a la pregunta de investigación")

st.success("""
Aunque los municipios con mejores resultados académicos presentaban mejores condiciones educativas,
no encontramos evidencia clara de que resistieran significativamente mejor los efectos educativos de la pandemia.
""")