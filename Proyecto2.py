
from pathlib import Path

import numpy as np
import pandas as pd

# Ubicación de las carpetas y archivos
BASE = Path(__file__).resolve().parent
PROC = BASE / "Data" / "Processed"
RUTA_LIMPIO = PROC / "saber11_limpio.csv"
RUTA_MAESTRO = PROC / "maestro_municipal_2017_2021.csv"

ANIOS_BASE = [2017, 2019]
MIN_EVAL = 30
MIN_EVAL_MPIO = 30

# Departamentos que pertenecen a cada región
REGION = {
    "Caribe": ["08", "13", "20", "23", "44", "47", "70", "88"],
    "Pacifica": ["19", "27", "52", "76"],
    "Andina": ["05", "11", "15", "17", "25", "41", "54", "63", "66", "68", "73"],
    "Orinoquia": ["50", "81", "85", "99", "95"],
    "Amazonia": ["18", "86", "91", "94", "97"],
}

MAPA_REGION = {
    c: r for r, cods in REGION.items() for c in cods
}

# Cambiar los nombres de algunas columnas para trabajar más fácil
RENOMBRAR_MAESTRO = {
    "deserción_media": "desercion",
    "aprobación_media": "aprobacion",
    "reprobación_media": "reprobacion",
    "repitencia_media": "repitencia",
    "población_5_16": "poblacion_5_16",
}


# 1. Calcular los promedios de Saber 11

def promedio_grupo(df, mascara, sufijo, claves):
    g = df[mascara].groupby(claves)

    out = g.agg(
        n=("punt_global", "size"),
        glob=("punt_global", "mean"),
        lect=("punt_lectura", "mean")
    )

    # Tener en cuenta solo los grupos con mínimo 30 estudiantes
    out = out[out["n"] >= MIN_EVAL].drop(columns="n")

    out.columns = [
        f"punt_global_{sufijo}",
        f"punt_lectura_{sufijo}"
    ]

    return out


def construir_brechas(s, claves):
    # Calcular los promedios y porcentajes de cada grupo
    base = s.groupby(claves).agg(
        n_eval=("punt_global", "size"),
        punt_global=("punt_global", "mean"),
        punt_lectura=("punt_lectura", "mean"),
        pct_no_oficial=("es_no_oficial", "mean"),
        pct_rural=("es_rural", "mean"),
        estrato_prom=("estrato", "mean")
    )

    # Convertir los porcentajes a una escala de 0 a 100
    base[["pct_no_oficial", "pct_rural"]] *= 100

    # Calcular los promedios para cada tipo de colegio y zona
    b = pd.concat([
        base,
        promedio_grupo(s, s["es_no_oficial"] == 0, "oficial", claves),
        promedio_grupo(s, s["es_no_oficial"] == 1, "no_oficial", claves),
        promedio_grupo(s, s["es_rural"] == 0, "urbano", claves),
        promedio_grupo(s, s["es_rural"] == 1, "rural", claves)
    ], axis=1).reset_index()

    # Calcular las diferencias entre los grupos
    b["brecha_no_oficial_vs_oficial"] = (
        b["punt_global_no_oficial"] - b["punt_global_oficial"]
    )

    b["brecha_urbano_vs_rural"] = (
        b["punt_global_urbano"] - b["punt_global_rural"]
    )

    b["brecha_lectura_urbano_rural"] = (
        b["punt_lectura_urbano"] - b["punt_lectura_rural"]
    )

    return b


# 2. Revisar los cambios en los datos del MEN durante la pandemia

def cargar_maestro():
    m = pd.read_csv(
        RUTA_MAESTRO,
        dtype={"cod_mpio": str, "cod_depto": str}
    )

    # Cambiar algunos nombres de columnas
    m = m.rename(columns=RENOMBRAR_MAESTRO)

    m["cod_mpio"] = m["cod_mpio"].str.zfill(5)
    m["cod_depto"] = m["cod_depto"].str.zfill(2)

    # Agregar la región de cada municipio
    m["region"] = m["cod_depto"].map(MAPA_REGION)

    # Identificar si el municipio tiene oferta de educación superior
    m["con_ies"] = (
        m["tiene_oferta_ies"] == "Con oferta IES"
    ).astype(int)

    return m


def cambios_pandemia(m):
    # Seleccionar los datos que se van a comparar
    vars_ = [
        "reprobacion", "desercion", "aprobacion",
        "cobertura_neta_media", "indice_calidad_media",
        "matricula_pregrado", "con_ies"
    ]

    # Organizar los datos por municipio y año
    ancho = m.pivot_table(
        index="cod_mpio",
        columns="anio",
        values=vars_
    )

    ancho.columns = [
        f"{v}_{a}" for v, a in ancho.columns
    ]

    out = pd.DataFrame(index=ancho.index)

    # Comparar los resultados de 2019 con los de 2020 y 2021
    for v in [
        "reprobacion", "desercion",
        "aprobacion", "cobertura_neta_media"
    ]:
        out[f"{v}_2019"] = ancho[f"{v}_2019"]

        out[f"d_{v}_19_20"] = (
            ancho[f"{v}_2020"] - ancho[f"{v}_2019"]
        )

        out[f"d_{v}_19_21"] = (
            ancho[f"{v}_2021"] - ancho[f"{v}_2019"]
        )

    out["con_ies_2019"] = ancho["con_ies_2019"]
    out["con_ies_2020"] = ancho["con_ies_2020"]

    # Identificar los municipios que dejaron de tener oferta de educación superior
    out["perdio_ies_19_20"] = (
        (ancho["con_ies_2019"] == 1) &
        (ancho["con_ies_2020"] == 0)
    ).astype(int)

    # Calcular el cambio en la cantidad de estudiantes matriculados
    # Se compara 2019 con 2020 porque los datos de 2021 parecen incompletos
    m19 = ancho["matricula_pregrado_2019"]
    m20 = ancho["matricula_pregrado_2020"]

    out["cambio_matricula_19_20_pct"] = np.where(
        m19 > 0,
        (m20 - m19) / m19 * 100,
        np.nan
    )

    # Agregar el nombre del municipio, departamento y región
    info = m.drop_duplicates("cod_mpio").set_index("cod_mpio")[
        ["municipio", "departamento", "cod_depto", "region"]
    ]

    return out.join(info).reset_index()


# 3. Mostrar los resultados del análisis

def describir(m, base, cambios, n_limpias):
    pd.set_option("display.width", 170)
    pd.set_option("display.max_columns", 40)

    # Comparar la cantidad de estudiantes de Saber 11 con los datos del MEN
    print("\n=== 0. Comparación de datos ===")

    en_maestro = (
        m[m.anio.isin(ANIOS_BASE)]
        .groupby("anio")["total_evaluados_s11"]
        .sum().round(0).to_dict()
    )

    for a in ANIOS_BASE:
        n_l = n_limpias.get(a, 0)
        n_m = en_maestro.get(a, 0)

        dif = abs(n_l - n_m) / n_m * 100 if n_m else float("nan")

        print(
            f"  {a}: limpio = {n_l:,} | "
            f"maestro = {n_m:,.0f} | diferencia = {dif:.1f}%"
            + ("   <-- revisar" if dif > 5 else "")
        )

    # Mostrar las diferencias entre colegios urbanos y rurales
    print("\n=== 1. Diferencias en Saber 11 (2017 y 2019) ===")

    print("Diferencia entre urbano y rural:")
    print(
        base["brecha_urbano_vs_rural"]
        .describe().round(1).to_string()
    )

    print("\nDiferencia entre colegios no oficiales y oficiales:")
    print(
        base["brecha_no_oficial_vs_oficial"]
        .describe().round(1).to_string()
    )

    # Mostrar los resultados separados por región
    print("\n=== 2. Resultados por región ===")

    print(
        base.groupby("region").agg(
            municipios=("cod_mpio", "size"),
            lectura=("punt_lectura", "median"),
            global_=("punt_global", "median"),
            pct_rural=("pct_rural", "median"),
            pct_no_oficial=("pct_no_oficial", "median"),
            pct_con_ies_2019=("con_ies_2019", lambda x: x.mean() * 100)
        ).round(1)
    )

    # Mostrar los resultados del MEN por año
    print("\n=== 3. Datos del MEN por año ===")

    print(
        m.groupby("anio").agg(
            reprobacion=("reprobacion", "median"),
            desercion=("desercion", "median"),
            aprobacion=("aprobacion", "median"),
            cobertura=("cobertura_neta_media", "median")
        ).round(2)
    )

    # Dividir los municipios en cuatro grupos según sus resultados de lectura
    base = base.copy()

    base["cuartil_lectura"] = pd.qcut(
        base["punt_lectura"],
        4,
        labels=["Q1 (baja)", "Q2", "Q3", "Q4 (alta)"]
    )

    # Dividir los municipios según su porcentaje de estudiantes rurales
    base["cuartil_rural"] = pd.qcut(
        base["pct_rural"].rank(method="first"),
        4,
        labels=["Q1 (urbano)", "Q2", "Q3", "Q4 (rural)"]
    )

    cols = [
        "d_reprobacion_19_20", "d_desercion_19_20",
        "d_reprobacion_19_21", "d_desercion_19_21",
        "cambio_matricula_19_20_pct"
    ]

    # Comparar los cambios de la pandemia según los grupos anteriores
    for etiqueta, col in [
        ("lectura en la línea de base", "cuartil_lectura"),
        ("ruralidad en la línea de base", "cuartil_rural")
    ]:
        print(
            f"\n=== 4. Cambios durante la pandemia según {etiqueta} ==="
        )

        print(
            base.groupby(col, observed=True)[cols]
            .median().round(2).assign(
                municipios=base.groupby(
                    col, observed=True
                ).size()
            )
        )

    # Revisar la relación entre las variables del estudio
    print("\n=== 5. Relación entre los datos (Spearman) ===")

    v = [
        "punt_lectura", "pct_rural", "pct_no_oficial",
        "estrato_prom", "cobertura_neta_media_2019",
        "d_reprobacion_19_20", "d_desercion_19_21",
        "cambio_matricula_19_20_pct", "con_ies_2019"
    ]

    print(
        base[[c for c in v if c in base.columns]]
        .corr(method="spearman").round(2).to_string()
    )

    return base


# 4. Crear los modelos para analizar los datos

def modelos(base):
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print(
            "\nInstala statsmodels para ejecutar los modelos: "
            "pip install statsmodels"
        )
        return

    b = base.copy()

    def ajustar(titulo, formula, tipo="ols", cols=()):
        d = b.dropna(subset=list(cols)).copy()

        print(f"\n=== {titulo} (n = {len(d):,}) ===")

        if len(d) < 50:
            print(
                "Hay muy pocos municipios con datos completos. "
                "Revisa la información."
            )
            return

        if tipo == "ols":
            print(
                smf.ols(formula, data=d)
                .fit(cov_type="HC3").summary().tables[1]
            )
        else:
            print(
                smf.logit(formula, data=d)
                .fit(disp=False).summary().tables[1]
            )

    ctrl = "estrato_prom + cobertura_neta_media_2019 + C(region)"

    # Modelo A: revisar qué municipios tuvieron más reprobación en 2020
    ajustar(
        "Modelo A: cambio en reprobación 2019 a 2020 (OLS)",
        f"d_reprobacion_19_20 ~ punt_lectura + pct_rural + "
        f"pct_no_oficial + reprobacion_2019 + {ctrl}",
        cols=[
            "d_reprobacion_19_20", "punt_lectura",
            "pct_rural", "pct_no_oficial", "reprobacion_2019",
            "estrato_prom", "cobertura_neta_media_2019", "region"
        ]
    )

    # Modelo B: revisar qué municipios tuvieron más deserción hasta 2021
    ajustar(
        "Modelo B: cambio en deserción 2019 a 2021 (OLS)",
        f"d_desercion_19_21 ~ punt_lectura + pct_rural + "
        f"pct_no_oficial + desercion_2019 + {ctrl}",
        cols=[
            "d_desercion_19_21", "punt_lectura",
            "pct_rural", "pct_no_oficial", "desercion_2019",
            "estrato_prom", "cobertura_neta_media_2019", "region"
        ]
    )

    # Modelo C: revisar qué se relaciona con tener oferta de educación superior
    ajustar(
        "Modelo C: oferta de educación superior en 2019 (Logit)",
        f"con_ies_2019 ~ punt_lectura + pct_rural + "
        f"pct_no_oficial + {ctrl}",
        tipo="logit",
        cols=[
            "con_ies_2019", "punt_lectura", "pct_rural",
            "pct_no_oficial", "estrato_prom",
            "cobertura_neta_media_2019", "region"
        ]
    )

    # Modelo D: analizar los resultados de lectura según las características del municipio
    ajustar(
        "Modelo D: puntaje de lectura en la línea de base (OLS)",
        "punt_lectura ~ pct_rural + pct_no_oficial + estrato_prom "
        "+ cobertura_neta_media_2019 + C(region)",
        cols=[
            "punt_lectura", "pct_rural", "pct_no_oficial",
            "estrato_prom", "cobertura_neta_media_2019", "region"
        ]
    )


# 5. Crear los gráficos del análisis

def graficos(m, base):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Unir los datos del MEN con los grupos de lectura y ruralidad
    g = m.merge(
        base[["cod_mpio", "cuartil_lectura", "cuartil_rural"]],
        on="cod_mpio"
    )

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))

    # Crear gráficos de reprobación y deserción
    for k, (var, tit) in enumerate([
        ("reprobacion", "Reprobación (%)"),
        ("desercion", "Deserción (%)")
    ]):
        t = (
            g.groupby(
                ["anio", "cuartil_lectura"],
                observed=True
            )[var].median().unstack()
        )

        t.plot(ax=ax[k], marker="o")

        # Marcar el año 2020 en el gráfico
        ax[k].axvline(2020, color="gray", ls="--", lw=1)

        ax[k].set(
            title=f"{tit} según grupo de lectura",
            xlabel="",
            ylabel="Mediana municipal"
        )

        ax[k].set_xticks(sorted(g["anio"].unique()))

    # Crear un gráfico que compare ruralidad y resultados de lectura
    ax[2].scatter(
        base["pct_rural"],
        base["punt_lectura"],
        s=6,
        alpha=.4
    )

    ax[2].set(
        xlabel="% de estudiantes rurales (2017 y 2019)",
        ylabel="Promedio de lectura",
        title="Ruralidad y resultados de lectura"
    )

    plt.tight_layout()

    # Guardar los gráficos en la carpeta Processed
    plt.savefig(PROC / "graficos_pandemia.png", dpi=150)

    print(f"\nGráfico guardado en {PROC / 'graficos_pandemia.png'}")


# 6. Ejecutar todo el análisis

def main():
    # Revisar que los dos archivos necesarios existan
    for ruta in (RUTA_LIMPIO, RUTA_MAESTRO):
        if not ruta.exists():
            raise FileNotFoundError(
                f"No se encontró {ruta}. "
                "Ejecuta primero el código de limpieza y revisa "
                "que el archivo maestro esté en Data/Processed."
            )

    # Leer los datos de Saber 11
    s = pd.read_csv(
        RUTA_LIMPIO,
        dtype={"cod_mpio": str, "cod_depto": str}
    )

    # Trabajar solamente con los años 2017 y 2019
    s = s[s["anio"].isin(ANIOS_BASE)]

    # Calcular las diferencias por municipio y año
    construir_brechas(
        s, ["cod_mpio", "anio"]
    ).to_csv(
        PROC / "saber11_brechas_municipal.csv",
        index=False
    )

    # Calcular los resultados de los dos años juntos
    base = construir_brechas(s, ["cod_mpio"])

    # Tener en cuenta municipios con mínimo 30 estudiantes
    base = base[base["n_eval"] >= MIN_EVAL_MPIO]

    base.to_csv(
        PROC / "saber11_linea_base_municipal.csv",
        index=False
    )

    # Cargar los datos del MEN y calcular los cambios de la pandemia
    m = cargar_maestro()
    cambios = cambios_pandemia(m)

    # Unir los resultados de Saber 11 con los datos del MEN
    base = base.merge(
        cambios,
        on="cod_mpio",
        how="inner"
    )

    base.to_csv(
        PROC / "analisis_municipal.csv",
        index=False
    )

    print(
        f"Municipios con datos de Saber 11 y del MEN: {len(base):,}"
    )

    # Mostrar los resultados y ejecutar los modelos y gráficos
    base = describir(
        m,
        base,
        cambios,
        s.groupby("anio").size().to_dict()
    )

    modelos(base)
    graficos(m, base)


if __name__ == "__main__":
    main()

