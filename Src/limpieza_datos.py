from pathlib import Path
import numpy as np
import pandas as pd

RAW_DIR = Path("Data/Raw")
PROCESSED_DIR = Path("Data/Processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ANIOS_OBJETIVO = [2017, 2018, 2019, 2020, 2021]

FILE_MEDIA = RAW_DIR / "MEN_ESTADISTICAS_EN_EDUCACION_EN_PREESCOLAR,_BÁSICA_Y_MEDIA_POR_MUNICIPIO_20261120.csv"
FILE_ICFES = RAW_DIR / "Resultados_únicos_Saber_11_20261121.csv"
FILE_SUPERIOR = RAW_DIR / "MEN_ESTADISTICAS_MATRICULA_POR_MUNICIPIOS_ES_20261120.csv"

def limpiar_codigo_divipola(serie: pd.Series, longitud: int = 5) -> pd.Series:
    """Estandariza códigos DANE como texto preservando ceros a la izquierda."""
    return (serie.astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(longitud))

def a_numerico(serie: pd.Series) -> pd.Series:
    """Convierte columnas numéricas en formato colombiano (punto miles, coma decimal, signo %)."""
    if pd.api.types.is_numeric_dtype(serie):
        return pd.to_numeric(serie, errors="coerce")
    limpia = (
        serie.astype(str)
        .str.strip()
        .str.replace("%", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(limpia, errors="coerce")
#En la siguiente funcion se limpia el dataset 1: Educacion basica y media por municipio

def procesar_basica_media(ruta: Path) -> pd.DataFrame:
    df = pd.read_csv(ruta, dtype=str)
    df.columns = df.columns.str.strip().str.upper()

    # Seleccionar columnas clave del diccionario MEN[cite: 7, 8, 9, 10]
    columnas_necesarias = [
        "AÑO",
        "CÓDIGO_MUNICIPIO",
        "MUNICIPIO",
        "CÓDIGO_DEPARTAMENTO",
        "DEPARTAMENTO",
        "POBLACIÓN_5_16",
        "COBERTURA_NETA_MEDIA",
        "COBERTURA_BRUTA_MEDIA",
        "TAMAÑO_PROMEDIO_DE_GRUPO",
        "SEDES_CONECTADAS_A_INTERNET",
        "DESERCIÓN_MEDIA",
        "APROBACIÓN_MEDIA",
        "REPROBACIÓN_MEDIA",
        "REPITENCIA_MEDIA",
    ]
    
    columnas_validas =[]
    for c in columnas_necesarias:
        if c in df.columns:
            columnas_validas.append(c)
    
    df = df[columnas_validas].copy()

    # Estandarizar llaves
    df["anio"] = a_numerico(df["AÑO"]).astype("Int64")
    df["cod_mpio"] = limpiar_codigo_divipola(df["CÓDIGO_MUNICIPIO"], 5)
    df["cod_depto"] = df["cod_mpio"].str[:2]

    # Filtrar ventana temporal 2017-2021 y códigos municipales válidos
    df = df[df["anio"].isin(ANIOS_OBJETIVO)& (~df["cod_mpio"].isin(["00000", "99999", "00nan"]))].copy()

    # Convertir métricas a numérico
    cols_metricas = [
        "POBLACIÓN_5_16",
        "COBERTURA_NETA_MEDIA",
        "COBERTURA_BRUTA_MEDIA",
        "TAMAÑO_PROMEDIO_DE_GRUPO",
        "SEDES_CONECTADAS_A_INTERNET",
        "DESERCIÓN_MEDIA",
        "APROBACIÓN_MEDIA",
        "REPROBACIÓN_MEDIA",
        "REPITENCIA_MEDIA",
    ]
    for col in cols_metricas:
        if col in df.columns:
            df[col.lower()] = a_numerico(df[col])

    df = df.drop_duplicates(subset=["cod_mpio", "anio"]).copy()

    
    comp_aprob = df["aprobación_media"].clip(0, 100)
    comp_internet = df["sedes_conectadas_a_internet"].clip(0, 100)
    comp_cobertura = df["cobertura_neta_media"].clip(0, 100)
    comp_retencion = (100 - df["deserción_media"]).clip(0, 100)
    comp_no_repitencia = (100 - df["repitencia_media"]).clip(0, 100)

    df["indice_calidad_media"] = pd.concat([
            comp_aprob,
            comp_internet,
            comp_cobertura,
            comp_retencion,
            comp_no_repitencia,
        ],
        axis=1,
    ).mean(axis=1)

    columnas_finales = [
        "cod_mpio",
        "cod_depto",
        "MUNICIPIO",
        "DEPARTAMENTO",
        "anio",
        "población_5_16",
        "cobertura_neta_media",
        "tamaño_promedio_de_grupo",
        "sedes_conectadas_a_internet",
        "deserción_media",
        "aprobación_media",
        "reprobación_media",
        "repitencia_media",
        "indice_calidad_media",
    ]
    
    columnas_retorno = []
    for c in columnas_finales:
        if c in df.columns:
            columnas_retorno.append(c)
    
    df = df[columnas_retorno]
    df = df.rename(columns={"MUNICIPIO": "municipio", "DEPARTAMENTO": "departamento"})
    return df
def procesar_saber11(ruta: Path) -> pd.DataFrame:
    # Inspeccionar primero solo los encabezados para filtrar las columnas válidas
    encabezados = pd.read_csv(ruta, nrows=0).columns
    
    cols_necesarias = [
        "PERIODO",
        "ESTU_CONSECUTIVO",
        "COLE_COD_MCPIO_UBICACION",
        "COLE_AREA_UBICACION",
        "COLE_NATURALEZA",
        "PUNT_GLOBAL",
        "PUNT_MATEMATICAS",
        "PUNT_LECTURA_CRITICA",
    ]
    
    columnas_validas = []
    for c in encabezados:
        if c.strip().upper() in cols_necesarias:
            columnas_validas.append(c)

    # Leer el archivo completo cargando únicamente las columnas filtradas
    df = pd.read_csv(ruta, usecols=columnas_validas, dtype=str)
    df.columns = df.columns.str.strip().str.upper()

    # Estandarizar llaves (extraer los primeros 4 dígitos de PERIODO para el año)[cite: 5]
    df["anio"] = a_numerico(df["PERIODO"].str.slice(0, 4)).astype("Int64")
    df["cod_mpio"] = limpiar_codigo_divipola(df["COLE_COD_MCPIO_UBICACION"], 5)

    # Filtrar ventana temporal 2017-2021 y códigos municipales válidos
    df = df[df["anio"].isin(ANIOS_OBJETIVO) & (~df["cod_mpio"].isin(["00000", "99999", "00nan"]))].copy()

    # Eliminar duplicados por examinando si existe la columna[cite: 5]
    if "ESTU_CONSECUTIVO" in df.columns:
        df = df.drop_duplicates(subset=["ESTU_CONSECUTIVO"]).copy()

    # Convertir puntajes a numérico
    cols_puntajes = ["PUNT_GLOBAL", "PUNT_MATEMATICAS", "PUNT_LECTURA_CRITICA"]
    for col in cols_puntajes:
        if col in df.columns:
            df[col] = a_numerico(df[col])

    # Crear variable indicadora de ruralidad (1 si es RURAL, 0 en otro caso)[cite: 5]
    if "COLE_AREA_UBICACION" in df.columns:
        es_rural_bool = df["COLE_AREA_UBICACION"].str.upper().str.strip() == "RURAL"
        df["es_rural"] = es_rural_bool.astype(int)
    else:
        df["es_rural"] = np.nan

    # Aislar el puntaje global exclusivo de colegios oficiales para comparar con MEN[cite: 8, 9, 10]
    if "COLE_NATURALEZA" in df.columns:
        es_oficial = df["COLE_NATURALEZA"].str.upper().str.strip() == "OFICIAL"
        df["punt_global_oficial"] = np.where(es_oficial, df["PUNT_GLOBAL"], np.nan)
    else:
        df["punt_global_oficial"] = np.nan

    # Agrupar microdatos a nivel municipio - año
    agg_icfes = (df.groupby(["cod_mpio", "anio"], as_index=False).agg(
            total_evaluados_s11=("PUNT_GLOBAL", "count"),
            punt_global_prom=("PUNT_GLOBAL", "mean"),
            punt_global_std=("PUNT_GLOBAL", "std"),
            punt_global_oficial_prom=("punt_global_oficial", "mean"),
            punt_matematicas_prom=("PUNT_MATEMATICAS", "mean"),
            punt_lectura_prom=("PUNT_LECTURA_CRITICA", "mean"),
            pct_evaluados_rural=("es_rural", "mean"),
        )
    )

    agg_icfes["pct_evaluados_rural"] = agg_icfes["pct_evaluados_rural"] * 100

    agg_icfes["brecha_total_vs_oficial_s11"] = (agg_icfes["punt_global_prom"] - agg_icfes["punt_global_oficial_prom"])
    return agg_icfes

def procesar_educacion_superior(ruta: Path) -> pd.DataFrame:
    df = pd.read_csv(ruta, dtype=str)
    df.columns = df.columns.str.strip().str.upper()

    col_mpio = ""
    col_anio = ""
    for c in df.columns:
        if "MUNICIPIO" in c and ("COD" in c or "CÓD" in c):
            col_mpio = c
        if "AÑO" in c or "A_O" in c:
            col_anio = c

    df["anio"] = a_numerico(df[col_anio]).astype("Int64")
    df["cod_mpio"] = limpiar_codigo_divipola(df[col_mpio], 5)

    df = df[df["anio"].isin(ANIOS_OBJETIVO) & (~df["cod_mpio"].isin(["00000", "99999", "00nan"]))].copy()

    cols_necesarias = [
        "cod_mpio",
        "anio",
        "TECNICA PROFESIONAL",
        "TECNOLOGICA",
        "UNIVERSITARIA",
        "IES CON OFERTA",
    ]
    
    columnas_validas = []
    for c in cols_necesarias:
        if c in df.columns:
            columnas_validas.append(c)
            
    df = df[columnas_validas].copy()

    cols_metricas = [
        "TECNICA PROFESIONAL",
        "TECNOLOGICA",
        "UNIVERSITARIA",
        "IES CON OFERTA",
    ]
    for col in cols_metricas:
        if col in df.columns:
            df[col] = a_numerico(df[col]).fillna(0)

    df_agg = (df.groupby(["cod_mpio", "anio"], as_index=False).agg(
            matricula_tecnica=("TECNICA PROFESIONAL", "sum"),
            matricula_tecnologica=("TECNOLOGICA", "sum"),
            matricula_universitaria=("UNIVERSITARIA", "sum"),
            ies_con_oferta=("IES CON OFERTA", "max"),
        )
    )

    df_agg["matricula_pregrado"] = (df_agg["matricula_tecnica"] + df_agg["matricula_tecnologica"] + df_agg["matricula_universitaria"])

    df_agg["tiene_oferta_ies"] = np.where(df_agg["ies_con_oferta"] > 0, "Con oferta IES", "Sin oferta IES")
    return df_agg

def ejecutar_etl():
    print("1. Procesando fuentes individuales...")
    df_media = procesar_basica_media(FILE_MEDIA)
    df_icfes = procesar_saber11(FILE_ICFES)
    df_sup = procesar_educacion_superior(FILE_SUPERIOR)

    print(f"   - Básica/Media: {len(df_media):,} filas (municipio-año)")
    print(f"   - Saber 11 agg: {len(df_icfes):,} filas (municipio-año)")
    print(f"   - Ed. Superior: {len(df_sup):,} filas (municipio-año)")

    tablas_a_validar = [
        ("Media", df_media),
        ("ICFES", df_icfes),
        ("Superior", df_sup),
    ]
    for nombre, tabla in tablas_a_validar:
        dups = tabla.duplicated(subset=["cod_mpio", "anio"]).sum()
        if dups > 0:
            raise ValueError(f"Error de integridad: {nombre} tiene {dups} llaves duplicadas.")

    # Left join tomando Básica y Media como base territorial completa
    maestro = df_media.merge(df_icfes, on=["cod_mpio", "anio"], how="left", indicator="merge_icfes")
    maestro = maestro.merge(
        df_sup,
        on=["cod_mpio", "anio"],
        how="left",
        indicator="merge_superior",
    )

    # Auditoría de cobertura del cruce para documentar en el proyecto
    print("\n2. Auditoría de integración:")
    print("   Cruce Media <-> Saber 11:\n", maestro["merge_icfes"].value_counts())
    print("   Cruce <-> Ed. Superior:\n", maestro["merge_superior"].value_counts())

    # Imputar 0 en matrícula e IES cuando un municipio no reporta oferta local
    cols_cero = [
        "matricula_tecnica",
        "matricula_tecnologica",
        "matricula_universitaria",
        "matricula_pregrado",
        "ies_con_oferta",
    ]
    for col in cols_cero:
        if col in maestro.columns:
            maestro[col] = maestro[col].fillna(0)

    maestro["tiene_oferta_ies"] = maestro["tiene_oferta_ies"].fillna("Sin oferta IES")

    # Indicador cruzado: matrícula de pregrado por cada 100 evaluados en Saber 11
    maestro["razon_pregrado_por_100_s11"] = np.where(
        maestro["total_evaluados_s11"] > 0,
        (maestro["matricula_pregrado"] / maestro["total_evaluados_s11"]) * 100,
        np.nan,
    )

    # Eliminar columnas auxiliares de auditoría y redondear columnas decimales
    maestro = maestro.drop(columns=["merge_icfes", "merge_superior"])
    cols_float = maestro.select_dtypes(include="float").columns
    for col in cols_float:
        maestro[col] = maestro[col].round(2)

    # ==========================================================================
    # AGREGACIÓN DEPARTAMENTAL (PARA MAPA NACIONAL DE TU COMPAÑERO)
    # ==========================================================================
    # Columnas auxiliares para ponderar puntajes por cantidad de estudiantes evaluados
    maestro["_pond_s11"] = maestro["punt_global_prom"] * maestro["total_evaluados_s11"]
    maestro["_pond_calidad"] = (
        maestro["indice_calidad_media"] * maestro["total_evaluados_s11"]
    )
    # Columna binaria (1 si el municipio tiene al menos una IES, 0 si no)
    maestro["_mpio_con_ies"] = (maestro["ies_con_oferta"] > 0).astype(int)

    depto = (
        maestro.groupby(["cod_depto", "departamento", "anio"], as_index=False)
        .agg(
            total_municipios=("cod_mpio", "nunique"),
            municipios_con_ies=("_mpio_con_ies", "sum"),
            total_evaluados_s11=("total_evaluados_s11", "sum"),
            matricula_pregrado=("matricula_pregrado", "sum"),
            desercion_media_prom=("deserción_media", "mean"),
            internet_sedes_prom=("sedes_conectadas_a_internet", "mean"),
            _suma_pond_s11=("_pond_s11", "sum"),
            _suma_pond_calidad=("_pond_calidad", "sum"),
        )
    )

    # Calcular promedios departamentales ponderados y porcentaje de cobertura de IES
    depto["punt_global_prom"] = (
        depto["_suma_pond_s11"] / depto["total_evaluados_s11"]
    ).round(2)
    depto["indice_calidad_media"] = (
        depto["_suma_pond_calidad"] / depto["total_evaluados_s11"]
    ).round(2)
    depto["pct_municipios_con_ies"] = (
        (depto["municipios_con_ies"] / depto["total_municipios"]) * 100
    ).round(1)

    # Limpiar columnas temporales de cálculo
    depto = depto.drop(columns=["_suma_pond_s11", "_suma_pond_calidad"])
    maestro = maestro.drop(columns=["_pond_s11", "_pond_calidad", "_mpio_con_ies"])

    # Guardar productos finales en Data/Processed/
    maestro.to_csv(PROCESSED_DIR / "maestro_municipal_2017_2021.csv", index=False)
    depto.to_csv(PROCESSED_DIR / "maestro_departamental_2017_2021.csv", index=False)

    print("\n3. Archivos generados exitosamente en Data/Processed/:")
    print("   - maestro_municipal_2017_2021.csv")
    print("   - maestro_departamental_2017_2021.csv")

if __name__ == "__main__":
    ejecutar_etl()
    #Lo que hay debajo de esto es una prueba que verifica que todo funciona#
    mpio = pd.read_csv(
        "Data/Processed/maestro_municipal_2017_2021.csv",
        dtype={"cod_mpio": str, "cod_depto": str}
    )
    depto = pd.read_csv(
        "Data/Processed/maestro_departamental_2017_2021.csv",
        dtype={"cod_depto": str}
    )

    print("\n=== 1. DIMENSIONES Y LLAVES ===")
    print("Filas y columnas municipal:", mpio.shape)
    print("Años presentes:", sorted(mpio["anio"].unique()))
    print("Longitud de cod_mpio (debe ser {5}):", set(mpio["cod_mpio"].str.len()))
    print("Longitud de cod_depto (debe ser {2}):", set(mpio["cod_depto"].str.len()))

    print("\n=== 2. RANGOS NUMÉRICOS (VERIFICAR QUE NO SE INFLARON DECIMALES) ===")
    cols_revisar = [
        "deserción_media",
        "aprobación_media",
        "sedes_conectadas_a_internet",
        "indice_calidad_media",
        "punt_global_prom",
        "matricula_pregrado",
        "razon_pregrado_por_100_s11",
    ]
    print(mpio[cols_revisar].describe().loc[["count", "mean", "min", "50%", "max"]])

    print("\n=== 3. COBERTURA DEL CRUCE ENTRE FUENTES ===")
    print("Municipios-año sin datos Saber 11:", mpio["punt_global_prom"].isna().sum())
    print("Distribución oferta IES:\n", mpio["tiene_oferta_ies"].value_counts())

    print("\n=== 4. MUESTRA REAL DE CONTROL (MEDELLÍN 2019 Y UN MUNICIPIO PEQUEÑO) ===")
    print(mpio[(mpio["cod_mpio"] == "05001") & (mpio["anio"] == 2019)][
        ["cod_mpio", "municipio", "anio", "indice_calidad_media", "punt_global_prom", "matricula_pregrado", "tiene_oferta_ies"]
    ])