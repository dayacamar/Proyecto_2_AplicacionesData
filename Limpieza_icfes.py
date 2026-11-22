
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# Rutas de las carpetas del proyecto
BASE = Path(__file__).resolve().parent
RAW = BASE / "Data" / "Raw"
PROC = BASE / "Data" / "Processed"

PATRON_RAW = "Resultados*Saber*11*.csv"

ANIOS = [2017, 2018, 2019, 2020, 2021]
MIN_FILAS_ANIO = 300_000
CHUNK = 200_000

COLUMNAS = [
    "PERIODO", "ESTU_CONSECUTIVO",
    "COLE_COD_MCPIO_UBICACION", "COLE_MCPIO_UBICACION", "COLE_DEPTO_UBICACION",
    "COLE_NATURALEZA", "COLE_AREA_UBICACION", "FAMI_ESTRATOVIVIENDA",
    "PUNT_LECTURA_CRITICA", "PUNT_MATEMATICAS", "PUNT_SOCIALES_CIUDADANAS",
    "PUNT_C_NATURALES", "PUNT_INGLES", "PUNT_GLOBAL",
]

PARCIALES = [
    "PUNT_LECTURA_CRITICA", "PUNT_MATEMATICAS",
    "PUNT_SOCIALES_CIUDADANAS", "PUNT_C_NATURALES",
    "PUNT_INGLES"
]


def buscar_archivo():
    """Busca el archivo original en las carpetas del proyecto."""
    for carpeta in (RAW, BASE, Path.cwd()):
        encontrados = sorted(carpeta.glob(PATRON_RAW))
        if encontrados:
            return encontrados[-1]

    raise FileNotFoundError(
        f"No se encontró el archivo {PATRON_RAW}. "
        "Revisa que esté guardado en Data/Raw."
    )


def sin_tildes(serie):
    """Pasa los textos a mayúsculas y les quita las tildes."""
    return (
        serie.astype("string").str.strip().str.upper()
        .map(lambda x: unicodedata.normalize("NFKD", x)
             .encode("ascii", "ignore").decode()
             if isinstance(x, str) else x)
    )


def limpiar_chunk(df, descartes):
    n0 = len(df)

    # 1. Revisar que el periodo tenga el formato correcto
    ok = df["PERIODO"].str.fullmatch(r"\d{5}", na=False)
    descartes["periodo_invalido"] += int((~ok).sum())
    df = df[ok].copy()

    df["anio"] = df["PERIODO"].str[:4].astype(int)
    df = df[df["anio"].isin(ANIOS)]

    # 2. Revisar que el código del municipio tenga 5 números
    df["cod_mpio"] = df["COLE_COD_MCPIO_UBICACION"].str.strip().str.zfill(5)
    ok = df["cod_mpio"].str.fullmatch(r"\d{5}", na=False)
    descartes["municipio_invalido"] += int((~ok).sum())

    df = df[ok].copy()
    df["cod_depto"] = df["cod_mpio"].str[:2]

    # 3. Convertir los puntajes a números y revisar sus valores
    for c in PARCIALES + ["PUNT_GLOBAL"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    ok = df["PUNT_GLOBAL"].between(0, 500)
    descartes["puntaje_global_invalido"] += int((~ok).sum())
    df = df[ok].copy()

    for c in PARCIALES:
        df.loc[~df[c].between(0, 100), c] = np.nan

    # 4. Organizar la información del colegio
    nat = sin_tildes(df["COLE_NATURALEZA"])
    zona = sin_tildes(df["COLE_AREA_UBICACION"])

    df["es_no_oficial"] = nat.map({
        "OFICIAL": 0.0,
        "NO OFICIAL": 1.0
    }).astype("float32")

    df["es_rural"] = zona.map({
        "URBANO": 0.0,
        "RURAL": 1.0
    }).astype("float32")

    # 5. Sacar el número del estrato
    df["estrato"] = pd.to_numeric(
        df["FAMI_ESTRATOVIVIENDA"].str.extract(r"(\d)")[0],
        errors="coerce"
    ).astype("float32")

    descartes["leidas"] += n0

    return df


def main():
    # Crear la carpeta donde se guardarán los datos limpios
    PROC.mkdir(parents=True, exist_ok=True)

    ruta = buscar_archivo()
    print(f"Leyendo archivo: {ruta}")

    # Contar los registros que se van descartando
    descartes = {
        "leidas": 0,
        "periodo_invalido": 0,
        "municipio_invalido": 0,
        "puntaje_global_invalido": 0
    }

    partes = []

    # Leer el archivo por partes para manejar mejor los datos
    lector = pd.read_csv(
        ruta,
        usecols=COLUMNAS,
        dtype=str,
        chunksize=CHUNK,
        on_bad_lines="skip",
        encoding="utf-8"
    )

    for chunk in lector:
        partes.append(limpiar_chunk(chunk, descartes))

    df = pd.concat(partes, ignore_index=True)

    # Eliminar registros repetidos de un mismo estudiante y periodo
    antes = len(df)
    df = df.drop_duplicates(
        subset=["PERIODO", "ESTU_CONSECUTIVO"]
    )
    descartes["duplicados"] = antes - len(df)

    # Seleccionar las columnas que se van a guardar
    columnas_salida = [
        "PERIODO", "anio", "cod_mpio", "cod_depto",
        "es_no_oficial", "es_rural", "estrato"
    ] + PARCIALES + ["PUNT_GLOBAL"]

    df = df[columnas_salida].rename(columns={
        "PERIODO": "periodo",
        "PUNT_GLOBAL": "punt_global",
        "PUNT_LECTURA_CRITICA": "punt_lectura",
        "PUNT_MATEMATICAS": "punt_matematicas",
        "PUNT_SOCIALES_CIUDADANAS": "punt_sociales",
        "PUNT_C_NATURALES": "punt_naturales",
        "PUNT_INGLES": "punt_ingles"
    })

    # Guardar el archivo final con los datos organizados
    df.to_csv(PROC / "saber11_limpio.csv", index=False)

## COMPROBAR QUE FUNCIONE LA LIMPIEZA

    # Crear un resumen de los datos por año y periodo
    rep = df.groupby(["anio", "periodo"]).agg(
        filas=("punt_global", "size"),
        municipios=("cod_mpio", "nunique"),
        pct_rural=("es_rural", lambda x: round(x.mean() * 100, 1)),
        pct_no_oficial=("es_no_oficial", lambda x: round(x.mean() * 100, 1)),
        pct_sin_estrato=("estrato", lambda x: round(x.isna().mean() * 100, 1)),
    ).reset_index()

    # Guardar el resumen en un archivo CSV
    rep.to_csv(PROC / "saber11_reporte_calidad.csv", index=False)

    # Mostrar cuántas filas se descartaron durante la limpieza
    print("\n--- Datos descartados ---")
    for k, v in descartes.items():
        print(f"  {k}: {v:,}")

    print(f"\nTotal de filas limpias: {len(df):,}")

    # Mostrar cuántos registros hay por año
    print("\n--- Registros por año ---")
    por_anio = df.groupby("anio").size()
    print(por_anio.to_string())

    # Revisar si algún año tiene menos datos de los esperados
    incompletos = [
        a for a in ANIOS
        if por_anio.get(a, 0) < MIN_FILAS_ANIO
    ]

    if incompletos:
        print(
            f"\n[ALERTA] Estos años tienen menos de {MIN_FILAS_ANIO:,} filas: "
            f"{incompletos}"
        )
        print(
            "Es posible que falten datos. Revisa y descarga nuevamente "
            "la información de Saber 11 antes de analizar estos años."
        )

    # Mostrar dónde quedaron guardados los archivos
    print(f"\nArchivo guardado en: {PROC / 'saber11_limpio.csv'}")


if __name__ == "__main__":
    main()

"""
descargar_saber11.py  -  Descarga COMPLETA de Resultados únicos Saber 11 desde datos.gov.co

Por qué: el botón "Exportar CSV" del portal suele cortarse a mitad del archivo (tu CSV
terminó con un error 500 del servidor). Esta descarga pide los datos por periodo y por
páginas, reintenta si el servidor falla y puede reanudarse si se interrumpe.

Salida : Data/Raw/Resultados_Saber_11_completo_<fecha>.csv  (columnas en MAYÚSCULAS,
         igual que el CSV del portal, para que limpieza_icfes.py lo lea sin cambios)
Uso    : pip install requests pandas
         python Src/descargar_saber11.py
"""
import io
import shutil
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "Data" / "Raw"
PARTES = RAW / "saber11_partes"

URL = "https://www.datos.gov.co/resource/kgxf-xxbe.csv"   # API SODA del dataset kgxf-xxbe
ANIOS = [2017, 2018, 2019, 2020, 2021]
PAGINA = 50_000
APP_TOKEN = None   # opcional: token gratuito de datos.gov.co para menos límites de uso

CAMPOS = ["periodo", "estu_consecutivo", "cole_cod_mcpio_ubicacion", "cole_mcpio_ubicacion",
          "cole_depto_ubicacion", "cole_naturaleza", "cole_area_ubicacion",
          "fami_estratovivienda", "punt_lectura_critica", "punt_matematicas",
          "punt_sociales_ciudadanas", "punt_c_naturales", "punt_ingles", "punt_global"]

HEADERS = {"X-App-Token": APP_TOKEN} if APP_TOKEN else {}


def pedir(params, intentos=6):
    """GET con reintentos y espera creciente (el servidor a veces responde 500)."""
    for i in range(intentos):
        try:
            r = requests.get(URL, params=params, headers=HEADERS, timeout=180)
            if r.status_code == 400:
                return None            # consulta inválida (lo maneja quien llama)
            r.raise_for_status()
            return r.text
        except requests.RequestException as e:
            espera = 3 * 2 ** i
            print(f"    fallo ({e.__class__.__name__}); reintento en {espera}s")
            time.sleep(espera)
    raise RuntimeError("El servidor no respondió tras varios intentos. Intenta más tarde.")


def periodos_disponibles():
    txt = pedir({"$select": "periodo, count(*)", "$group": "periodo",
                 "$order": "periodo", "$limit": 1000})
    df = pd.read_csv(io.StringIO(txt), dtype={"periodo": str})
    df.columns = ["periodo", "total"]
    df = df[df["periodo"].str.fullmatch(r"\d{5}", na=False)]
    df["total"] = df["total"].astype(int)
    return df[df["periodo"].str[:4].astype(int).isin(ANIOS)].reset_index(drop=True)


def descargar_periodo(periodo, esperado):
    destino = PARTES / f"periodo_{periodo}.csv"
    if destino.exists():
        n = sum(1 for _ in open(destino, encoding="utf-8")) - 1
        if n == esperado:
            print(f"  {periodo}: ya descargado ({n:,} filas)")
            return n
    destino.unlink(missing_ok=True)

    filtros = [f"periodo = '{periodo}'", f"periodo = {periodo}"]   # texto o numérico
    where = filtros[0]
    offset, escritas = 0, 0
    while True:
        params = {"$select": ",".join(CAMPOS), "$where": where,
                  "$order": ":id", "$limit": PAGINA, "$offset": offset}
        txt = pedir(params)
        if txt is None and where == filtros[0]:
            where = filtros[1]                                   # reintenta sin comillas
            continue
        if txt is None:
            raise RuntimeError(f"Consulta inválida para el periodo {periodo}")
        df = pd.read_csv(io.StringIO(txt), dtype=str)
        if df.empty:
            break
        df.columns = [c.upper() for c in df.columns]
        df.to_csv(destino, mode="a", index=False, header=(offset == 0), encoding="utf-8")
        escritas += len(df)
        offset += PAGINA
        print(f"  {periodo}: {escritas:,} / {esperado:,}")
        if len(df) < PAGINA:
            break
    return escritas


def main():
    PARTES.mkdir(parents=True, exist_ok=True)
    per = periodos_disponibles()
    print("Periodos disponibles en el portal:")
    print(per.to_string(index=False))

    faltan = []
    for p, total in zip(per["periodo"], per["total"]):
        n = descargar_periodo(p, total)
        if n != total:
            faltan.append((p, n, total))

    if faltan:
        print("\n[ALERTA] Periodos con filas distintas a las que reporta el portal:")
        for p, n, t in faltan:
            print(f"  {p}: descargadas {n:,} de {t:,}")
        print("Vuelve a correr el script: continúa y completa lo que falte.")

    salida = RAW / f"Resultados_Saber_11_completo_{date.today():%Y%m%d}.csv"
    with open(salida, "wb") as out:
        primero = True
        for p in per["periodo"]:
            parte = PARTES / f"periodo_{p}.csv"
            if not parte.exists():
                continue
            with open(parte, "rb") as f:
                if not primero:
                    f.readline()               # salta el encabezado repetido
                shutil.copyfileobj(f, out)
            primero = False
    print(f"\nListo: {salida}")
    print("Siguiente paso: python Src/limpieza_icfes.py")


if __name__ == "__main__":
    main()