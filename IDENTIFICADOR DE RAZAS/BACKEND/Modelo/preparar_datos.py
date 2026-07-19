import argparse
import random
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

# --------------------------------------------------------------------
# Rutas y constantes
# --------------------------------------------------------------------
CARPETA_BACKEND = Path(__file__).parent
CARPETA_IMAGENES = CARPETA_BACKEND / "imagenes"
CARPETA_DESCARGAS = CARPETA_BACKEND / "descargas"  # cachea el tar y lo extraído

# El dataset solo se sirve por HTTP plano (la versión HTTPS devuelve 404).
URL_IMAGENES = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
TAMANO_TAR = 793_579_520  # bytes exactos, para saber si una descarga quedó a medias

SEMILLA = 123  # mismo azar que entrenamiento.py: elección de fotos reproducible

# --------------------------------------------------------------------
# Mapa: clave del proyecto  ->  cómo se llama esa raza dentro de Stanford Dogs
# --------------------------------------------------------------------
# Stanford Dogs nombra sus carpetas como "n02099712-Labrador_retriever".
# Aquí solo guardamos la parte legible que hay que buscar. El match se hace
# por subcadena y sin distinguir mayúsculas contra las carpetas REALES del
# archivo, así que no dependemos de acordarnos del código "n02099712".
#
# Cuando una raza tiene varias variantes en Stanford (el caniche: toy, miniatura
# y estándar) se listan las tres y se fusionan en una sola carpeta.
#
# Las 5 últimas casi seguro NO existen en Stanford Dogs. Se dejan con un nombre
# específico (no genérico) para que NO capturen por error otra raza parecida
# —"Australian_shepherd" no debe confundirse con el "Australian_terrier" que sí
# está en el dataset—. Si el match da cero, el script las marca como manuales.
CLAVE_A_STANFORD = {
    "labrador_retriever": ["Labrador_retriever"],
    "golden_retriever": ["golden_retriever"],
    "pastor_aleman": ["German_shepherd"],
    "caniche": ["toy_poodle", "miniature_poodle", "standard_poodle"],
    "beagle": ["beagle"],
    "rottweiler": ["Rottweiler"],
    "yorkshire_terrier": ["Yorkshire_terrier"],
    "boxer": ["boxer"],
    "husky_siberiano": ["Siberian_husky"],
    "doberman": ["Doberman"],
    "gran_danes": ["Great_Dane"],
    "pug": ["pug"],
    "chihuahua": ["Chihuahua"],
    "border_collie": ["Border_collie"],
    "schnauzer_miniatura": ["miniature_schnauzer"],
    "shih_tzu": ["Shih-Tzu"],
    "cocker_spaniel": ["cocker_spaniel"],
    "san_bernardo": ["Saint_Bernard"],
    "chow_chow": ["chow"],
    "pomerania": ["Pomeranian"],
    # --- Casi seguro ausentes en Stanford Dogs: aporte manual ---
    "dachshund": ["dachshund"],
    "bulldog_frances": ["French_bulldog"],
    "bulldog_ingles": ["English_bulldog"],
    "pastor_australiano": ["Australian_shepherd"],
    "akita": ["Akita"],
}


# Recuerda el último porcentaje impreso para NO escribir en cada bloque de 8 KB.
# Imprimir y hacer flush miles de veces por segundo frena mucho la descarga.
_ultimo_porcentaje = -1


def barra_progreso(bloques, tamano_bloque, total):
    """Callback de urlretrieve: pinta el avance solo cuando sube un 1%."""
    global _ultimo_porcentaje
    descargado = bloques * tamano_bloque
    total = total if total > 0 else TAMANO_TAR
    porcentaje = int(min(descargado / total * 100, 100))
    if porcentaje == _ultimo_porcentaje:
        return
    _ultimo_porcentaje = porcentaje
    mb = descargado / 1_048_576
    mb_total = total / 1_048_576
    sys.stdout.write(f"\r  Descargando... {porcentaje:3d}%  ({mb:6.0f} / {mb_total:.0f} MB)")
    sys.stdout.flush()


def obtener_tar(ruta_tar_manual: str | None) -> Path:
    """
    Devuelve la ruta a images.tar, descargándolo si hace falta.

    - Si el usuario pasó --tar, se usa ese archivo y no se descarga nada.
    - Si ya existe una descarga previa completa en descargas/, se reutiliza.
    - Si no, se descarga desde Stanford (~757 MB; puede tardar varios minutos).
    """
    if ruta_tar_manual:
        ruta = Path(ruta_tar_manual)
        if not ruta.exists():
            sys.exit(f"[error] No existe el archivo indicado: {ruta}")
        return ruta

    CARPETA_DESCARGAS.mkdir(exist_ok=True)
    destino = CARPETA_DESCARGAS / "images.tar"

    # Una descarga previa se considera válida solo si pesa exactamente lo esperado.
    # Así una descarga cortada a la mitad no envenena el proceso.
    if destino.exists() and destino.stat().st_size == TAMANO_TAR:
        print(f"  Reutilizando descarga previa: {destino}")
        return destino

    print(f"  Origen: {URL_IMAGENES}")
    print("  Son ~757 MB. Con conexión normal esto tarda unos minutos.\n")
    try:
        urllib.request.urlretrieve(URL_IMAGENES, destino, reporthook=barra_progreso)
    except Exception as error:
        sys.exit(
            f"\n[error] Falló la descarga: {error}\n"
            "Puedes bajar images.tar a mano desde el navegador y luego correr:\n"
            "    python preparar_datos.py --tar RUTA\\images.tar"
        )
    print("\n  Descarga completa.")
    return destino


def extraer_tar(ruta_tar: Path) -> Path:
    """
    Descomprime images.tar y devuelve la carpeta que contiene las subcarpetas
    de razas (Stanford las mete todas dentro de una carpeta 'Images').

    Si ya estaba extraído de una corrida anterior, no lo repite.
    """
    carpeta_extraida = CARPETA_DESCARGAS / "Images"
    if carpeta_extraida.exists() and any(carpeta_extraida.iterdir()):
        print(f"  Reutilizando archivo ya extraído: {carpeta_extraida}")
        return carpeta_extraida

    CARPETA_DESCARGAS.mkdir(exist_ok=True)
    print("  Descomprimiendo (esto también tarda un poco)...")
    with tarfile.open(ruta_tar) as tar:
        # filter='data' evita rutas maliciosas fuera de la carpeta destino.
        tar.extractall(CARPETA_DESCARGAS, filter="data")
    return carpeta_extraida


def carpetas_por_clave(carpeta_fuente: Path) -> dict[str, list[Path]]:
    """
    Cruza las 25 claves del proyecto con las carpetas reales del dataset.

    Para cada clave devuelve la lista de carpetas de Stanford que le
    corresponden (puede ser 0, 1 o varias). El match es por subcadena e
    insensible a mayúsculas contra el nombre real de cada carpeta.
    """
    reales = [p for p in carpeta_fuente.iterdir() if p.is_dir()]
    asignadas: set[Path] = set()
    resultado: dict[str, list[Path]] = {}

    for clave, patrones in CLAVE_A_STANFORD.items():
        coincidencias = []
        for carpeta in reales:
            nombre = carpeta.name.lower()
            if carpeta in asignadas:
                continue
            if any(patron.lower() in nombre for patron in patrones):
                coincidencias.append(carpeta)
                asignadas.add(carpeta)
        resultado[clave] = coincidencias

    return resultado


def copiar_imagenes(clave: str, origenes: list[Path], por_clase: int, azar: random.Random) -> int:
    """
    Copia hasta `por_clase` fotos de las carpetas de origen a imagenes/<clave>/,
    renombrándolas 001.jpg, 002.jpg... Devuelve cuántas copió.

    La carpeta destino se vacía antes para que reejecutar el script no acumule
    duplicados ni mezcle sobras de una corrida anterior.
    """
    destino = CARPETA_IMAGENES / clave
    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir(parents=True)

    fotos: list[Path] = []
    for origen in origenes:
        fotos.extend(sorted(origen.glob("*.jpg")))

    azar.shuffle(fotos)
    fotos = fotos[:por_clase]

    for indice, foto in enumerate(fotos, start=1):
        shutil.copy(foto, destino / f"{indice:03d}.jpg")

    return len(fotos)


def main():
    analizador = argparse.ArgumentParser(description="Prepara el dataset de la Fase 2.")
    analizador.add_argument("--por-clase", type=int, default=200,
                            help="Máximo de fotos por raza (por defecto 200).")
    analizador.add_argument("--tar", default=None,
                            help="Ruta a un images.tar ya descargado (evita descargar).")
    analizador.add_argument("--fuente", default=None,
                            help="Ruta a las carpetas ya extraídas (evita descargar y descomprimir).")
    args = analizador.parse_args()

    print("=" * 64)
    print("FASE 2 - Preparación del dataset (Stanford Dogs -> 25 razas)")
    print("=" * 64)

    # 1. Conseguir las carpetas de razas, por el camino más corto disponible.
    if args.fuente:
        carpeta_fuente = Path(args.fuente)
        if not carpeta_fuente.exists():
            sys.exit(f"[error] No existe la carpeta indicada: {carpeta_fuente}")
        print(f"  Usando carpetas ya extraídas: {carpeta_fuente}")
    else:
        ruta_tar = obtener_tar(args.tar)
        carpeta_fuente = extraer_tar(ruta_tar)

    # 2. Emparejar cada clave del proyecto con sus carpetas reales.
    mapa = carpetas_por_clave(carpeta_fuente)

    # 3. Copiar fotos y llevar la cuenta.
    azar = random.Random(SEMILLA)
    CARPETA_IMAGENES.mkdir(exist_ok=True)

    resumen: list[tuple[str, int, str]] = []
    faltantes: list[str] = []

    print("\n  Copiando imágenes por raza:")
    for clave, origenes in mapa.items():
        if not origenes:
            # Sin coincidencia en Stanford. Si la carpeta ya tiene fotos (p. ej.
            # bajadas antes con completar_razas.py), se RESPETAN: re-ejecutar
            # este script no debe borrar esas imágenes.
            destino = CARPETA_IMAGENES / clave
            existentes = list(destino.glob("*.jpg")) if destino.exists() else []
            if existentes:
                resumen.append((clave, len(existentes), "ya presente (Commons)"))
                print(f"    {clave:<22} -> {len(existentes):>3}   (ya presente, no se toca)")
                continue

            destino.mkdir(parents=True, exist_ok=True)
            (destino / "LEEME_APORTAR.txt").write_text(
                "Esta raza no está en Stanford Dogs.\n"
                "Ejecuta completar_razas.py o coloca aquí ~150 fotos (jpg) y borra este archivo.\n",
                encoding="utf-8",
            )
            faltantes.append(clave)
            resumen.append((clave, 0, "MANUAL - no está en Stanford Dogs"))
            print(f"    {clave:<22} ->   0   (aporte manual)")
            continue

        cantidad = copiar_imagenes(clave, origenes, args.por_clase, azar)
        nombres = ", ".join(o.name.split("-", 1)[-1] for o in origenes)
        resumen.append((clave, cantidad, nombres))
        print(f"    {clave:<22} -> {cantidad:>3}   ({nombres})")

    # 4. Resumen final.
    total = sum(c for _, c, _ in resumen)
    con_datos = sum(1 for _, c, _ in resumen if c > 0)

    print("\n" + "=" * 64)
    print(f"  Razas con imágenes automáticas: {con_datos} / {len(CLAVE_A_STANFORD)}")
    print(f"  Imágenes copiadas en total:     {total}")
    if faltantes:
        print("\n  FALTA APORTE MANUAL en estas razas (carpeta creada, vacía):")
        for clave in faltantes:
            print(f"    - {clave}")
        print("\n  Pon ~150 fotos jpg en cada una (Google Imágenes, Kaggle,")
        print("  Wikimedia Commons) y borra su LEEME_APORTAR.txt.")
    print("\n  Cuando todas las carpetas tengan fotos, ejecuta:")
    print("    python entrenamiento.py")
    print("=" * 64)


if __name__ == "__main__":
    main()