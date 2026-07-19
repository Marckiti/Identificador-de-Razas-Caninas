import argparse
import json
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path

CARPETA_BACKEND = Path(__file__).parent
CARPETA_IMAGENES = CARPETA_BACKEND / "imagenes"

API_COMMONS = "https://commons.wikimedia.org/w/api.php"

# Wikimedia EXIGE un User-Agent identificable con forma de contactar; si no,
# devuelve HTTP 429 y corta la descarga. (Política de uso de sus servidores.)
CABECERA = {
    "User-Agent": "DetectorRazasBot/1.0 (proyecto universitario de clasificacion de razas de perros)"
}

ANCHO_MINIATURA = 500       # px: suficiente para entrenar a 224x224 y liviano
BYTES_MINIMOS = 12_000      # descarta iconos, banderas y miniaturas rotas
ESPERA_ENTRE_DESCARGAS = 1.0  # segundos: ritmo educado para no gatillar el 429
MAX_CATEGORIAS = 60         # tope de subcategorías a visitar por raza

# Cada clave del proyecto -> categorías candidatas en Commons.
# Se prueban en orden; se recorren ellas y sus subcategorías.
CLAVE_A_CATEGORIAS = {
    "dachshund": ["Category:Dachshund"],
    "bulldog_ingles": ["Category:Bulldog", "Category:Bulldogs"],
    "pastor_australiano": ["Category:Australian Shepherd", "Category:Australian Shepherd Dogs"],
    "akita": ["Category:Akita Inu", "Category:American Akita", "Category:Akita (dog)"],
}


def pedir_json(parametros: dict) -> dict:
    """GET a la API de Commons con reintentos ante el bloqueo por robot (429)."""
    url = API_COMMONS + "?" + urllib.parse.urlencode(parametros)
    for intento in range(4):
        try:
            peticion = urllib.request.Request(url, headers=CABECERA)
            with urllib.request.urlopen(peticion, timeout=30) as respuesta:
                return json.loads(respuesta.read().decode("utf-8"))
        except Exception as error:
            if "429" in str(error) and intento < 3:
                time.sleep(2 * (intento + 1))  # backoff: 2s, 4s, 6s
                continue
            return {}
    return {}


def archivos_de_categoria(categoria: str) -> list[str]:
    """Devuelve las URLs de miniatura de los archivos de imagen de una categoría."""
    urls = []
    continuar = {}
    while True:
        parametros = {
            "action": "query",
            "format": "json",
            "generator": "categorymembers",
            "gcmtitle": categoria,
            "gcmtype": "file",
            "gcmlimit": "200",
            "prop": "imageinfo",
            "iiprop": "url|mime",
            "iiurlwidth": str(ANCHO_MINIATURA),
            **continuar,
        }
        datos = pedir_json(parametros)
        paginas = datos.get("query", {}).get("pages", {})
        for pagina in paginas.values():
            info = (pagina.get("imageinfo") or [{}])[0]
            if info.get("mime") in ("image/jpeg", "image/png"):
                url = info.get("thumburl") or info.get("url")
                if url:
                    urls.append(url)
        if "continue" in datos:
            continuar = datos["continue"]
        else:
            break
    return urls


def subcategorias_de(categoria: str) -> list[str]:
    """Devuelve los nombres de las subcategorías directas de una categoría."""
    datos = pedir_json({
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": categoria,
        "cmtype": "subcat",
        "cmlimit": "200",
    })
    miembros = datos.get("query", {}).get("categorymembers", [])
    return [m["title"] for m in miembros]


def recolectar_urls(categorias_raiz: list[str], objetivo: int) -> list[str]:
    """
    Recorre las categorías dadas y sus subcategorías (en anchura) juntando URLs
    de imágenes hasta reunir de sobra (objetivo * 2) o agotar el árbol.
    """
    visitadas: set[str] = set()
    cola = list(categorias_raiz)
    urls: list[str] = []

    while cola and len(urls) < objetivo * 2 and len(visitadas) < MAX_CATEGORIAS:
        categoria = cola.pop(0)
        if categoria in visitadas:
            continue
        visitadas.add(categoria)

        urls.extend(archivos_de_categoria(categoria))

        for subcategoria in subcategorias_de(categoria):
            if subcategoria not in visitadas:
                cola.append(subcategoria)

    # Quita duplicados conservando el orden.
    vistas = set()
    unicas = []
    for url in urls:
        if url not in vistas:
            vistas.add(url)
            unicas.append(url)
    return unicas


def descargar(url: str, destino: Path) -> bool:
    """Descarga una imagen con reintentos ante 429. True si quedó válida."""
    for intento in range(3):
        try:
            peticion = urllib.request.Request(url, headers=CABECERA)
            with urllib.request.urlopen(peticion, timeout=30) as respuesta:
                contenido = respuesta.read()
            break
        except Exception as error:
            if "429" in str(error) and intento < 2:
                time.sleep(2 * (intento + 1))
                continue
            return False

    if len(contenido) < BYTES_MINIMOS:
        return False
    destino.write_bytes(contenido)
    return True


def completar_una(clave: str, categorias: list[str], por_clase: int) -> int:
    """Descarga hasta `por_clase` fotos de una raza a imagenes/<clave>/."""
    destino = CARPETA_IMAGENES / clave

    
    if destino.exists():
        existentes = list(destino.glob("*.jpg"))
        if len(existentes) >= por_clase * 0.6:
            print(f"ya tenía {len(existentes)} fotos, no se toca")
            return len(existentes)
        shutil.rmtree(destino)
    destino.mkdir(parents=True)

    urls = recolectar_urls(categorias, por_clase)

    guardadas = 0
    for url in urls:
        if guardadas >= por_clase:
            break
        archivo = destino / f"{guardadas + 1:03d}.jpg"
        if descargar(url, archivo):
            guardadas += 1
        time.sleep(ESPERA_ENTRE_DESCARGAS)

    return guardadas


def main():
    analizador = argparse.ArgumentParser(description="Completa las razas ausentes desde Wikimedia Commons.")
    analizador.add_argument("--por-clase", type=int, default=150,
                            help="Máximo de fotos por raza (por defecto 150).")
    args = analizador.parse_args()

    print("=" * 64)
    print("FASE 2 (complemento) - Razas ausentes desde Wikimedia Commons")
    print("=" * 64)
    print("  (descarga lenta a propósito para respetar el límite de Wikimedia)")

    CARPETA_IMAGENES.mkdir(exist_ok=True)
    total = 0
    pocas = []

    for clave, categorias in CLAVE_A_CATEGORIAS.items():
        print(f"  {clave:<20} ", end="", flush=True)
        try:
            cantidad = completar_una(clave, categorias, args.por_clase)
        except Exception as error:
            print(f"ERROR: {error}")
            pocas.append(clave)
            continue
        total += cantidad
        marca = "" if cantidad >= 80 else "  <-- pocas, revisar"
        print(f"-> {cantidad} fotos{marca}")
        if cantidad < 80:
            pocas.append(clave)

    print("\n" + "=" * 64)
    print(f"  Total descargado: {total} fotos.")
    if pocas:
        print("\n  Razas con pocas fotos (revisar o reintentar):")
        for clave in pocas:
            print(f"    - {clave}")
    print("=" * 64)


if __name__ == "__main__":
    main()