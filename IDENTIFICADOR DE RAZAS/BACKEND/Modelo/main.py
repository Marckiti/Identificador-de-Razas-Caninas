import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import prediccion

# Carpeta donde el entrenamiento deja el modelo y sus métricas.
CARPETA_MODELO = Path(__file__).parent / "modelo"

# Tamaño máximo aceptado por imagen: 5 MB.
# Sin este límite, cualquiera podría enviar un archivo de 2 GB y tumbar el servidor.
TAMANO_MAXIMO = 5 * 1024 * 1024

# Formatos que el backend acepta.
TIPOS_PERMITIDOS = {"image/jpeg", "image/png", "image/webp"}


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    """
    Código que se ejecuta al ARRANCAR y al APAGAR el servidor.

    Todo lo que está ANTES del `yield` corre una vez, al levantar la aplicación:
    aquí es donde se carga el modelo a memoria. Todo lo que está DESPUÉS corre
    al apagarla.

    Por eso la primera petición no tarda 8 segundos: el modelo ya estaba listo.
    """
    prediccion.cargar_modelo()
    yield
    print("[main] Servidor detenido.")


app = FastAPI(
    title="API Detector de Razas de Perros",
    description="Identifica la raza de un perro a partir de una imagen usando MobileNetV2.",
    version="1.0.0",
    lifespan=ciclo_de_vida,
)

# --------------------------------------------------------------------
# CORS
# --------------------------------------------------------------------
# El navegador BLOQUEA por seguridad las peticiones entre dominios distintos.
# El frontend vivirá en Netlify (https://mi-app.netlify.app) y el backend en
# Render (https://mi-api.onrender.com): son dominios diferentes.
# Este middleware le dice al navegador "confío en ese origen, déjalo pasar".
# Sin esto, React recibiría un error de CORS y jamás vería la respuesta.
origenes = os.getenv("ORIGENES_PERMITIDOS", "http://localhost:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origen.strip() for origen in origenes],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def estado():
    """
    Ruta de salud. Sirve para dos cosas:
      - Comprobar rápido en el navegador que el backend está vivo.
      - Que Render sepa que el servicio arrancó bien.
    """
    return {
        "mensaje": "API del Detector de Razas activa",
        "modelo_cargado": prediccion.modelo_listo(),
        "razas_soportadas": len(prediccion.informacion_razas),
    }


@app.get("/razas")
def listar_razas():
    """Devuelve la ficha de todas las razas que el modelo puede reconocer."""
    return prediccion.informacion_razas


@app.get("/metricas")
def obtener_metricas():
    """
    Devuelve las métricas del modelo para mostrarlas dentro de la app:
      - evaluacion: accuracy global, F1 y precision/recall/F1 por raza
        (lo genera evaluar.py en modelo/metricas.json).
      - entrenamiento: accuracy y loss de cada época (modelo/historial_entrenamiento.json).

    Cumple el requisito de la rúbrica "mostrar métricas del entrenamiento".
    """
    ruta_evaluacion = CARPETA_MODELO / "metricas.json"
    ruta_historial = CARPETA_MODELO / "historial_entrenamiento.json"

    if not ruta_evaluacion.exists():
        raise HTTPException(
            status_code=503,
            detail="Métricas no disponibles. Ejecuta el entrenamiento y la evaluación primero.",
        )

    respuesta = {"evaluacion": json.loads(ruta_evaluacion.read_text(encoding="utf-8"))}
    if ruta_historial.exists():
        respuesta["entrenamiento"] = json.loads(ruta_historial.read_text(encoding="utf-8"))
    return respuesta


@app.get("/matriz-confusion")
def obtener_matriz_confusion():
    """Devuelve la imagen PNG de la matriz de confusión generada al evaluar."""
    ruta = CARPETA_MODELO / "matriz_confusion.png"
    if not ruta.exists():
        raise HTTPException(status_code=404, detail="Matriz de confusión no disponible.")
    return FileResponse(ruta, media_type="image/png")


@app.post("/predecir")
async def predecir_raza(imagen: UploadFile = File(...)):
    """
    Ruta principal. Recibe una imagen y devuelve la raza detectada.

    Cómo llega la imagen:
      React arma un FormData con el archivo bajo la clave "imagen" y hace
      un POST multipart/form-data. FastAPI lo detecta gracias al tipo
      `UploadFile = File(...)` y deja el archivo disponible aquí.

    `async def` permite que el servidor atienda otras peticiones mientras
    espera la lectura del archivo, en vez de quedarse bloqueado.
    """
    # 1. Validar el tipo de archivo ANTES de gastar memoria leyéndolo.
    if imagen.content_type not in TIPOS_PERMITIDOS:
        raise HTTPException(
            status_code=400,
            detail="Formato no admitido. Sube una imagen JPG, PNG o WEBP.",
        )

    # 2. Leer los bytes crudos del archivo.
    contenido = await imagen.read()

    if len(contenido) > TAMANO_MAXIMO:
        raise HTTPException(
            status_code=400,
            detail="La imagen supera los 5 MB. Usa una versión más liviana.",
        )

    # 3. Delegar todo el trabajo de IA al módulo prediccion.
    try:
        resultado = prediccion.predecir(contenido)
    except Exception as error:
        # Cualquier fallo inesperado (imagen corrupta, modelo sin cargar...)
        # se convierte en un error HTTP claro en vez de un 500 sin explicación.
        raise HTTPException(
            status_code=500,
            detail=f"No se pudo analizar la imagen: {error}",
        )

    # 4. FastAPI convierte automáticamente este diccionario en JSON
    #    y le pone la cabecera Content-Type: application/json.
    return resultado