import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

# El intérprete de TFLite: liviano en producción, con respaldo en desarrollo.
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:  # solo pasa en la laptop, que tiene TensorFlow completo
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

# --------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------
CARPETA_BACKEND = Path(__file__).parent
RUTA_MODELO = CARPETA_BACKEND / "modelo" / "modelo_razas.tflite"
RUTA_CLASES = CARPETA_BACKEND / "modelo" / "clases.json"
RUTA_RAZAS = CARPETA_BACKEND / "razas.json"

# El modelo fue entrenado con imágenes de 224x224 píxeles.
TAMANO_IMAGEN = (224, 224)

# --------------------------------------------------------------------
# Estado en memoria
# --------------------------------------------------------------------
# Se cargan una sola vez, cuando arranca el servidor, y quedan en RAM.
_interprete = None
_clases: list[str] = []
_detalle_entrada = None   # info del tensor de entrada del modelo (índice, forma)
_detalle_salida = None    # info del tensor de salida

# La ficha informativa de cada raza. Es un JSON pequeño, se lee al importar.
informacion_razas: dict = json.loads(RUTA_RAZAS.read_text(encoding="utf-8"))


def cargar_modelo() -> None:
    """
    Carga el modelo TFLite y la lista de clases en memoria.

    Se llama UNA sola vez, desde el arranque de FastAPI (ver main.py).
    allocate_tensors() reserva la memoria que el modelo necesita para correr.
    """
    global _interprete, _clases, _detalle_entrada, _detalle_salida

    if not RUTA_MODELO.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {RUTA_MODELO}. "
            "Genéralo con: python entrenamiento.py && python convertir_tflite.py"
        )

    _interprete = Interpreter(model_path=str(RUTA_MODELO))
    _interprete.allocate_tensors()
    _detalle_entrada = _interprete.get_input_details()[0]
    _detalle_salida = _interprete.get_output_details()[0]
    _clases = json.loads(RUTA_CLASES.read_text(encoding="utf-8"))

    print(f"[prediccion] Modelo TFLite cargado. {len(_clases)} razas disponibles.")


def modelo_listo() -> bool:
    """Indica si el modelo ya está en memoria. Lo usa el endpoint de salud."""
    return _interprete is not None


def procesar_imagen(contenido: bytes) -> np.ndarray:
    """
    Convierte los bytes crudos que llegan por HTTP en el tensor que espera el modelo.

    Pasos:
      1. Abrir los bytes como imagen (Pillow).
      2. Convertir a RGB (descarta transparencia y arregla escala de grises).
      3. Redimensionar a 224x224.
      4. Convertir a NumPy float32 (0-255).
      5. Agregar la dimensión de lote: (224,224,3) -> (1,224,224,3).

    La normalización a valores entre -1 y 1 NO se hace aquí: está incluida
    DENTRO del modelo como una capa (ver convertir_tflite.py / entrenamiento.py).
    """
    imagen = Image.open(io.BytesIO(contenido)).convert("RGB")
    imagen = imagen.resize(TAMANO_IMAGEN)
    arreglo = np.asarray(imagen, dtype=np.float32)
    return np.expand_dims(arreglo, axis=0)


def predecir(contenido: bytes, cantidad_top: int = 3) -> dict:
    """
    Función principal del módulo: bytes de imagen -> resultado listo para JSON.

    Devuelve un diccionario con la raza ganadora, su confianza, la ficha de la
    raza y el Top 3 de candidatas (misma forma que antes).
    """
    if _interprete is None:
        raise RuntimeError("El modelo no está cargado.")

    tensor = procesar_imagen(contenido)

    # Correr el modelo: cargar la entrada, ejecutar y leer la salida.
    _interprete.set_tensor(_detalle_entrada["index"], tensor)
    _interprete.invoke()
    probabilidades = _interprete.get_tensor(_detalle_salida["index"])[0]

    # argsort ordena de menor a mayor y devuelve ÍNDICES.
    # [::-1] invierte el orden y [:cantidad_top] toma los mejores.
    indices_top = np.argsort(probabilidades)[::-1][:cantidad_top]

    top = []
    for indice in indices_top:
        clave = _clases[indice]
        ficha = informacion_razas.get(clave, {})
        top.append(
            {
                "clave": clave,
                "raza": ficha.get("nombre", clave),
                "confianza": round(float(probabilidades[indice]) * 100, 2),
            }
        )

    clave_ganadora = _clases[indices_top[0]]

    return {
        "raza": top[0]["raza"],
        "confianza": top[0]["confianza"],
        "informacion": informacion_razas.get(clave_ganadora, {}),
        "top": top,
    }