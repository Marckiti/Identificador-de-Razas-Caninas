import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image

CARPETA_BACKEND = Path(__file__).parent
CARPETA_MODELO = CARPETA_BACKEND / "modelo"
CARPETA_IMAGENES = CARPETA_BACKEND / "imagenes"

RUTA_KERAS = CARPETA_MODELO / "modelo_razas.keras"
RUTA_TFLITE = CARPETA_MODELO / "modelo_razas.tflite"
RUTA_CLASES = CARPETA_MODELO / "clases.json"

TAMANO_IMAGEN = (224, 224)


def construir_modelo_inferencia(modelo_entrenado):
    """
    Arma una copia del modelo SIN las capas de aumento de datos.

    Las capas RandomFlip/RandomRotation/RandomZoom/RandomContrast solo actúan
    durante el entrenamiento; en inferencia no hacen nada, pero estorban al
    convertir a TFLite. Se reconstruye la red con solo lo necesario para
    predecir y se copian los pesos ya entrenados.
    """
    # La base MobileNetV2 es la única sub-red dentro del modelo; ya trae sus
    # pesos entrenados, así que se reutiliza tal cual.
    base = next(capa for capa in modelo_entrenado.layers if isinstance(capa, tf.keras.Model))
    densa_entrenada = modelo_entrenado.layers[-1]

    entrada = tf.keras.Input(shape=(*TAMANO_IMAGEN, 3))
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(entrada)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    salida = layers.Dense(densa_entrenada.units, activation="softmax")(x)

    inferencia = tf.keras.Model(entrada, salida, name="detector_razas_inferencia")
    # Copiar los pesos entrenados de la capa final.
    inferencia.layers[-1].set_weights(densa_entrenada.get_weights())
    return inferencia


def cargar_imagen(ruta: Path) -> np.ndarray:
    imagen = Image.open(ruta).convert("RGB").resize(TAMANO_IMAGEN)
    return np.expand_dims(np.asarray(imagen, dtype=np.float32), axis=0)


def main():
    if not RUTA_KERAS.exists():
        raise SystemExit(f"No existe {RUTA_KERAS}. Entrená primero: python entrenamiento.py")

    print("Cargando modelo entrenado...")
    modelo = tf.keras.models.load_model(RUTA_KERAS)

    print("Reconstruyendo versión solo-inferencia (sin aumento de datos)...")
    inferencia = construir_modelo_inferencia(modelo)

    print("Convirtiendo a TensorFlow Lite con cuantización dinámica...")
    convertidor = tf.lite.TFLiteConverter.from_keras_model(inferencia)
    convertidor.optimizations = [tf.lite.Optimize.DEFAULT]
    modelo_tflite = convertidor.convert()

    RUTA_TFLITE.write_bytes(modelo_tflite)
    tamano_mb = len(modelo_tflite) / 1_048_576
    print(f"Guardado {RUTA_TFLITE} ({tamano_mb:.1f} MB)")

    # ---- Verificación: el .tflite debe coincidir con el .keras ----
    print("\nVerificando que el .tflite predice igual que el .keras...")
    clases = json.loads(RUTA_CLASES.read_text(encoding="utf-8"))
    interprete = tf.lite.Interpreter(model_content=modelo_tflite)
    interprete.allocate_tensors()
    entrada_info = interprete.get_input_details()[0]
    salida_info = interprete.get_output_details()[0]

    # Toma una imagen de muestra de cada una de las primeras 5 razas.
    carpetas = sorted(p for p in CARPETA_IMAGENES.iterdir() if p.is_dir())[:5]
    coincidencias = 0
    for carpeta in carpetas:
        fotos = list(carpeta.glob("*.jpg"))
        if not fotos:
            continue
        tensor = cargar_imagen(fotos[0])

        pred_keras = int(np.argmax(modelo.predict(tensor, verbose=0)[0]))

        interprete.set_tensor(entrada_info["index"], tensor)
        interprete.invoke()
        pred_tflite = int(np.argmax(interprete.get_tensor(salida_info["index"])[0]))

        igual = "OK" if pred_keras == pred_tflite else "DIFERENTE"
        if pred_keras == pred_tflite:
            coincidencias += 1
        print(f"  {carpeta.name:<20} keras={clases[pred_keras]:<18} tflite={clases[pred_tflite]:<18} {igual}")

    print(f"\nCoincidencias: {coincidencias}/{len(carpetas)}")
    print("Listo. El backend ahora puede usar el .tflite sin TensorFlow completo.")


if __name__ == "__main__":
    main()