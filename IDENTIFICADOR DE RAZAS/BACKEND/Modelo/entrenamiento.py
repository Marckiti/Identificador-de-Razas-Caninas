import argparse
import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers

# --------------------------------------------------------------------
# Configuración
# --------------------------------------------------------------------
CARPETA_BACKEND = Path(__file__).parent
CARPETA_IMAGENES = CARPETA_BACKEND / "imagenes"
CARPETA_MODELO = CARPETA_BACKEND / "modelo"

TAMANO_IMAGEN = (224, 224)  # tamaño nativo de MobileNetV2

# Cuántas imágenes procesa la red antes de ajustar sus pesos una vez.
# 32 es un buen equilibrio para 16 GB de RAM sin GPU. Si la laptop se queda
# sin memoria, bajar a 16.
TAMANO_LOTE = 32

# Cuántas veces la red ve el dataset completo. Son VALORES POR DEFECTO: se
# pueden cambiar al ejecutar (ver los parámetros --epocas-* más abajo).
EPOCAS_CONGELADO = 10    # etapa 1: solo se entrena la cabeza
EPOCAS_AJUSTE_FINO = 12  # etapa 2: se descongela la parte alta de MobileNetV2

# Cuántas capas del final de MobileNetV2 se descongelan en la etapa 2.
# Más capas = el modelo se adapta más a los perros, pero tarda más y puede
# sobreajustar. 60 es un buen equilibrio.
CAPAS_DESCONGELADAS = 60

# Learning rates por defecto de cada etapa.
LR_CABEZA = 1e-3   # alto: la cabeza arranca de cero
LR_AJUSTE = 1e-5   # 100x más chico: no destruir los pesos de ImageNet

# Qué porcentaje de las imágenes se aparta para validar (no se entrena con ellas).
PORCENTAJE_VALIDACION = 0.2

SEMILLA = 123  # fija el azar para que el experimento sea reproducible


def cargar_datos(tamano_lote=TAMANO_LOTE):
    """
    Construye los conjuntos de entrenamiento y validación desde las carpetas.

    image_dataset_from_directory hace tres cosas por nosotros:
      - Recorre las subcarpetas y usa su nombre como etiqueta.
      - Redimensiona cada imagen a 224x224.
      - Agrupa las imágenes en lotes de 32.

    IMPORTANTE: el conjunto de validación son imágenes que la red NUNCA usa
    para ajustar sus pesos. Sirven para medir si realmente aprendió a
    generalizar o solo memorizó (overfitting).
    """
    entrenamiento = tf.keras.utils.image_dataset_from_directory(
        CARPETA_IMAGENES,
        validation_split=PORCENTAJE_VALIDACION,
        subset="training",
        seed=SEMILLA,
        image_size=TAMANO_IMAGEN,
        batch_size=tamano_lote,
        label_mode="categorical",
    )

    validacion = tf.keras.utils.image_dataset_from_directory(
        CARPETA_IMAGENES,
        validation_split=PORCENTAJE_VALIDACION,
        subset="validation",
        seed=SEMILLA,
        image_size=TAMANO_IMAGEN,
        batch_size=tamano_lote,
        label_mode="categorical",
    )

    clases = entrenamiento.class_names

    # prefetch: mientras la CPU entrena con el lote actual, prepara el siguiente.
    # En una laptop sin GPU esto solo puede ahorrar tiempo real de entrenamiento.
    entrenamiento = entrenamiento.prefetch(tf.data.AUTOTUNE)
    validacion = validacion.prefetch(tf.data.AUTOTUNE)

    return entrenamiento, validacion, clases


def construir_modelo(cantidad_clases: int):
    """
    Arma la red usando Transfer Learning. Devuelve (modelo, base).

    La idea: MobileNetV2 ya fue entrenada con 1.2 millones de fotos (ImageNet).
    Sus capas iniciales ya saben detectar bordes, texturas, orejas, pelaje.
    No tiene sentido volver a aprender eso desde cero con 3.750 imágenes.
    Se aprovecha ese conocimiento y solo se enseña la parte final: qué raza es.
    """
    entrada = layers.Input(shape=(*TAMANO_IMAGEN, 3))

    # Aumento de datos: crea variaciones artificiales de cada foto (espejo,
    # rotación, zoom). La red ve un perro ligeramente distinto en cada época,
    # así que le cuesta más memorizar. Es la defensa más barata contra overfitting.
    # Estas capas SOLO se activan durante el entrenamiento; en producción se apagan solas.
    x = layers.RandomFlip("horizontal")(entrada)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomContrast(0.1)(x)

    # Normalización: MobileNetV2 espera valores entre -1 y 1, no entre 0 y 255.
    # Se incluye como CAPA dentro del modelo, no como paso externo, para que
    # prediccion.py no pueda equivocarse de escala.
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(x)

    base = tf.keras.applications.MobileNetV2(
        input_shape=(*TAMANO_IMAGEN, 3),
        include_top=False,   # se descarta el clasificador de 1000 clases de ImageNet
        weights="imagenet",  # se conservan los pesos ya aprendidos
    )
    base.trainable = False  # CAPAS CONGELADAS: sus pesos no se tocan en la etapa 1

    # training=False mantiene las capas de BatchNormalization en modo inferencia.
    # Omitir esto es el error más común del Transfer Learning y arruina la precisión.
    x = base(x, training=False)

    # De un mapa de (7, 7, 1280) a un vector de 1280 números: el "resumen" de la foto.
    x = layers.GlobalAveragePooling2D()(x)

    # Dropout: apaga al azar el 30% de las neuronas en cada paso de entrenamiento.
    # Obliga a la red a no depender de una sola pista y reduce el overfitting.
    x = layers.Dropout(0.3)(x)

    # Capa de salida: una neurona por raza. Softmax convierte los números en
    # probabilidades que suman 100%.
    salida = layers.Dense(cantidad_clases, activation="softmax")(x)

    modelo = tf.keras.Model(entrada, salida, name="detector_razas")

    # Se devuelve también `base` para poder descongelarla en la etapa 2 sin
    # tener que buscarla por su nombre interno de Keras.
    return modelo, base


def entrenar(epocas_cabeza=EPOCAS_CONGELADO, epocas_ajuste=EPOCAS_AJUSTE_FINO,
             tamano_lote=TAMANO_LOTE, lr_cabeza=LR_CABEZA, lr_ajuste=LR_AJUSTE,
             capas_descongeladas=CAPAS_DESCONGELADAS):
    """
    Entrena el modelo en dos etapas. Todos los parámetros son configurables
    (ver los argumentos de línea de comando en __main__).
    """
    print("[entrenamiento] Parámetros:")
    print(f"  épocas cabeza={epocas_cabeza}  épocas ajuste={epocas_ajuste}  batch={tamano_lote}")
    print(f"  lr cabeza={lr_cabeza}  lr ajuste={lr_ajuste}  capas descongeladas={capas_descongeladas}")

    entrenamiento, validacion, clases = cargar_datos(tamano_lote)
    print(f"\n[entrenamiento] {len(clases)} razas detectadas: {clases}\n")

    modelo, base = construir_modelo(len(clases))

    # ---------------- Etapa 1: cabeza nueva, base congelada ----------------
    # Learning rate "normal": la cabeza empieza desde cero y necesita avanzar rápido.
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_cabeza),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("=" * 60)
    print("ETAPA 1 - Entrenando solo la cabeza (base congelada)")
    print("=" * 60)
    historia1 = modelo.fit(
        entrenamiento,
        validation_data=validacion,
        epochs=epocas_cabeza,
    )

    # ---------------- Etapa 2: ajuste fino ----------------
    # Se descongelan las últimas N capas de MobileNetV2 para que se adapten a
    # las particularidades de los perros. Las primeras siguen congeladas porque
    # detectan bordes y texturas genéricos que ya están bien.
    base.trainable = True
    for capa in base.layers[:-capas_descongeladas]:
        capa.trainable = False

    # Learning rate mucho más pequeño. Con un valor alto se destruirían los
    # pesos de ImageNet en la primera época: sería como borrar lo ya aprendido.
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_ajuste),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("=" * 60)
    print(f"ETAPA 2 - Ajuste fino (últimas {capas_descongeladas} capas descongeladas)")
    print("=" * 60)
    historia2 = modelo.fit(
        entrenamiento,
        validation_data=validacion,
        epochs=epocas_ajuste,
        callbacks=[
            # Detiene el entrenamiento si la validación deja de mejorar durante
            # 4 épocas seguidas, y recupera los mejores pesos. Evita entrenar de
            # más y caer en overfitting.
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=4,
                restore_best_weights=True,
            )
        ],
    )

    # ---------------- Guardado ----------------
    CARPETA_MODELO.mkdir(exist_ok=True)
    modelo.save(CARPETA_MODELO / "modelo_razas.keras")

    # El orden de esta lista es sagrado: el índice 0 del modelo corresponde
    # a clases[0]. Si se pierde este archivo, las predicciones salen cruzadas.
    (CARPETA_MODELO / "clases.json").write_text(
        json.dumps(clases, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Historial de métricas por época: lo usa el endpoint /metricas del backend
    # para mostrar en la app cómo evolucionó el entrenamiento (requisito de la
    # rúbrica: "mostrar métricas del entrenamiento").
    guardar_historial(historia1, historia2, {
        "epocas_cabeza": epocas_cabeza,
        "epocas_ajuste": epocas_ajuste,
        "tamano_lote": tamano_lote,
        "lr_cabeza": lr_cabeza,
        "lr_ajuste": lr_ajuste,
        "capas_descongeladas": capas_descongeladas,
        "num_razas": len(clases),
    })

    print("\n[entrenamiento] Listo. Modelo e historial guardados en backend/modelo/")


def guardar_historial(historia1, historia2, config):
    """Junta las métricas por época de ambas etapas y las guarda en un JSON."""
    def a_lista(historia, etapa):
        filas = []
        h = historia.history
        for i in range(len(h["accuracy"])):
            filas.append({
                "etapa": etapa,
                "epoca": i + 1,
                "accuracy": round(float(h["accuracy"][i]), 4),
                "loss": round(float(h["loss"][i]), 4),
                "val_accuracy": round(float(h["val_accuracy"][i]), 4),
                "val_loss": round(float(h["val_loss"][i]), 4),
            })
        return filas

    epocas = a_lista(historia1, 1) + a_lista(historia2, 2)
    mejor_val = max(fila["val_accuracy"] for fila in epocas)
    datos = {
        "config": config,
        "mejor_val_accuracy": mejor_val,
        "epocas": epocas,
    }
    (CARPETA_MODELO / "historial_entrenamiento.json").write_text(
        json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    analizador = argparse.ArgumentParser(description="Entrena el detector de razas (parámetros configurables).")
    analizador.add_argument("--epocas-cabeza", type=int, default=EPOCAS_CONGELADO,
                            help=f"Épocas de la etapa 1 (por defecto {EPOCAS_CONGELADO}).")
    analizador.add_argument("--epocas-ajuste", type=int, default=EPOCAS_AJUSTE_FINO,
                            help=f"Épocas de la etapa 2 / ajuste fino (por defecto {EPOCAS_AJUSTE_FINO}).")
    analizador.add_argument("--batch", type=int, default=TAMANO_LOTE,
                            help=f"Tamaño de lote / batch size (por defecto {TAMANO_LOTE}).")
    analizador.add_argument("--lr-cabeza", type=float, default=LR_CABEZA,
                            help=f"Learning rate de la etapa 1 (por defecto {LR_CABEZA}).")
    analizador.add_argument("--lr-ajuste", type=float, default=LR_AJUSTE,
                            help=f"Learning rate del ajuste fino (por defecto {LR_AJUSTE}).")
    analizador.add_argument("--capas", type=int, default=CAPAS_DESCONGELADAS,
                            help=f"Capas a descongelar en la etapa 2 (por defecto {CAPAS_DESCONGELADAS}).")
    args = analizador.parse_args()

    entrenar(
        epocas_cabeza=args.epocas_cabeza,
        epocas_ajuste=args.epocas_ajuste,
        tamano_lote=args.batch,
        lr_cabeza=args.lr_cabeza,
        lr_ajuste=args.lr_ajuste,
        capas_descongeladas=args.capas,
    )