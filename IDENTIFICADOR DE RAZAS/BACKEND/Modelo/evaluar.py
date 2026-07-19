import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")  # backend sin ventana: solo guarda la imagen a archivo
import matplotlib.pyplot as plt

# Se reutiliza EXACTAMENTE la misma configuración que entrenamiento.py para que
# el conjunto de validación sea el mismo (misma semilla, mismo % de reparto).
from entrenamiento import (
    CARPETA_IMAGENES,
    CARPETA_MODELO,
    TAMANO_IMAGEN,
    TAMANO_LOTE,
    PORCENTAJE_VALIDACION,
    SEMILLA,
)

RUTA_MODELO = CARPETA_MODELO / "modelo_razas.keras"
RUTA_CLASES = CARPETA_MODELO / "clases.json"


def cargar_validacion():
    """
    Reconstruye el MISMO conjunto de validación que se apartó al entrenar.

    Como usamos la misma semilla y el mismo validation_split que en
    entrenamiento.py, image_dataset_from_directory devuelve exactamente las
    mismas imágenes que la red nunca vio.
    """
    validacion = tf.keras.utils.image_dataset_from_directory(
        CARPETA_IMAGENES,
        validation_split=PORCENTAJE_VALIDACION,
        subset="validation",
        seed=SEMILLA,
        image_size=TAMANO_IMAGEN,
        batch_size=TAMANO_LOTE,
        label_mode="int",   # etiquetas como número de clase (0, 1, 2...)
        # OJO: hay que usar shuffle=True (el valor por defecto, igual que en
        # entrenamiento.py) para que el reparto 80/20 tome el MISMO conjunto de
        # validación que se usó al entrenar. Con shuffle=False el split saldría
        # de las últimas carpetas en orden alfabético (solo unas pocas razas) y
        # las métricas no representarían al modelo real.
        # y_real e y_predicho igual quedan alineados porque se emparejan lote a
        # lote dentro de la misma pasada.
        shuffle=True,
    )
    clases_dataset = validacion.class_names
    return validacion, clases_dataset


def predecir_todo(modelo, validacion):
    """Corre el modelo sobre toda la validación y junta etiquetas reales y predichas."""
    y_real, y_predicho = [], []
    for lote_imagenes, lote_etiquetas in validacion:
        probabilidades = modelo.predict(lote_imagenes, verbose=0)
        y_predicho.extend(np.argmax(probabilidades, axis=1))
        y_real.extend(lote_etiquetas.numpy())
    return np.array(y_real), np.array(y_predicho)


def guardar_matriz(y_real, y_predicho, clases):
    """Dibuja la matriz de confusión como un mapa de calor y la guarda en PNG."""
    matriz = confusion_matrix(y_real, y_predicho, labels=list(range(len(clases))))

    fig, ejes = plt.subplots(figsize=(12, 10))
    imagen = ejes.imshow(matriz, cmap="Greens")
    fig.colorbar(imagen, ax=ejes, fraction=0.046, pad=0.04)

    ejes.set_xticks(range(len(clases)))
    ejes.set_yticks(range(len(clases)))
    ejes.set_xticklabels(clases, rotation=90, fontsize=7)
    ejes.set_yticklabels(clases, fontsize=7)
    ejes.set_xlabel("Raza predicha")
    ejes.set_ylabel("Raza real")
    ejes.set_title("Matriz de confusión (conjunto de validación)")

    # Escribe el número dentro de cada celda para poder leerla en la exposición.
    umbral = matriz.max() / 2 if matriz.max() > 0 else 1
    for i in range(len(clases)):
        for j in range(len(clases)):
            valor = matriz[i, j]
            if valor == 0:
                continue
            ejes.text(j, i, str(valor), ha="center", va="center",
                      color="white" if valor > umbral else "black", fontsize=6)

    fig.tight_layout()
    destino = CARPETA_MODELO / "matriz_confusion.png"
    fig.savefig(destino, dpi=140)
    plt.close(fig)
    return destino


def main():
    if not RUTA_MODELO.exists():
        raise SystemExit(
            f"No existe el modelo en {RUTA_MODELO}.\n"
            "Entrená primero con: python entrenamiento.py"
        )

    print("Cargando modelo y validación...")
    modelo = tf.keras.models.load_model(RUTA_MODELO)
    clases = json.loads(RUTA_CLASES.read_text(encoding="utf-8"))
    validacion, clases_dataset = cargar_validacion()

    # Chequeo de seguridad: el orden de clases del modelo debe coincidir con el
    # que ve el dataset. Si no, las métricas saldrían cruzadas.
    if list(clases) != list(clases_dataset):
        raise SystemExit(
            "El orden de clases.json no coincide con las carpetas actuales.\n"
            "¿Cambiaste las carpetas de imagenes/ después de entrenar? Reentrená."
        )

    print("Prediciendo sobre el conjunto de validación...")
    y_real, y_predicho = predecir_todo(modelo, validacion)

    # --- Reporte de texto: precision / recall / F1 por raza ---
    # labels=range(25) fuerza a que el reporte incluya SIEMPRE las 25 razas,
    # aunque alguna no apareciera en este lote de validación.
    reporte = classification_report(
        y_real, y_predicho, labels=list(range(len(clases))),
        target_names=clases, digits=3, zero_division=0,
    )
    exactitud = float((y_real == y_predicho).mean())
    encabezado = (
        "REPORTE DE EVALUACIÓN - Detector de razas\n"
        f"Imágenes de validación: {len(y_real)}\n"
        f"Exactitud global (accuracy): {exactitud * 100:.2f}%\n"
        + "=" * 64 + "\n"
    )
    print("\n" + encabezado + reporte)

    (CARPETA_MODELO / "reporte_evaluacion.txt").write_text(
        encabezado + reporte, encoding="utf-8"
    )

    # --- Métricas en JSON (las consume el endpoint /metricas del backend) ---
    reporte_dict = classification_report(
        y_real, y_predicho, labels=list(range(len(clases))),
        target_names=clases, zero_division=0, output_dict=True,
    )
    por_clase = [
        {
            "raza": raza,
            "precision": round(reporte_dict[raza]["precision"], 3),
            "recall": round(reporte_dict[raza]["recall"], 3),
            "f1": round(reporte_dict[raza]["f1-score"], 3),
            "soporte": int(reporte_dict[raza]["support"]),
        }
        for raza in clases
    ]
    metricas = {
        "accuracy": round(exactitud, 4),
        "n_validacion": int(len(y_real)),
        "macro_f1": round(reporte_dict["macro avg"]["f1-score"], 3),
        "por_clase": por_clase,
    }
    (CARPETA_MODELO / "metricas.json").write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- Matriz de confusión en imagen ---
    ruta_png = guardar_matriz(y_real, y_predicho, clases)

    print("\nGuardado:")
    print(f"  - {CARPETA_MODELO / 'reporte_evaluacion.txt'}")
    print(f"  - {ruta_png}")


if __name__ == "__main__":
    main()