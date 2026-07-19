import subprocess
import sys
import time
from pathlib import Path

CARPETA = Path(__file__).parent

# (etiqueta, script, argumentos). El orden importa: cada paso depende del anterior.
PASOS = [
    ("Preparar dataset (Stanford Dogs)", "preparar_datos.py", ["--por-clase", "250"]),
    ("Completar razas (Wikimedia Commons)", "completar_razas.py", ["--por-clase", "150"]),
    ("Entrenar el modelo", "entrenamiento.py", []),
    ("Evaluar el modelo", "evaluar.py", []),
    ("Convertir a TensorFlow Lite", "convertir_tflite.py", []),
]


def correr(etiqueta: str, script: str, argumentos: list[str]) -> bool:
    print(f"\n{'#' * 70}", flush=True)
    print(f"### PASO: {etiqueta}", flush=True)
    print(f"### -> python {script} {' '.join(argumentos)}", flush=True)
    print(f"{'#' * 70}\n", flush=True)

    inicio = time.time()
    # -u = salida sin buffer, para que el log se vea en tiempo real.
    proceso = subprocess.run(
        [sys.executable, "-u", str(CARPETA / script), *argumentos],
        cwd=CARPETA,
    )
    minutos = (time.time() - inicio) / 60

    if proceso.returncode != 0:
        print(f"\n### FALLO en '{etiqueta}' (código {proceso.returncode}). Pipeline detenido.", flush=True)
        return False

    print(f"\n### OK: '{etiqueta}' terminó en {minutos:.1f} min.", flush=True)
    return True


def main():
    print("=" * 70, flush=True)
    print("PIPELINE COMPLETO - Detector de razas (Fases 2 a 4)", flush=True)
    print("=" * 70, flush=True)

    inicio_total = time.time()
    for etiqueta, script, argumentos in PASOS:
        if not correr(etiqueta, script, argumentos):
            sys.exit(1)

    total = (time.time() - inicio_total) / 60
    print(f"\n{'=' * 70}", flush=True)
    print(f"### PIPELINE COMPLETO en {total:.1f} min. Modelo entrenado y evaluado.", flush=True)
    print("### Siguiente: probar el backend con  uvicorn main:app --reload", flush=True)
    print(f"{'=' * 70}", flush=True)


if __name__ == "__main__":
    main()