# Identificador de razas caninas

Aplicación web que reconoce la raza de un perro a partir de una foto, usando una
red neuronal **MobileNetV2** entrenada con Transfer Learning sobre **25 razas**.
Precisión ~92% en validación.

Proyecto de la materia **Fundamentos de Inteligencia Artificial**.


**ENDPOINTS**

| `GET` | `/` | Estado del servicio (si el modelo cargó) |
| `GET` | `/razas` | Ficha de las 25 razas |
| `POST` | `/predecir` | Recibe una imagen y devuelve la predicción (raza, confianza, Top 3) |
| `GET` | `/metricas` | Métricas del modelo (accuracy, precision/recall/F1 por raza, historial) |
| `GET` | `/matriz-confusion` | Imagen PNG de la matriz de confusión |
| `GET` | `/docs` | Documentación interactiva |
