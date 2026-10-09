#  Traveler Insights UPTC 2026 — Fine-Tuning de LLaMA con CRISP-ML(Q)

Este repositorio contiene la solución oficial del equipo para la competencia **Traveler Insights UPTC 2026**. El proyecto aborda la clasificación de sentimiento en reseñas turísticas utilizando un modelo **TinyLlama-1.1B** ajustado mediante **QLoRA (4-bit)**, documentado bajo la metodología estándar de ingeniería de software para ML: **CRISP-ML(Q)**.

##  Aspectos Clave del Proyecto
- **Arquitectura:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0` + QLoRA (PEFT).
- **Métrica Alcanzada (F1-Macro):** `0.7239` en validación local (superando al baseline clásico de TF-IDF + Logistic Regression con `0.6376`).
- **Accuracy Global:** `91.46%`.
- **Eficiencia:** Modelo cuantizado en 4 bits con adaptador de solo `12.08 MB` y latencia de inferencia de `22.28 ms/ejemplo`.
- **Pipeline Completo:** Incluye validación local estratificada, análisis de incertidumbre con Bootstrap, auditoría de 14 errores, pruebas de estrés de robustez y simulación de Data Drift.

---

##  Estructura del Repositorio
```text
├── notebook/
│   └── team-llama1140.ipynb      # Notebook de Kaggle con el pipeline completo (Fase 1 a 6)
├── data/
│   ├── train.csv                 # Dataset de entrenamiento
│   ├── test.csv                  # Dataset de prueba
│   └── submission.csv            # Predicciones finales generadas para Kaggle
├── report/
│   └── Informe_CRISP_ML_Q.pdf    # Documento técnico final
├── README.md                     # Descripción del proyecto
└── requirements.txt              # Dependencias del proyecto