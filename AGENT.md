
## Arquitectura
- EDA en notebooks/
- lógica productiva en src/
- modelos almacenados en models/

## Convenciones
- No entrenar modelos desde notebooks finales
- Toda transformación debe vivir en src/data/
- Las métricas deben guardarse en reports/metrics/

## Pipeline ML
1. Cargar datos
2. Limpiar nulos
3. Encoding
4. Split
5. Entrenamiento
6. Evaluación
7. Registro del modelo

## Reglas para agentes
- Nunca modificar datasets raw
- Mantener reproducibilidad
- Validar métricas antes de guardar modelos
- Detectar data leakage
- Mantener separación train/test

# Modelos
Los modelos entrenados deben guardarse usando:
joblib.dump()

Formato:
models/trained/model_name.pkl

# Validación mínima
- De acuerdo al tipo de modelo, se deben cumplir ciertos umbrales de métricas (ej. accuracy > 0.8)
- Validar que no haya data leakage (ej. features que no deberían estar en el set de entrenamiento)
- Verificar que el modelo no esté sobreajustado (ej. diferencia entre train/test > 0.1)

## Estructura
```project/
oraculo-mundial2026/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
│
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   ├── preprocess.py
│   │   └── validation.py
│   │
│   ├── features/
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   ├── train.py
│   │   ├── predict.py
│   │   ├── evaluate.py
│   │   └── registry.py
│   │
│   └── utils/
│       ├── logger.py
│       └── config.py
│
├── tests/
│
├── models/
│   └── trained/
│
├── reports/
│   ├── figures/
│   └── metrics/
│
├── requirements.txt
├── README.md
├── AGENT.md
├── .env
└── docker-compose.yml
```