## El Oraculo del Balon - Mundial 2026

Pipeline de modelado y evaluacion para prediccion probabilistica de partidos y simulacion del Mundial 2026.
La data ya esta procesada en `data/processed/final_dataset.csv` y no se regenera por tiempo.

## Que hace el proyecto
1. Carga el dataset ya procesado.
2. Genera features para clasificacion probabilistica de partidos.
3. Entrena y guarda modelos en `models/trained/`.
4. Evalua metricas en `reports/metrics/`.
5. Usa el modelo entrenado para simular el Mundial 2026 con Monte Carlo.

## Estructura clave
- `src/`: logica productiva (features, modelos, evaluacion, simulacion)
- `models/trained/`: modelos entrenados con `joblib`
- `reports/metrics/`: metricas guardadas por modelo
- `reports/figures/`: visualizaciones del EDA

## Flujo recomendado
### 1. Preparar el entorno
- Activar el entorno virtual del proyecto.
- Instalar dependencias.

### 2. Revisar datos
- Verificar `data/processed/final_dataset.csv`.
- No regenerar `data/raw/` salvo que quieras rehacer toda la limpieza.

### 3. Explorar datos
- Abrir `notebooks/01_eda.ipynb` para analisis exploratorio.
- Abrir `notebooks/02_preprocessing.ipynb` para revisar features y leakage.

### 4. Entrenar modelos
- Ejecutar el pipeline de entrenamiento desde `src/models/train.py`.
- Se generan `logreg.pkl` y `xgb.pkl`.

### 5. Simular el torneo
- Abrir `notebooks/04_simulation.ipynb`.
- Cargar el modelo entrenado, construir probabilidades y correr Monte Carlo.

## Comandos
### Entrenar modelos
```bash
python -m src.models.train
```

### Entrenar con busqueda de hiperparametros
```bash
python -m src.models.train --search --n-iter 8
```

### Ejecutar simulacion
- Abre `notebooks/04_simulation.ipynb` y ejecuta las celdas en orden.
- El notebook carga `models/trained/xgb.pkl` por defecto.

### Usar desde Python
```python
from src.models.train import train_models

train_models(use_search=True, n_iter=8)
```

## Que archivo se crea en cada etapa
- Entrenamiento: `models/trained/logreg.pkl`, `models/trained/xgb.pkl`
- Metricas: `reports/metrics/logreg_metrics.json`, `reports/metrics/xgb_metrics.json`
- Figuras: `reports/figures/`

## Quick start
1. Crear y activar entorno
2. Instalar dependencias: `pip install -r requirements.txt`
3. Entrenar modelos: `python -m src.models.train`

## Output esperado
- Modelos: `models/trained/logreg.pkl`, `models/trained/xgb.pkl`
- Metricas: `reports/metrics/logreg_metrics.json`, `reports/metrics/xgb_metrics.json`

## Notas
- No se entrena desde notebooks finales (ver `AGENT.md`).
- Se evita data leakage y se mantiene separacion train/test.
- La simulacion usa el modelo entrenado para producir probabilidades de partido.
