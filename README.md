## El Oraculo del Balon - Mundial 2026

Pipeline de modelado y evaluacion para prediccion probabilistica de partidos y simulacion del Mundial 2026.
La data ya esta procesada en `data/processed/final_dataset.csv` y no se regenera por tiempo.

## Estructura clave
- `src/`: logica productiva (features, modelos, evaluacion, simulacion)
- `models/trained/`: modelos entrenados con `joblib`
- `reports/metrics/`: metricas guardadas por modelo
- `reports/figures/`: visualizaciones del EDA

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
