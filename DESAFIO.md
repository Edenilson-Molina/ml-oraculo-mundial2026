# Plan de Desarrollo — Desafío Práctico

# “El Oráculo del Balón - Predicción Mundial 2026”

## 1. Introducción

El objetivo de este proyecto es desarrollar un sistema de Machine Learning capaz de estimar la probabilidad de que una selección nacional gane la Copa Mundial FIFA 2026, utilizando modelos predictivos a nivel de partido y una simulación Monte Carlo del torneo completo.

El proyecto seguirá un enfoque basado en ciencia de datos aplicada al deporte, integrando:

- Obtención y limpieza de datos históricos.
- Ingeniería de características avanzadas.
- Modelos de clasificación probabilística.
- Simulación del torneo 2026 bajo el nuevo formato de 48 equipos.
- Evaluación mediante métricas de calibración.

Este plan está diseñado específicamente para cumplir con la rúbrica de evaluación y alcanzar el nivel **Excelente (90-100%)**.

---

# 2. Objetivos

## Objetivo General

Construir un pipeline reproducible de Machine Learning capaz de:

1. Predecir probabilidades de victoria/empate/derrota entre selecciones.
2. Simular el Mundial 2026 mediante Monte Carlo.
3. Generar un Top-5 de selecciones con mayor probabilidad de campeonar.

---

## Objetivos Específicos

- Integrar múltiples fuentes de datos deportivas históricas y actuales.
- Diseñar características relevantes como ELO, forma reciente y calidad de plantilla.
- Entrenar y comparar múltiples modelos de clasificación.
- Implementar decay temporal para ponderar partidos recientes.
- Simular correctamente el nuevo formato FIFA 2026.
- Evaluar calibración probabilística mediante Log Loss y Brier Score.
- Generar un reporte técnico estilo IEEE.

---

# 3. Arquitectura General del Proyecto

```
Obtención de Datos        
        ↓
Limpieza y Estandarización        
        ↓
EDA (Análisis Exploratorio)        
        ↓
Feature Engineering        
        ↓
Entrenamiento de Modelos        
        ↓
Evaluación y Validación        
        ↓
Simulación Monte Carlo        
        ↓
Resultados Finales + Reporte
```

---

# 4. Obtención de Datos

## 4.1 Fuentes de Datos

Para cumplir la rúbrica se utilizarán múltiples fuentes de datos.

### Dataset Principal

- Resultados internacionales históricos (1872–2024)
- GitHub: martj42/international_results

### Dataset Complementario

- Ranking FIFA histórico.
- Elo Ratings internacionales.
- Datos históricos de mundiales.
- Estadísticas avanzadas opcionales (xG, posesión, tiros).

### APIs Opcionales

- API-Football
- Football-Data.org

### Web Scraping Opcional

- FBref
- Transfermarkt

---

## 4.2 Variables Iniciales

Variables disponibles:

- Fecha
- Equipo local
- Equipo visitante
- Goles
- Torneo
- País
- Localía
- Resultado

---

# 5. Limpieza y Preparación de Datos

## Actividades

### 5.1 Limpieza

- Eliminación de duplicados.
- Conversión de fechas.
- Normalización de nombres de selecciones.
- Tratamiento de valores nulos.

### 5.2 Transformaciones

- Conversión del resultado:
- Victoria local
- Empate
- Victoria visitante

### 5.3 Filtrado Temporal

- Mayor peso a partidos recientes.
- Eliminación de partidos excesivamente antiguos si afectan rendimiento.

---

# 6. Análisis Exploratorio de Datos (EDA)

## Objetivos del EDA

- Analizar distribución de resultados.
- Identificar selecciones dominantes.
- Analizar comportamiento histórico por continentes.
- Evaluar frecuencia de empates.
- Analizar correlaciones entre rankings y victorias.

---

## Visualizaciones Propuestas

- Histograma de goles.
- Heatmap de correlación.
- Distribución de victorias por país.
- Evolución temporal del ELO.
- Comparación FIFA Ranking vs Win Rate.

---

# 7. Ingeniería de Características (Features)

Para alcanzar la máxima puntuación se implementarán al menos 6 features relevantes.

## Features Principales

### 1. Diferencia ELO

```
ELO_Equipo_A - ELO_Equipo_B
```

### 2. Diferencia Ranking FIFA

```
Ranking_A - Ranking_B
```

### 3. Forma Reciente

- Victorias últimos 5 partidos.
- Goles promedio recientes.

### 4. Diferencia de Goles Recientes

```
GF recientes - GC recientes
```

### 5. Localía

- Partido neutral.
- País anfitrión.
- Ventaja local.

### 6. Decay Temporal

Se aplicará ponderación exponencial:

$w = e^{-\lambda t}$$A$$k$$y = A e^{-kt} \approx 6 e^{-0.6t}$Sliders update the exponential decay curve and half-life.ytDonde:

- $t$ = antigüedad del partido
- $\lambda$ = tasa de degradación

### 7. Calidad de Plantilla

- Valor de mercado promedio.
- Cantidad de jugadores en ligas top.

### 8. Métricas Avanzadas

- xG
- Posesión
- Tiros al arco

---

# 8. Modelos de Machine Learning

La rúbrica exige al menos dos modelos.

## Modelos Seleccionados

### Modelo 1 — Regresión Logística

Ventajas:

- Interpretabilidad.
- Buen baseline probabilístico.

### Modelo 2 — XGBoost

Ventajas:

- Excelente rendimiento en datos tabulares.
- Manejo robusto de relaciones no lineales.

### Modelo 3 (Opcional) — Random Forest

Para comparación adicional.

---

# 9. Validación y Ajuste de Hiperparámetros

## Estrategia

### Validación Cruzada

```
Stratified K-Fold Cross Validation
```

### Ajuste de Hiperparámetros

- GridSearchCV
- RandomizedSearchCV

---

## Hiperparámetros a Optimizar

### XGBoost

- learning_rate
- max_depth
- n_estimators
- subsample

### Logistic Regression

- C
- penalty

---

# 10. Métricas de Evaluación

La rúbrica exige métricas probabilísticas.

## Métricas

### Log Loss

$LogLoss = -\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{M} y_{ij}\log(p_{ij})$

### Brier Score

$BS = \frac{1}{N}\sum_{i=1}^{N}(f_i-o_i)^2$

### Métricas Secundarias

- Accuracy
- F1-score
- Matriz de confusión

---

# 11. Aspectos Críticos del Proyecto

## 11.1 Nuevo Formato Mundial 2026

Se implementará:

- 48 equipos
- 12 grupos de 4
- Clasificación de 32 equipos

---

## 11.2 Simulación Monte Carlo

La rúbrica exige ≥10,000 simulaciones.

## Proceso

1. Simular cada partido usando probabilidades del modelo.
2. Resolver fase de grupos.
3. Resolver eliminación directa.
4. Repetir 10,000 veces.
5. Contabilizar campeones.

---

## 11.3 Manejo del Desbalance

Se entrenará a nivel de partido:

- Gana
- Empata
- Pierde

NO a nivel de campeón.

---

## 11.4 Limitaciones

Se incluirá discusión sobre:

- Lesiones
- Cambios tácticos
- Falta de datos de debutantes
- Incertidumbre deportiva

---

# 12. Validación Histórica

Se realizará una simulación retrospectiva:

- Mundial 2018
- Mundial 2022

Objetivo:

- Comparar predicciones vs resultados reales.

---

# 13. Estructura del Repositorio

```
worldcup-2026-predictor/
│
├── data/
├── notebooks/
├── src/
│   ├── preprocessing/
│   ├── features/
│   ├── models/
│   ├── simulation/
│   └── evaluation/
├── reports/
├── requirements.txt
├── README.md
└── main.py
```

---

# 14. Tecnologías

## Lenguaje

- Python 3.10+

## Librerías

- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- scipy

---

# 15. Cronograma de Desarrollo

| Semana | Actividad |
| --- | --- |
| 1 | Obtención y limpieza de datos |
| 2 | EDA y visualizaciones |
| 3 | Ingeniería de características |
| 4 | Entrenamiento de modelos |
| 5 | Ajuste de hiperparámetros |
| 6 | Evaluación y calibración |
| 7 | Simulación Monte Carlo |
| 8 | Reporte técnico y GitHub |

---

# 16. Entregables Finales

## 1. Repositorio GitHub

Incluye:

- Código modular.
- README.
- requirements.txt

## 2. Reporte Técnico

Formato estilo IEEE:

- Introducción
- Metodología
- Resultados
- Limitaciones
- Conclusiones

## 3. Notebook de Simulación

Con:

- Top-5 selecciones.
- Probabilidades finales.
- Intervalos de confianza.

---

# 17. Estrategia para Obtener la Máxima Calificación

## Para obtener nivel “Excelente” se debe:

✅ Utilizar múltiples fuentes de datos. ✅ Implementar mínimo 6 features relevantes. ✅ Usar al menos 2 modelos. ✅ Aplicar GridSearch o RandomSearch. ✅ Implementar decay temporal. ✅ Simular correctamente el Mundial 2026. ✅ Ejecutar ≥10,000 simulaciones Monte Carlo. ✅ Evaluar con Log Loss y Brier Score. ✅ Validar con mundiales históricos. ✅ Mantener código modular y reproducible. ✅ Elaborar reporte técnico estilo IEEE.

---

# 18. Conclusión

Este proyecto no solo busca predecir un campeón mundial, sino construir un sistema probabilístico robusto, reproducible y alineado con prácticas reales de ciencia de datos aplicada al deporte.

La combinación de:

- Machine Learning,
- ingeniería de características,
- simulación Monte Carlo,
- y análisis estadístico,

permitirá desarrollar una solución sólida que cumpla completamente con la rúbrica del desafío.