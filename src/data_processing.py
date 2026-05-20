import pandas as pd
import numpy as np
from datetime import datetime

def load_and_clean_matches(filepath: str) -> pd.DataFrame:
    """
    Carga el dataset histórico de partidos y realiza una limpieza rigurosa.
    """
    # 1. Cargar datos (Simulando el formato estándar de Kaggle/GitHub)
    # Columnas esperadas: date, home_team, away_team, home_score, away_score, tournament, neutral
    df = pd.read_csv(filepath)
    
    # 2. Manejo de valores nulos y formatos inconsistentes
    df = df.dropna(subset=['date', 'home_team', 'away_team', 'home_score', 'away_score'])
    df['date'] = pd.to_datetime(df['date'])
    
    # Estandarizar nombres de equipos (quitar espacios en blanco extra)
    df['home_team'] = df['home_team'].str.strip()
    df['away_team'] = df['away_team'].str.strip()
    
    # 3. Crear columna objetivo (Target) a nivel de filas originales
    # 2 = Gana Local, 1 = Empate, 0 = Gana Visitante
    conditions = [
        (df['home_score'] > df['away_score']),
        (df['home_score'] == df['away_score']),
        (df['home_score'] < df['away_score'])
    ]
    choices = [2, 1, 0]
    df['result'] = np.select(conditions, choices, default=1)
    
    return df

def apply_time_decay(df: pd.DataFrame, lambda_val: float = 0.24) -> pd.DataFrame:
    """
    Calcula el decaimiento exponencial temporal (Time Decay).
    """
    # Tomamos como referencia el año actual del torneo (2026)
    current_year = 2026
    
    # Calcular la diferencia en años respecto a la fecha del partido
    df['years_ago'] = current_year - df['date'].dt.year
    
    # Aplicar la fórmula: e^(-lambda * t)
    df['weight'] = np.exp(-lambda_val * df['years_ago'])
    
    # Asegurar que los pesos no sean ridículamente cercanos a cero para partidos extremadamente viejos
    # (Opcional: puedes decidir filtrar partidos anteriores a cierto año para limpiar el dataset)
    df = df[df['years_ago'] <= 30].copy() # Filtramos últimos 30 años para mantener relevancia
    
    return df