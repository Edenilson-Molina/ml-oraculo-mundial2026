import pandas as pd
import numpy as np
from datetime import datetime

def load_and_clean_matches(
    matches_path: str, 
    ranking_path: str, 
    gdp_path: str = None, 
    pop_path: str = None
) -> pd.DataFrame:
    """
    Pipeline unificado de la Fase 1: Carga, filtra desde 1998, estandariza nombres,
    cruza temporalmente con Rankings FIFA y añade datos socioeconómicos.
    """
    # 1. Cargar conjuntos de datos
    df_matches = pd.read_csv(matches_path)
    df_ranking = pd.read_csv(ranking_path)
    
    # 2. Limpieza básica y conversión de fechas obligatoria para merge_asof
    df_matches = df_matches.dropna(subset=['date', 'home_team', 'away_team', 'home_score', 'away_score'])
    df_matches['date'] = pd.to_datetime(df_matches['date'])
    df_ranking['rank_date'] = pd.to_datetime(df_ranking['rank_date'])
    
    # Estandarizar strings
    df_matches['home_team'] = df_matches['home_team'].str.strip()
    df_matches['away_team'] = df_matches['away_team'].str.strip()
    df_ranking['country_full'] = df_ranking['country_full'].str.strip()
    
    # 3. Filtrar por época (1998 en adelante) para evitar arrastrar datos irrelevantes
    df_matches = df_matches[df_matches['date'] >= '1998-01-01'].copy()
    
    # 4. Diccionario de estandarización para romper los nulos (NaN) del Ranking
    mapeo_paises = {
        'USA': 'United States',
        'Korea Republic': 'South Korea',
        'Iran IR': 'Iran',
        'Cote d\'Ivoire': 'Ivory Coast',
        'DR Congo': 'Congo DR',
        'China PR': 'China',
        'Zambia': 'Zambia'
    }
    df_ranking['country_full'] = df_ranking['country_full'].replace(mapeo_paises)
    
    # Ambos datasets DEBEN estar ordenados por fecha para el merge_asof
    df_matches = df_matches.sort_values('date')
    df_ranking = df_ranking.sort_values('rank_date')
    
    # 5. Cruce asíncrono temporal (Asof Merge) para el Equipo Local
    df_final = pd.merge_asof(
        df_matches, 
        df_ranking[['rank_date', 'country_full', 'rank']], 
        left_on='date', 
        right_on='rank_date', 
        left_by='home_team', 
        right_by='country_full', 
        direction='backward'
    ).rename(columns={'rank': 'rank_home'}).drop(columns=['rank_date', 'country_full'])
    
    # 6. Cruce asíncrono temporal (Asof Merge) para el Equipo Visitante
    df_final = pd.merge_asof(
        df_final, 
        df_ranking[['rank_date', 'country_full', 'rank']], 
        left_on='date', 
        right_on='rank_date', 
        left_by='away_team', 
        right_by='country_full', 
        direction='backward'
    ).rename(columns={'rank': 'rank_away'}).drop(columns=['rank_date', 'country_full'])
    
    # Rellenar nulos remanentes en rankings con un valor neutral alto (Peor ranking posible)
    df_final['rank_home'] = df_final['rank_home'].fillna(211)
    df_final['rank_away'] = df_final['rank_away'].fillna(211)
    
    # 7. Inyección de Datos Socioeconómicos (Banco Mundial) si están presentes
    if gdp_path and pop_path:
        df_gdp = pd.read_csv(gdp_path) # Columnas esperadas: País, PIB
        df_pop = pd.read_csv(pop_path) # Columnas esperadas: País, Población
        
        # Merge para Local
        df_final = df_final.merge(df_gdp, left_on='home_team', right_on='País', how='left').rename(columns={'PIB': 'gdp_home'}).drop(columns=['País'])
        df_final = df_final.merge(df_pop, left_on='home_team', right_on='País', how='left').rename(columns={'Población': 'pop_home'}).drop(columns=['País'])
        
        # Merge para Visitante
        df_final = df_final.merge(df_gdp, left_on='away_team', right_on='País', how='left').rename(columns={'PIB': 'gdp_away'}).drop(columns=['País'])
        df_final = df_final.merge(df_pop, left_on='away_team', right_on='País', how='left').rename(columns={'Población': 'pop_away'}).drop(columns=['País'])
        
        # Llenar nulos socioeconómicos con la mediana para no arruinar XGBoost
        df_final['gdp_home'] = df_final['gdp_home'].fillna(df_final['gdp_home'].median())
        df_final['gdp_away'] = df_final['gdp_away'].fillna(df_final['gdp_away'].median())
        df_final['pop_home'] = df_final['pop_home'].fillna(df_final['pop_home'].median())
        df_final['pop_away'] = df_final['pop_away'].fillna(df_final['pop_away'].median())

    # 8. Crear columna objetivo (Target) alineada con la configuración del equipo
    # 2 = Gana Local, 1 = Empate, 0 = Gana Visitante
    conditions = [
        (df_final['home_score'] > df_final['away_score']),
        (df_final['home_score'] == df_final['away_score']),
        (df_final['home_score'] < df_final['away_score'])
    ]
    choices = [2, 1, 0]
    df_final['result'] = np.select(conditions, choices, default=1)
    
    return df_final

def apply_time_decay(df: pd.DataFrame, lambda_val: float = 0.24) -> pd.DataFrame:
    """
    Calcula el decaimiento exponencial temporal (Time Decay).
    """
    current_year = 2026
    df['years_ago'] = current_year - df['date'].dt.year
    df['weight'] = np.exp(-lambda_val * df['years_ago'])
    return df