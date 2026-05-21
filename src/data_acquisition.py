"""
Módulo de adquisición de datos para el proyecto "El Oráculo del Balón"

Este módulo obtiene datos de 3 fuentes:
1. Partidos históricos internacionales (desde GitHub)
2. Ranking FIFA histórico (desde Kaggle - necesitas descargarlo manualmente)
3. Calidad de plantilla de selecciones (desde LanusStats)

Uso:
    python src/data_acquisition.py
"""

import os
import re
from urllib.parse import quote

import pandas as pd

# ============================================
# CONFIGURACIÓN DE RUTAS
# ============================================

# Obtener la raíz del proyecto
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")

# Crear directorios si no existen
os.makedirs(DATA_RAW, exist_ok=True)
os.makedirs(DATA_PROCESSED, exist_ok=True)


# ============================================
# FUENTE 1: PARTIDOS HISTÓRICOS (DESDE GITHUB)
# ============================================

def download_historical_matches():
    """
    Descarga el dataset de partidos internacionales desde GitHub.
    
    Fuente: martj42/international_results
    URL: https://github.com/martj42/international_results
    
    Returns:
        pandas.DataFrame: DataFrame con los partidos históricos
    """
    print("📥 Descargando partidos históricos...")
    
    url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    output_path = os.path.join(DATA_RAW, "01_historical_matches.csv")
    
    try:
        df = pd.read_csv(url)
        df.to_csv(output_path, index=False)
        print(f"   ✅ Guardado: {output_path}")
        print(f"   📊 Registros: {len(df)} partidos")
        print(f"   📅 Período: {df['date'].min()} a {df['date'].max()}")
        return df
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


# ============================================
# FUENTE 2: RANKING FIFA (AVISO: DESCARGA MANUAL)
# ============================================

def check_fifa_rankings():
    """
    Verifica si el archivo de ranking FIFA existe.
    Si no existe, indica al usuario cómo descargarlo.
    
    El archivo debe descargarse manualmente desde Kaggle:
    https://www.kaggle.com/datasets/cashncarry/fifaworldranking
    """
    expected_path = os.path.join(DATA_RAW, "02_fifa_rankings.csv")
    
    if os.path.exists(expected_path):
        print(f"✅ Ranking FIFA encontrado: {expected_path}")
        df = pd.read_csv(expected_path)
        print(f"   📊 Registros: {len(df)}")
        return df
    else:
        print(f"⚠️ Archivo no encontrado: {expected_path}")
        print(f"\n📌 INSTRUCCIÓN:")
        print(f"   1. Ve a https://www.kaggle.com/datasets/cashncarry/fifaworldranking")
        print(f"   2. Descarga el archivo CSV")
        print(f"   3. Guárdalo como '{expected_path}'")
        print(f"   4. Vuelve a ejecutar este script")
        return None


# ============================================
# FUENTE 3: CALIDAD DE PLANTILLA (LANUSSTATS)
# ============================================

TEAM_NAME_ALIASES = {
    "USA": "United States",
    "United States of America": "United States",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Cote d'Ivoire": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "Curaçao": "Curacao",
    "Cabo Verde": "Cape Verde",
    "North Macedonia": "Macedonia",
    "DR Congo": "Congo DR",
    "Cape Verde Islands": "Cape Verde",
    "China PR": "China",
    "Korea DPR": "North Korea",
}


def _parse_int(value):
    if value is None:
        return None
    digits = re.sub(r"[^0-9]", "", str(value))
    return int(digits) if digits else None


def _parse_market_value(value):
    if not value:
        return None
    cleaned = (
        str(value)
        .replace("Total market value", "")
        .replace("€", "")
        .replace("$", "")
        .replace("£", "")
        .strip()
    )
    match = re.search(r"([0-9.,]+)\s*([kmb]n?|m|bn)?", cleaned, re.IGNORECASE)
    if not match:
        return None
    number = float(match.group(1).replace(",", ""))
    unit = (match.group(2) or "").lower()
    if unit in {"bn", "b"}:
        return number * 1_000_000_000
    if unit == "m":
        return number * 1_000_000
    if unit == "k":
        return number * 1_000
    return number


def _load_teams_from_rankings(top_n=48):
    ranking_path = os.path.join(DATA_RAW, "02_fifa_rankings.csv")
    if not os.path.exists(ranking_path):
        return None
    df = pd.read_csv(ranking_path)
    if "rank_date" not in df.columns or "country_full" not in df.columns or "rank" not in df.columns:
        return None
    latest_date = df["rank_date"].max()
    latest = df[df["rank_date"] == latest_date].copy()
    latest = latest.sort_values("rank", ascending=True)
    return latest["country_full"].head(top_n).tolist()


def _find_club_box(soup):
    for box in soup.select("div.box"):
        h2 = box.find("h2")
        if not h2:
            continue
        title = h2.get_text(strip=True).lower()
        if "clubs" in title or "vereine" in title or "teams" in title:
            return box
    return None


def _find_team_row(rows, query):
    def normalize_name(text):
        return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()

    def is_youth_team(text):
        return bool(re.search(r"\bu\d{2}\b", text.casefold()))

    query_norm = normalize_name(query)
    best_row = None
    best_score = -1

    for row in rows:
        link = row.find("a", href=True)
        if not link:
            continue
        club_name = link.get_text(strip=True)
        tds = row.find_all("td")
        competition = tds[3].get_text(" ", strip=True) if len(tds) > 3 else ""
        club_norm = normalize_name(club_name)
        comp_norm = normalize_name(competition)

        score = 0
        if club_norm == query_norm:
            score += 5
        if club_norm == f"{query_norm} world cup":
            score += 6
        if club_norm.startswith(query_norm):
            score += 2
        if query_norm in club_norm:
            score += 1
        if "world cup" in comp_norm:
            score += 3
        if "qualification" in comp_norm or "playoffs" in comp_norm or "qualifiers" in comp_norm:
            score += 2
        if is_youth_team(club_name) or is_youth_team(competition):
            score -= 5

        if score > best_score:
            best_score = score
            best_row = row

    return best_row


def _extract_team_row_data(row):
    if row is None:
        return None
    tds = row.find_all("td")
    if len(tds) < 7:
        return None
    club_name = tds[1].get_text(" ", strip=True)
    squad_size = _parse_int(tds[5].get_text(" ", strip=True))
    market_value = _parse_market_value(tds[6].get_text(" ", strip=True))
    team_link = row.find("a", href=True)
    team_href = team_link["href"] if team_link else None
    team_id_match = re.search(r"/verein/(\d+)", team_href or "")
    team_id = team_id_match.group(1) if team_id_match else None
    return {
        "transfermarkt_team_name": club_name,
        "transfermarkt_team_id": team_id,
        "transfermarkt_team_href": team_href,
        "squad_size": squad_size,
        "market_value_eur": market_value,
    }


def _extract_team_header_stats(soup):
    stats = {}
    for li in soup.select("li.data-header__label"):
        text = li.get_text(" ", strip=True)
        if ":" not in text:
            continue
        label, value = text.split(":", 1)
        label = label.strip().lower()
        value = value.strip()
        if label == "squad size":
            stats["squad_size"] = _parse_int(value)
        elif label == "average age":
            try:
                stats["avg_age"] = float(value)
            except ValueError:
                stats["avg_age"] = None
        elif label == "fifa world ranking":
            stats["fifa_rank"] = _parse_int(value)

    market_node = soup.select_one("a.data-header__market-value-wrapper")
    if market_node:
        stats["market_value_eur"] = _parse_market_value(market_node.get_text(" ", strip=True))
    return stats


def get_squad_quality_with_lanusstats(team_list=None, top_n=48):
    """
    Obtiene calidad de plantilla de selecciones usando LanusStats (Transfermarkt).

    Retorna un DataFrame con:
    - team_name: nombre de la seleccion (segun FIFA)
    - transfermarkt_team_name: nombre de la seleccion en Transfermarkt
    - transfermarkt_team_id: ID de equipo en Transfermarkt
    - squad_size: tamano de la plantilla
    - avg_age: edad promedio
    - market_value_eur: valor de mercado total (euros)
    - fifa_rank: ranking FIFA (si esta disponible en Transfermarkt)
    """
    print("\n📊 Obteniendo calidad de plantilla con LanusStats...")

    try:
        import LanusStats as ls
    except ImportError:
        print("   ❌ LanusStats no instalado.")
        print("   📌 Ejecuta: pip install LanusStats")
        return None

    if team_list is None:
        team_list = _load_teams_from_rankings(top_n=top_n)
        if not team_list:
            team_list = [
                "Argentina",
                "Brazil",
                "France",
                "Spain",
                "England",
                "Germany",
            ]

    results = []
    transfermarkt = ls.Transfermarkt()

    for team_name in team_list:
        query = TEAM_NAME_ALIASES.get(team_name, team_name)
        print(f"   Procesando: {team_name} -> {query}")

        team_data = {
            "team_name": team_name,
            "search_name": query,
            "transfermarkt_team_name": None,
            "transfermarkt_team_id": None,
            "transfermarkt_team_href": None,
            "squad_size": None,
            "avg_age": None,
            "market_value_eur": None,
            "fifa_rank": None,
        }

        try:
            search_url = (
                "https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query="
                + quote(query)
            )
            search_soup = transfermarkt.transfermarkt_request_to_soup(search_url)
            club_box = _find_club_box(search_soup)
            rows = club_box.select("table.items tr") if club_box else []
            row = _find_team_row(rows, query)
            row_data = _extract_team_row_data(row)
            if row_data:
                team_data.update(row_data)

            team_href = team_data.get("transfermarkt_team_href")
            if team_href:
                team_url = f"https://www.transfermarkt.com{team_href}"
                team_soup = transfermarkt.transfermarkt_request_to_soup(team_url)
                header_stats = _extract_team_header_stats(team_soup)
                team_data.update({k: v for k, v in header_stats.items() if v is not None})

        except Exception as e:
            print(f"      Error: {e}")

        results.append(team_data)

    df = pd.DataFrame(results)
    output_path = os.path.join(DATA_RAW, "03_squad_quality.csv")
    df.to_csv(output_path, index=False)
    print(f"   ✅ Guardado: {output_path}")

    missing_rates = df.isna().mean().sort_values(ascending=False)
    print("   📉 Valores faltantes (%):")
    for col, rate in missing_rates.items():
        print(f"      - {col}: {rate:.1%}")

    return df


# ============================================
# FUNCIÓN PRINCIPAL
# ============================================

def main():
    """
    Ejecuta la adquisición de todas las fuentes de datos.
    """
    print("=" * 60)
    print("🏆 ADQUISICIÓN DE DATOS - ORÁCULO DEL BALÓN 2026")
    print("=" * 60)
    
    # Fuente 1: Partidos históricos (automático)
    print("\n1️⃣ FUENTE 1: Partidos históricos")
    print("-" * 40)
    df_matches = download_historical_matches()
    
    # Fuente 2: Ranking FIFA (requiere descarga manual)
    print("\n2️⃣ FUENTE 2: Ranking FIFA")
    print("-" * 40)
    df_rankings = check_fifa_rankings()
    
    # Fuente 3: Calidad de plantilla (LanusStats)
    print("\n3️⃣ FUENTE 3: Calidad de plantilla")
    print("-" * 40)
    df_squad = get_squad_quality_with_lanusstats()
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📋 RESUMEN DE DATOS ADQUIRIDOS")
    print("=" * 60)
    print(f"✅ Partidos históricos: {'OK' if df_matches is not None else 'FALLÓ'}")
    print(f"✅ Ranking FIFA: {'OK' if df_rankings is not None else 'PENDIENTE (descarga manual)'}")
    print(f"✅ Calidad plantilla: {'OK' if df_squad is not None else 'FALLÓ'}")
    
    print("\n💡 Siguiente paso:")
    print("   Ejecuta: python src/cleaning.py")


# ============================================
# EJECUCIÓN DIRECTA
# ============================================

if __name__ == "__main__":
    main()