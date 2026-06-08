"""
Official FIFA World Cup 2026 groups with English team names
matching the dataset conventions.
"""

from typing import Dict, List

# Mapeo de nombres oficiales (español/FIFA) a nombres en el dataset
TEAM_NAME_MAP: Dict[str, str] = {
    "México": "Mexico",
    "Sudáfrica": "South Africa",
    "República de Corea": "South Korea",
    "República Checa": "Czech Republic",
    "Bosnia y Herzegovina": "Bosnia and Herzegovina",
    "Catar": "Qatar",
    "Suiza": "Switzerland",
    "Marruecos": "Morocco",
    "Haití": "Haiti",
    "Escocia": "Scotland",
    "Estados Unidos": "United States",
    "Turquía": "Turkey",
    "Alemania": "Germany",
    "Curazao": "Cura\u00e7ao",
    "Costa de Marfil": "Ivory Coast",
    "Países Bajos": "Netherlands",
    "Japón": "Japan",
    "Suecia": "Sweden",
    "Túnez": "Tunisia",
    "Bélgica": "Belgium",
    "Egipto": "Egypt",
    "RI de Irán": "Iran",
    "Nueva Zelanda": "New Zealand",
    "España": "Spain",
    "Cabo Verde": "Cape Verde",
    "Arabia Saudí": "Saudi Arabia",
    "Francia": "France",
    "Irak": "Iraq",
    "Noruega": "Norway",
    "Argelia": "Algeria",
    "Jordania": "Jordan",
    "RD Congo": "DR Congo",
    "Uzbekistán": "Uzbekistan",
    "Inglaterra": "England",
    "Croacia": "Croatia",
    "Panamá": "Panama",
    # Estos ya coinciden en ambos idiomas
    "Brasil": "Brazil",
    "Argentina": "Argentina",
    "Portugal": "Portugal",
    "Colombia": "Colombia",
    "Ecuador": "Ecuador",
    "Paraguay": "Paraguay",
    "Australia": "Australia",
    "Uruguay": "Uruguay",
    "Senegal": "Senegal",
    "Austria": "Austria",
    "Ghana": "Ghana",
    "Canada": "Canada",
    "Canadá": "Canada",
}


def _translate(name: str) -> str:
    """Translate a FIFA/Spanish team name to the dataset English name."""
    return TEAM_NAME_MAP.get(name, name)


# Grupos oficiales del Mundial 2026 (nombres del dataset)
GROUPS_2026: Dict[str, List[str]] = {
    "A": [_translate(t) for t in ["México", "Sudáfrica", "República de Corea", "República Checa"]],
    "B": [_translate(t) for t in ["Canadá", "Bosnia and Herzegovina", "Catar", "Suiza"]],
    "C": [_translate(t) for t in ["Brasil", "Marruecos", "Haití", "Escocia"]],
    "D": [_translate(t) for t in ["Estados Unidos", "Paraguay", "Australia", "Turquía"]],
    "E": [_translate(t) for t in ["Alemania", "Curazao", "Costa de Marfil", "Ecuador"]],
    "F": [_translate(t) for t in ["Países Bajos", "Japón", "Suecia", "Túnez"]],
    "G": [_translate(t) for t in ["Bélgica", "Egipto", "RI de Irán", "Nueva Zelanda"]],
    "H": [_translate(t) for t in ["España", "Cabo Verde", "Arabia Saudí", "Uruguay"]],
    "I": [_translate(t) for t in ["Francia", "Senegal", "Irak", "Noruega"]],
    "J": [_translate(t) for t in ["Argentina", "Argelia", "Austria", "Jordania"]],
    "K": [_translate(t) for t in ["Portugal", "RD Congo", "Uzbekistán", "Colombia"]],
    "L": [_translate(t) for t in ["Inglaterra", "Croacia", "Ghana", "Panamá"]],
}

# Lista plana de las 48 selecciones (nombres del dataset)
TEAMS_2026: List[str] = [team for group in GROUPS_2026.values() for team in group]
