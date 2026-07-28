"""
Chargement des données d'encadrement des loyers de Paris
Source : Open Data Paris (opendata.paris.fr) - jeu "logement-encadrement-des-loyers"
"""

import io
import json
import requests
import pandas as pd
import streamlit as st

# --- Endpoints Opendatasoft (Ville de Paris) ---
BASE = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets"
DS_LOYERS = "logement-encadrement-des-loyers"
DS_QUARTIERS = "quartier_paris"

TIMEOUT = 30


# ----------------------------------------------------------------------
# 1. Les loyers de référence
# ----------------------------------------------------------------------
@st.cache_data(ttl=60 * 60 * 24, show_spinner="Chargement des loyers de référence…")
def load_loyers() -> pd.DataFrame:
    """
    Récupère l'intégralité du référentiel des loyers (~ 5 000 lignes / an).
    Colonnes clés :
        annee, id_zone, nom_quartier, piece, epoque, meuble_txt,
        ref, max, min, geo_shape, geo_point_2d
    """
    url = f"{BASE}/{DS_LOYERS}/exports/json"
    params = {"limit": -1}
    try:
        r = requests.get(url, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
    except Exception as e:
        st.warning(f"API indisponible ({e}). Utilisation du jeu de démonstration.")
        return _fallback_loyers()

    return _normalize(df)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonise les noms de colonnes selon les millésimes du jeu de données."""
    rename = {
        "annee": "annee",
        "id_zone": "id_zone",
        "id_quartier": "id_quartier",
        "nom_quartier": "quartier",
        "piece": "pieces",
        "epoque": "epoque",
        "meuble_txt": "meuble",
        "ref": "loyer_ref",
        "max": "loyer_max",
        "min": "loyer_min",
        "geo_shape": "geo_shape",
        "geo_point_2d": "geo_point",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Typage
    for c in ["loyer_ref", "loyer_max", "loyer_min"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "annee" in df:
        df["annee"] = pd.to_numeric(df["annee"], errors="coerce").astype("Int64")
    if "pieces" in df:
        df["pieces"] = pd.to_numeric(df["pieces"], errors="coerce").astype("Int64")

    # Normalisation meublé
    if "meuble" in df:
        df["meuble"] = (
            df["meuble"]
            .astype(str)
            .str.lower()
            .map(lambda s: "meublé" if "non" not in s and "meubl" in s else "non meublé")
        )

    # Arrondissement déduit de l'id_zone / quartier
    if "id_quartier" in df:
        df["arrondissement"] = (
            pd.to_numeric(df["id_quartier"], errors="coerce") // 4 + 1
        ).astype("Int64")

    return df


# ----------------------------------------------------------------------
# 2. La géométrie des quartiers (GeoJSON)
# ----------------------------------------------------------------------
@st.cache_data(ttl=60 * 60 * 24, show_spinner="Chargement du fond de carte…")
def load_geojson_quartiers() -> dict:
    """
    Récupère le GeoJSON des 80 quartiers administratifs de Paris.
    """
    url = f"{BASE}/{DS_QUARTIERS}/exports/geojson"
    try:
        r = requests.get(url, params={"limit": -1}, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Impossible de charger le fond de carte : {e}")
        return {"type": "FeatureCollection", "features": []}


@st.cache_data(ttl=60 * 60 * 24)
def load_geojson_from_loyers(df: pd.DataFrame) -> dict:
    """
    Alternative : reconstruit un GeoJSON directement depuis la colonne
    geo_shape du jeu de loyers (utile si le dataset quartier_paris change).
    """
    feats, seen = [], set()
    for _, row in df.iterrows():
        key = row.get("quartier")
        shape = row.get("geo_shape")
        if not key or key in seen or not isinstance(shape, dict):
            continue
        seen.add(key)
        geom = shape.get("geometry", shape)
        feats.append(
            {
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "quartier": key,
                    "id_zone": row.get("id_zone"),
                    "id_quartier": row.get("id_quartier"),
                },
            }
        )
    return {"type": "FeatureCollection", "features": feats}


# ----------------------------------------------------------------------
# 3. Jeu de secours (offline)
# ----------------------------------------------------------------------
def _fallback_loyers() -> pd.DataFrame:
    """Petit échantillon synthétique pour permettre une démo hors-ligne."""
    import numpy as np

    quartiers = [
        ("St-Germain-l'Auxerrois", 1, 1), ("Halles", 1, 1),
        ("Palais-Royal", 1, 1), ("Place-Vendôme", 1, 1),
        ("Gaillon", 2, 1), ("Vivienne", 2, 1), ("Mail", 2, 1), ("Bonne-Nouvelle", 2, 1),
        ("Arts-et-Metiers", 3, 2), ("Enfants-Rouges", 3, 2),
        ("Archives", 3, 2), ("Sainte-Avoie", 3, 2),
        ("St-Merri", 4, 2), ("St-Gervais", 4, 2), ("Arsenal", 4, 2), ("Notre-Dame", 4, 2),
        ("St-Victor", 5, 3), ("Jardin-des-Plantes", 5, 3),
        ("Val-de-Grace", 5, 3), ("Sorbonne", 5, 3),
        ("Monnaie", 6, 3), ("Odeon", 6, 3),
        ("Notre-Dame-des-Champs", 6, 3), ("St-Germain-des-Pres", 6, 3),
        ("St-Thomas-d'Aquin", 7, 4), ("Invalides", 7, 4),
        ("Ecole-Militaire", 7, 4), ("Gros-Caillou", 7, 4),
        ("Champs-Elysees", 8, 4), ("Faubourg-du-Roule", 8, 4),
        ("Madeleine", 8, 4), ("Europe", 8, 4),
    ]
    rows = []
    rng = np.random.default_rng(42)
    for q, arr, zone in quartiers:
        base = 28 - 0.35 * arr + rng.normal(0, 1.2)
        for p in [1, 2, 3, 4]:
            for ep in ["Avant 1946", "1946-1970", "1971-1990", "Apres 1990"]:
                for m in ["meublé", "non meublé"]:
                    ref = base - 1.1 * (p - 1) + (1.8 if m == "meublé" else 0)
                    ref += {"Avant 1946": 0.6, "1946-1970": -1.0,
                            "1971-1990": -0.4, "Apres 1990": 1.4}[ep]
                    rows.append(dict(
                        annee=2025, id_zone=zone, quartier=q, arrondissement=arr,
                        pieces=p, epoque=ep, meuble=m,
                        loyer_ref=round(ref, 1),
                        loyer_max=round(ref * 1.2, 1),
                        loyer_min=round(ref * 0.7, 1),
                    ))
    return pd.DataFrame(rows)
