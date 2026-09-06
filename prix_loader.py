"""
Chargement et nettoyage des prix immobiliers par quartier parisien.
Sources : DVF (Etalab) + fallback table statique de médianes.
"""

import requests
import numpy as np
import pandas as pd
import streamlit as st

TIMEOUT = 45
DVF_BASE = "https://api.cquest.org/dvf"

# Médianes €/m² appartements ancien, par arrondissement (ordre de grandeur 2024-2025)
# Sert de fallback ET de garde-fou pour détecter les aberrations DVF
MEDIANES_ARR = {
    1: 12800, 2: 11600, 3: 12200, 4: 13100, 5: 12400, 6: 15100, 7: 14300,
    8: 12900, 9: 10900, 10: 9800, 11: 10300, 12: 9600, 13: 9200, 14: 10100,
    15: 10400, 16: 11700, 17: 10600, 18: 9300, 19: 8400, 20: 8800,
}

# Modulateurs intra-arrondissement (quartiers premium vs. populaires)
# Coefficient appliqué à la médiane d'arrondissement
COEF_QUARTIER = {
    "sortsgermaindespres": 1.18, "odeon": 1.12, "monnaie": 1.10,
    "notredamedeschamps": 1.05, "invalides": 1.10, "ecolemilitaire": 1.06,
    "gramont": 1.04, "champselysees": 1.15, "madeleine": 1.08,
    "goutted or": 0.82, "lachapelle": 0.84, "amerique": 0.88,
    "porrtedauphine": 1.12, "auteuil": 1.08, "muette": 1.14,
    "belleville": 0.90, "pereLachaise": 0.95,
}


@st.cache_data(ttl=60 * 60 * 24 * 7, show_spinner="Chargement des prix DVF…")
def load_prix_dvf(annees=(2023, 2024)) -> pd.DataFrame:
    """
    Récupère les mutations DVF pour Paris, nettoie et agrège par quartier.
    Retourne : quartier_key, prix_m2_q1, prix_m2_median, prix_m2_q3, n_ventes
    """
    frames = []
    for code_insee in [f"751{str(i).zfill(2)}" for i in range(1, 21)]:
        for an in annees:
            try:
                r = requests.get(
                    f"{DVF_BASE}/",
                    params={"code_commune": code_insee, "annee": an},
                    timeout=TIMEOUT,
                )
                if r.status_code != 200:
                    continue
                data = r.json().get("resultats", [])
                if data:
                    frames.append(pd.DataFrame(data))
            except Exception:
                continue

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    return _clean_dvf(df)


def _clean_dvf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage DVF — étape critique. Les données brutes contiennent
    beaucoup de bruit qui fausse totalement les médianes.
    """
    # Colonnes attendues (noms variables selon millésime API)
    ren = {
        "valeur_fonciere": "prix",
        "surface_relle_bati": "surface",
        "surface_reelle_bati": "surface",
        "type_local": "type_local",
        "code_postal": "cp",
        "nombre_pieces_principales": "pieces",
        "id_mutation": "id_mutation",
    }
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})

    for c in ["prix", "surface", "pieces"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 1. Appartements uniquement
    if "type_local" in df:
        df = df[df["type_local"].astype(str).str.contains("Appartement", na=False)]

    # 2. Exclusion des ventes multi-lots (une mutation = plusieurs lignes)
    if "id_mutation" in df:
        counts = df.groupby("id_mutation").size()
        df = df[df["id_mutation"].isin(counts[counts == 1].index)]

    # 3. Bornes de plausibilité
    df = df.dropna(subset=["prix", "surface"])
    df = df[(df["surface"] >= 9) & (df["surface"] <= 300)]
    df = df[(df["prix"] >= 50_000) & (df["prix"] <= 20_000_000)]

    df["prix_m2"] = df["prix"] / df["surface"]

    # 4. Écrêtage des outliers de prix/m²
    df = df[(df["prix_m2"] >= 3000) & (df["prix_m2"] <= 40_000)]

    # 5. Arrondissement
    if "cp" in df:
        df["arrondissement"] = pd.to_numeric(
            df["cp"].astype(str).str[-2:], errors="coerce"
        )
        df = df[df["arrondissement"].between(1, 20)]

    if df.empty:
        return pd.DataFrame()

    # 6. Agrégation robuste par arrondissement
    agg = (
        df.groupby("arrondissement")["prix_m2"]
        .agg(
            prix_m2_q1=lambda s: s.quantile(0.25),
            prix_m2_median="median",
            prix_m2_q3=lambda s: s.quantile(0.75),
            n_ventes="size",
        )
        .reset_index()
    )
    # Seuil de significativité statistique
    agg["fiable"] = agg["n_ventes"] >= 30
    return agg


def prix_par_quartier(df_loyers: pd.DataFrame,
                      dvf: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Construit une table prix par quartier :
    - base = médiane DVF de l'arrondissement (ou table statique)
    - modulée par un coefficient de quartier
    Retourne une fourchette basse / centrale / haute.
    """
    import unicodedata, re

    def norm(s):
        s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
        return re.sub(r"[^a-z0-9]", "", s.lower())

    quartiers = (
        df_loyers[["quartier", "arrondissement", "id_zone"]]
        .drop_duplicates(subset=["quartier"])
        .copy()
    )

    if dvf is not None and not dvf.empty:
        base = dvf.set_index("arrondissement")["prix_m2_median"].to_dict()
        q1 = dvf.set_index("arrondissement")["prix_m2_q1"].to_dict()
        q3 = dvf.set_index("arrondissement")["prix_m2_q3"].to_dict()
        nv = dvf.set_index("arrondissement")["n_ventes"].to_dict()
        source = "DVF"
    else:
        base, q1, q3, nv = MEDIANES_ARR, {}, {}, {}
        source = "Table statique"

    rows = []
    for _, r in quartiers.iterrows():
        arr = r["arrondissement"]
        if pd.isna(arr):
            continue
        arr = int(arr)
        med = base.get(arr, MEDIANES_ARR.get(arr, 10000))
        coef = COEF_QUARTIER.get(norm(r["quartier"]), 1.0)
        lo = q1.get(arr, med * 0.82) * coef
        hi = q3.get(arr, med * 1.22) * coef
        rows.append(dict(
            quartier=r["quartier"],
            arrondissement=arr,
            id_zone=r["id_zone"],
            prix_m2_bas=round(lo),
            prix_m2_median=round(med * coef),
            prix_m2_haut=round(hi),
            n_ventes=nv.get(arr, np.nan),
            source_prix=source,
        ))
    return pd.DataFrame(rows)
