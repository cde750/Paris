"""
prix_loader.py — Chargement des prix immobiliers parisiens.

PRINCIPE : aucune valeur de prix n'est codée en dur dans ce module.
Les prix proviennent soit des fichiers DVF officiels (Etalab), soit d'une
saisie manuelle explicite de l'utilisateur. En cas d'échec du chargement,
le module lève une exception — il ne produit jamais de données de substitution.

Source DVF : https://files.data.gouv.fr/geo-dvf/latest/csv/
Documentation : https://doc.data.gouv.fr/api/dvf/
"""

from __future__ import annotations

import io
import re
import unicodedata
import zipfile
from dataclasses import dataclass, field

import pandas as pd
import requests
import streamlit as st

# ─────────────────────────── Configuration ───────────────────────────

DVF_URL = "https://files.data.gouv.fr/geo-dvf/latest/csv/{annee}/departements/75.csv.gz"
TIMEOUT = 120
CODES_INSEE_PARIS = [f"751{i:02d}" for i in range(1, 21)]

# Seuils de significativité statistique
N_MIN_QUARTIER = 25      # en dessous, on remonte à l'arrondissement
N_MIN_ARRDT = 40         # en dessous, la ligne est marquée non fiable

# Bornes de plausibilité (filtrage du bruit DVF, pas des estimations de prix)
SURFACE_MIN, SURFACE_MAX = 9, 400
PRIX_MIN, PRIX_MAX = 50_000, 30_000_000
PRIX_M2_MIN, PRIX_M2_MAX = 2_000, 60_000


class PrixDataError(RuntimeError):
    """Impossible de constituer une base de prix exploitable."""


@dataclass
class ResultatPrix:
    """Résultat de chargement, avec métadonnées de qualité."""
    data: pd.DataFrame
    source: str
    millesimes: list[int] = field(default_factory=list)
    n_mutations_brutes: int = 0
    n_mutations_retenues: int = 0
    avertissements: list[str] = field(default_factory=list)

    @property
    def taux_retention(self) -> float:
        if not self.n_mutations_brutes:
            return 0.0
        return self.n_mutations_retenues / self.n_mutations_brutes


# ─────────────────────────── Utilitaires ───────────────────────────

def norm(s: str) -> str:
    """Normalisation pour jointure : sans accents, minuscules, alphanumérique."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s.lower())


# ─────────────────────── Téléchargement DVF ───────────────────────

@st.cache_data(ttl=60 * 60 * 24 * 30, show_spinner=False)
def _download_dvf_annee(annee: int) -> pd.DataFrame:
    """
    Télécharge le fichier DVF départemental (75) pour une année.
    Lève une exception en cas d'échec — pas de retour vide silencieux.
    """
    url = DVF_URL.format(annee=annee)
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
    except requests.RequestException as e:
        raise PrixDataError(f"DVF {annee} : échec du téléchargement ({e})") from e

    usecols = [
        "id_mutation", "date_mutation", "nature_mutation", "valeur_fonciere",
        "code_postal", "code_commune", "nom_commune",
        "code_type_local", "type_local", "surface_reelle_bati",
        "nombre_pieces_principales", "surface_terrain",
        "longitude", "latitude",
    ]
    try:
        df = pd.read_csv(
            io.BytesIO(r.content),
            compression="gzip",
            usecols=lambda c: c in usecols,
            dtype={"code_postal": "string", "code_commune": "string",
                   "id_mutation": "string"},
            low_memory=False,
        )
    except Exception as e:
        raise PrixDataError(f"DVF {annee} : fichier illisible ({e})") from e

    manquantes = {"id_mutation", "valeur_fonciere", "surface_reelle_bati",
                  "type_local", "code_commune"} - set(df.columns)
    if manquantes:
        raise PrixDataError(
            f"DVF {annee} : colonnes absentes du fichier {sorted(manquantes)}. "
            "Le schéma Etalab a peut-être changé."
        )

    df["_millesime"] = annee
    return df


# ─────────────────────────── Nettoyage ───────────────────────────

def _nettoyer_mutations(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Reconstruit les mutations au niveau transaction (et non ligne de lot).

    Logique retenue :
      1. On garde les ventes pures (exclusion échanges, expropriations, VEFA…)
      2. On regroupe par id_mutation
      3. Une mutation est retenue si elle contient exactement UN appartement
         (code_type_local == 2), quels que soient les annexes (cave, parking)
      4. La surface est celle de l'appartement seul ; le prix est celui de la
         mutation entière — c'est cohérent avec la pratique de marché
      5. On exclut les mutations comportant plusieurs locaux d'habitation
         (appartements multiples ou maison + appartement) : prix non ventilable

    Retourne (mutations propres, liste d'avertissements).
    """
    warns: list[str] = []
    n0 = df["id_mutation"].nunique()

    # 1. Ventes uniquement
    if "nature_mutation" in df.columns:
        df = df[df["nature_mutation"].astype(str).str.strip() == "Vente"]

    # Paris uniquement (le fichier 75 ne devrait contenir que Paris, on vérifie)
    df = df[df["code_commune"].isin(CODES_INSEE_PARIS)]
    if df.empty:
        raise PrixDataError("Aucune mutation parisienne dans les fichiers DVF.")

    for c in ["valeur_fonciere", "surface_reelle_bati",
              "nombre_pieces_principales", "code_type_local"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 2-3-5. Composition de chaque mutation
    est_appt = df["code_type_local"] == 2
    est_maison = df["code_type_local"] == 1

    compo = df.groupby("id_mutation").agg(
        n_appt=("code_type_local", lambda s: (s == 2).sum()),
        n_maison=("code_type_local", lambda s: (s == 1).sum()),
        prix=("valeur_fonciere", "max"),      # constant sur la mutation
        cp=("code_postal", "first"),
        millesime=("_millesime", "first"),
        date=("date_mutation", "first"),
    )

    surf_appt = (df[est_appt].groupby("id_mutation")["surface_reelle_bati"]
                 .sum().rename("surface"))
    pieces = (df[est_appt].groupby("id_mutation")["nombre_pieces_principales"]
              .max().rename("pieces"))

    mut = compo.join(surf_appt).join(pieces)

    n_multi = int(((mut["n_appt"] > 1) | (mut["n_maison"] > 0)).sum())
    mut = mut[(mut["n_appt"] == 1) & (mut["n_maison"] == 0)]
    if n_multi:
        warns.append(
            f"{n_multi:,} mutations multi-lots d'habitation exclues "
            "(prix non ventilable par lot)."
        )

    # 4. Bornes de plausibilité
    mut = mut.dropna(subset=["prix", "surface"])
    mut = mut[mut["surface"].between(SURFACE_MIN, SURFACE_MAX)]
    mut = mut[mut["prix"].between(PRIX_MIN, PRIX_MAX)]

    mut["prix_m2"] = mut["prix"] / mut["surface"]
    n_avant_ecretage = len(mut)
    mut = mut[mut["prix_m2"].between(PRIX_M2_MIN, PRIX_M2_MAX)]
    n_ecrete = n_avant_ecretage - len(mut)
    if n_ecrete:
        warns.append(
            f"{n_ecrete:,} mutations écartées pour prix/m² hors bornes "
            f"[{PRIX_M2_MIN:,} – {PRIX_M2_MAX:,} €]."
        )

    # Arrondissement depuis le code commune (plus fiable que le code postal)
    cc = df.groupby("id_mutation")["code_commune"].first()
    mut = mut.join(cc)
    mut["arrondissement"] = mut["code_commune"].str[-2:].astype(int)

    if mut.empty:
        raise PrixDataError(
            "Aucune mutation exploitable après nettoyage. "
            "Vérifiez les millésimes demandés."
        )

    taux = len(mut) / n0 if n0 else 0
    if taux < 0.15:
        warns.append(
            f"Taux de rétention faible ({taux:.0%}) : le nettoyage écarte "
            "l'essentiel des mutations. À investiguer avant usage."
        )

    return mut.reset_index(), warns


# ─────────────── Rattachement géographique aux quartiers ───────────────

def _rattacher_quartiers(mut: pd.DataFrame,
                         gdf_quartiers) -> tuple[pd.DataFrame, list[str]]:
    """
    Jointure spatiale des mutations sur les 80 quartiers administratifs.
    Nécessite geopandas et un GeoDataFrame de quartiers (colonnes :
    'quartier', 'geometry').

    Sans géométrie disponible, retourne les mutations sans colonne quartier —
    l'agrégation se fera au niveau arrondissement, et ce sera signalé.
    """
    warns: list[str] = []
    if gdf_quartiers is None or "longitude" not in mut.columns:
        warns.append(
            "Pas de rattachement par quartier : agrégation à l'arrondissement. "
            "Fournissez le GeoJSON des quartiers pour affiner."
        )
        mut["quartier"] = pd.NA
        return mut, warns

    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError:
        warns.append("geopandas absent : agrégation à l'arrondissement.")
        mut["quartier"] = pd.NA
        return mut, warns

    geo = mut.dropna(subset=["longitude", "latitude"]).copy()
    n_sans_coord = len(mut) - len(geo)
    if n_sans_coord:
        warns.append(f"{n_sans_coord:,} mutations sans géolocalisation.")

    pts = gpd.GeoDataFrame(
        geo,
        geometry=[Point(xy) for xy in zip(geo["longitude"], geo["latitude"])],
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(
        pts, gdf_quartiers[["quartier", "geometry"]].to_crs("EPSG:4326"),
        how="left", predicate="within",
    )
    mut = mut.merge(
        joined[["id_mutation", "quartier"]].drop_duplicates("id_mutation"),
        on="id_mutation", how="left",
    )
    n_orphelines = int(mut["quartier"].isna().sum())
    if n_orphelines:
        warns.append(
            f"{n_orphelines:,} mutations hors polygones de quartier "
            "(repli sur l'arrondissement pour celles-ci)."
        )
    return mut, warns


# ─────────────────────────── Agrégation ───────────────────────────

def _agreger(mut: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège en deux niveaux et retient le plus fin qui soit significatif.
    Aucune extrapolation : un quartier sans données prend la valeur de son
    arrondissement, et la colonne 'niveau_agregation' le dit explicitement.
    """
    def stats(g):
        return pd.Series({
            "prix_m2_p25": g.quantile(0.25),
            "prix_m2_median": g.median(),
            "prix_m2_p75": g.quantile(0.75),
            "n_ventes": g.size,
        })

    par_arr = (mut.groupby("arrondissement")["prix_m2"]
               .apply(stats).unstack().reset_index())
    par_arr["fiable"] = par_arr["n_ventes"] >= N_MIN_ARRDT

    if mut["quartier"].notna().any():
        par_q = (mut.dropna(subset=["quartier"])
                 .groupby(["arrondissement", "quartier"])["prix_m2"]
                 .apply(stats).unstack().reset_index())
        par_q = par_q[par_q["n_ventes"] >= N_MIN_QUARTIER]
    else:
        par_q = pd.DataFrame(columns=["arrondissement", "quartier",
                                      "prix_m2_p25", "prix_m2_median",
                                      "prix_m2_p75", "n_ventes"])
    return par_arr, par_q


def prix_par_quartier(df_loyers: pd.DataFrame,
                      resultat: ResultatPrix) -> pd.DataFrame:
    """
    Construit la table de prix alignée sur les quartiers de df_loyers.

    Chaque ligne porte :
      - prix_m2_bas / median / haut  (P25 / médiane / P75 observés)
      - niveau_agregation : 'quartier' ou 'arrondissement'
      - n_ventes, fiable, source_prix

    Ne produit AUCUNE valeur pour un quartier dont l'arrondissement n'a pas
    de données : la ligne est présente mais les prix sont NaN.
    """
    if resultat.data.empty:
        raise PrixDataError("Base de prix vide : rien à joindre.")

    par_arr, par_q = _agreger(resultat.data)

    quartiers = (df_loyers[["quartier", "arrondissement", "id_zone"]]
                 .drop_duplicates(subset=["quartier"]).copy())
    quartiers["arrondissement"] = pd.to_numeric(
        quartiers["arrondissement"], errors="coerce")
    quartiers = quartiers.dropna(subset=["arrondissement"])
    quartiers["arrondissement"] = quartiers["arrondissement"].astype(int)

    # Clé de jointure normalisée pour absorber les écarts de libellé
    quartiers["_k"] = quartiers["quartier"].map(norm)
    if not par_q.empty:
        par_q = par_q.copy()
        par_q["_k"] = par_q["quartier"].map(norm)

    rows = []
    idx_arr = par_arr.set_index("arrondissement")
    idx_q = par_q.set_index("_k") if not par_q.empty else None

    for _, r in quartiers.iterrows():
        arr = r["arrondissement"]
        src, niveau = resultat.source, None
        p25 = med = p75 = n = pd.NA
        fiable = False

        if idx_q is not None and r["_k"] in idx_q.index:
            s = idx_q.loc[r["_k"]]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[0]
            p25, med, p75, n = (s["prix_m2_p25"], s["prix_m2_median"],
                                s["prix_m2_p75"], s["n_ventes"])
            niveau, fiable = "quartier", True
        elif arr in idx_arr.index:
            s = idx_arr.loc[arr]
            p25, med, p75, n = (s["prix_m2_p25"], s["prix_m2_median"],
                                s["prix_m2_p75"], s["n_ventes"])
            niveau, fiable = "arrondissement", bool(s["fiable"])

        rows.append(dict(
            quartier=r["quartier"],
            arrondissement=arr,
            id_zone=r["id_zone"],
            prix_m2_bas=round(p25) if pd.notna(p25) else pd.NA,
            prix_m2_median=round(med) if pd.notna(med) else pd.NA,
            prix_m2_haut=round(p75) if pd.notna(p75) else pd.NA,
            n_ventes=int(n) if pd.notna(n) else pd.NA,
            niveau_agregation=niveau or "aucune donnée",
            fiable=fiable,
            source_prix=src,
            millesimes=", ".join(map(str, resultat.millesimes)),
        ))

    out = pd.DataFrame(rows)
    n_vides = int(out["prix_m2_median"].isna().sum())
    if n_vides:
        st.warning(
            f"{n_vides} quartier(s) sans donnée de prix. "
            "Les rendements ne seront pas calculés pour ces zones."
        )
    return out


# ─────────────────────── Point d'entrée principal ───────────────────────

def charger_prix(annees: tuple[int, ...] = (2023, 2024),
                 gdf_quartiers=None) -> ResultatPrix:
    """
    Charge, nettoie et agrège les prix DVF.
    Lève PrixDataError si aucune année n'est exploitable.
    """
    frames, echecs, ok = [], [], []
    for an in annees:
        try:
            frames.append(_download_dvf_annee(an))
            ok.append(an)
        except PrixDataError as e:
            echecs.append(str(e))

    if not frames:
        raise PrixDataError(
            "Aucun millésime DVF n'a pu être chargé.\n" + "\n".join(echecs)
        )

    brut = pd.concat(frames, ignore_index=True)
    n_brutes = brut["id_mutation"].nunique()

    mut, w1 = _nettoyer_mutations(brut)
    mut, w2 = _rattacher_quartiers(mut, gdf_quartiers)

    return ResultatPrix(
        data=mut,
        source="DVF Etalab (files.data.gouv.fr/geo-dvf)",
        millesimes=ok,
        n_mutations_brutes=n_brutes,
        n_mutations_retenues=len(mut),
        avertissements=echecs + w1 + w2,
    )


# ─────────────────── Mode saisie manuelle (secours) ───────────────────

def table_saisie_vide(df_loyers: pd.DataFrame) -> pd.DataFrame:
    """
    Squelette à remplir par l'utilisateur quand DVF est indisponible.
    Volontairement vide : aucune valeur suggérée.
    """
    q = (df_loyers[["quartier", "arrondissement", "id_zone"]]
         .drop_duplicates(subset=["quartier"]).copy())
    q["prix_m2_bas"] = pd.NA
    q["prix_m2_median"] = pd.NA
    q["prix_m2_haut"] = pd.NA
    q["n_ventes"] = pd.NA
    q["niveau_agregation"] = "saisie manuelle"
    q["fiable"] = False
    q["source_prix"] = "Saisie utilisateur"
    q["millesimes"] = ""
    return q


def bloc_qualite(resultat: ResultatPrix) -> None:
    """Affiche un rapport de qualité — à appeler dans l'onglet investisseur."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Source", "DVF officiel")
    c2.metric("Millésimes", ", ".join(map(str, resultat.millesimes)))
    c3.metric("Mutations retenues", f"{resultat.n_mutations_retenues:,}")
    c4.metric("Taux de rétention", f"{resultat.taux_retention:.0%}")

    if resultat.avertissements:
        with st.expander(f"⚠️ {len(resultat.avertissements)} avertissement(s) "
                         "sur la qualité des données"):
            for a in resultat.avertissements:
                st.markdown(f"- {a}")
