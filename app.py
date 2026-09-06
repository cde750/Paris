from prix_loader import load_prix_dvf, prix_par_quartier, MEDIANES_ARR
from rentabilite import (HypothesesInvest, analyser, analyser_secteurs,
                         monte_carlo)
from dataclasses import asdict



import json
import numpy as np
import pandas as pd
import streamlit as st
import folium
import branca.colormap as cm
from folium.plugins import Fullscreen, MiniMap
from streamlit_folium import st_folium
import plotly.express as px

from data_loader import (
    load_loyers,
    load_geojson_quartiers,
    load_geojson_from_loyers,
)

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Encadrement des loyers · Paris",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .block-container {padding-top: 2rem; padding-bottom: 1rem;}
  div[data-testid="stMetricValue"] {font-size: 1.6rem;}
  .legal-box {
      background:#f0f7ff; border-left:4px solid #1f77b4;
      padding:.8rem 1rem; border-radius:4px; font-size:.87rem;
  }
</style>
""", unsafe_allow_html=True)

st.title("🏙️ Encadrement des loyers à Paris")
st.caption(
    "Sectorisation par quartier · Loyers de référence majorés · "
    "Source : Open Data Ville de Paris / OLAP-DRIHL"
)

# ══════════════════════════════════════════════════════════════════════
# DONNÉES
# ══════════════════════════════════════════════════════════════════════
df = load_loyers()

if df.empty:
    st.error("Aucune donnée disponible.")
    st.stop()

gj = load_geojson_quartiers()
if not gj["features"]:
    gj = load_geojson_from_loyers(df)


def norm(s: str) -> str:
    """Normalise un nom de quartier pour le matching GeoJSON ↔ DataFrame."""
    import unicodedata, re
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]", "", s.lower())
    return s


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR — PARAMÈTRES DU LOGEMENT
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Paramètres du logement")

    annees = sorted(df["annee"].dropna().unique(), reverse=True)
    annee = st.selectbox("Millésime de l'arrêté", annees, index=0)

    d = df[df["annee"] == annee]

    pieces_opts = sorted(d["pieces"].dropna().unique())
    pieces_lbl = {1: "1 pièce (studio)", 2: "2 pièces",
                  3: "3 pièces", 4: "4 pièces et +"}
    pieces = st.select_slider(
        "Nombre de pièces principales",
        options=pieces_opts,
        value=pieces_opts[0],
        format_func=lambda p: pieces_lbl.get(int(p), f"{p} p."),
    )

    epoques = list(d["epoque"].dropna().unique())
    ordre = ["Avant 1946", "1946-1970", "1971-1990", "Apres 1990", "Après 1990"]
    epoques = sorted(epoques, key=lambda e: ordre.index(e) if e in ordre else 99)
    epoque = st.selectbox("Époque de construction", epoques)

    meuble = st.radio(
        "Type de location",
        ["non meublé", "meublé"],
        horizontal=True,
        format_func=lambda m: "🛋️ Meublé" if m == "meublé" else "📦 Vide",
    )

    st.divider()
    st.header("📐 Votre bien")

    surface = st.number_input(
        "Surface habitable (m²)", min_value=5.0, max_value=400.0,
        value=35.0, step=0.5,
    )

    complement = st.number_input(
        "Complément de loyer (€/mois)",
        min_value=0.0, value=0.0, step=10.0,
        help="Uniquement pour des caractéristiques exceptionnelles "
             "(terrasse, vue, hauteur sous plafond…) non prises en compte "
             "dans le loyer de référence. Doit figurer au bail et être justifiable.",
    )

    charges = st.number_input(
        "Provision pour charges (€/mois)", min_value=0.0, value=0.0, step=10.0
    )

    st.divider()
    st.header("🎨 Affichage carte")

    indicateur = st.radio(
        "Indicateur cartographié",
        ["Loyer majoré (€/m²)", "Loyer de référence (€/m²)",
         "Loyer minoré (€/m²)", "Loyer max. mensuel (€)"],
        index=0,
    )

    palette = st.selectbox(
        "Palette",
        ["YlOrRd", "RdYlGn_r", "viridis", "plasma", "Blues"],
        index=0,
    )

    n_bins = st.slider("Nombre de classes", 3, 12, 8)
    opacite = st.slider("Opacité des polygones", 0.1, 1.0, 0.75, 0.05)
    show_labels = st.checkbox("Afficher les étiquettes de quartier", value=False)

    arr_filter = st.multiselect(
        "Filtrer des arrondissements",
        options=sorted(d["arrondissement"].dropna().unique())
        if "arrondissement" in d else [],
        default=[],
        format_func=lambda a: f"{int(a)}ᵉ" if a != 1 else "1ᵉʳ",
    )

# ══════════════════════════════════════════════════════════════════════
# FILTRAGE
# ══════════════════════════════════════════════════════════════════════
sel = d[
    (d["pieces"] == pieces)
    & (d["epoque"] == epoque)
    & (d["meuble"] == meuble)
].copy()

if arr_filter and "arrondissement" in sel:
    sel = sel[sel["arrondissement"].isin(arr_filter)]

if sel.empty:
    st.warning("Aucune donnée pour cette combinaison de critères.")
    st.stop()

# Agrégation par quartier (sécurité si doublons)
agg = (
    sel.groupby("quartier", as_index=False)
    .agg(
        loyer_ref=("loyer_ref", "mean"),
        loyer_max=("loyer_max", "mean"),
        loyer_min=("loyer_min", "mean"),
        id_zone=("id_zone", "first"),
        arrondissement=("arrondissement", "first")
        if "arrondissement" in sel else ("loyer_ref", "size"),
    )
)

# Colonne cartographiée
col_map = {
    "Loyer majoré (€/m²)": "loyer_max",
    "Loyer de référence (€/m²)": "loyer_ref",
    "Loyer minoré (€/m²)": "loyer_min",
}
if indicateur == "Loyer max. mensuel (€)":
    agg["loyer_mensuel_max"] = agg["loyer_max"] * surface + complement
    col = "loyer_mensuel_max"
    unit = "€/mois"
else:
    col = col_map[indicateur]
    unit = "€/m²"

agg["_key"] = agg["quartier"].map(norm)
lookup = agg.set_index("_key").to_dict("index")

# ══════════════════════════════════════════════════════════════════════
# KPI GLOBAUX
# ══════════════════════════════════════════════════════════════════════
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Quartiers couverts", f"{len(agg)}")
c2.metric("Majoré médian", f"{agg['loyer_max'].median():.1f} €/m²")
c3.metric("Majoré min.", f"{agg['loyer_max'].min():.1f} €/m²")
c4.metric("Majoré max.", f"{agg['loyer_max'].max():.1f} €/m²")
c5.metric(
    "Amplitude",
    f"×{agg['loyer_max'].max() / max(agg['loyer_max'].min(), .01):.2f}",
)

st.divider()

# ══════════════════════════════════════════════════════════════════════
# CARTE
# ══════════════════════════════════════════════════════════════════════
tab_carte, tab_simu, tab_invest, tab_data, tab_analyse = st.tabs(
    ["🗺️ Carte", "🧮 Simulateur", "💰 Investisseur", "📋 Données", "📈 Analyses"]
)

with tab_carte:
    left, right = st.columns([3, 1])

    with left:
        vmin, vmax = float(agg[col].min()), float(agg[col].max())
        if vmin == vmax:
            vmax = vmin + 1

        colormap = cm.linear.__getattr__(
            palette if hasattr(cm.linear, palette) else "YlOrRd"
        ).scale(vmin, vmax) if hasattr(cm.linear, palette) else \
            cm.LinearColormap(["#ffffb2", "#fd8d3c", "#bd0026"], vmin=vmin, vmax=vmax)

        colormap = colormap.to_step(n=n_bins)
        colormap.caption = f"{indicateur}"

        m = folium.Map(
            location=[48.8566, 2.3400],
            zoom_start=12,
            tiles="CartoDB positron",
            control_scale=True,
        )

        # ------- Injection des valeurs dans le GeoJSON -------
        feats = []
        for f in gj["features"]:
            props = f["properties"]
            # Le champ de nom varie selon les millésimes du dataset
            name = (
                props.get("l_qu")
                or props.get("nom_quartier")
                or props.get("quartier")
                or props.get("l_q")
                or ""
            )
            rec = lookup.get(norm(name))
            if rec is None:
                continue
            newp = dict(props)
            newp.update({
                "quartier": rec["quartier"],
                "arrondissement": rec.get("arrondissement"),
                "zone": rec.get("id_zone"),
                "ref": round(rec["loyer_ref"], 2),
                "maj": round(rec["loyer_max"], 2),
                "min": round(rec["loyer_min"], 2),
                "value": round(rec[col], 2),
                "mensuel_max": round(rec["loyer_max"] * surface + complement, 0),
                "mensuel_ref": round(rec["loyer_ref"] * surface, 0),
            })
            feats.append({"type": "Feature",
                          "geometry": f["geometry"],
                          "properties": newp})

        gj_map = {"type": "FeatureCollection", "features": feats}

        def style_fn(feature):
            v = feature["properties"]["value"]
            return {
                "fillColor": colormap(v),
                "color": "#444444",
                "weight": 0.7,
                "fillOpacity": opacite,
            }

        def highlight_fn(feature):
            return {"weight": 3, "color": "#000000", "fillOpacity": 0.9}

        folium.GeoJson(
            gj_map,
            name="Encadrement des loyers",
            style_function=style_fn,
            highlight_function=highlight_fn,
            tooltip=folium.GeoJsonTooltip(
                fields=["quartier", "arrondissement", "zone",
                        "ref", "maj", "min", "mensuel_max"],
                aliases=["Quartier", "Arrond.", "Zone",
                         "Référence (€/m²)", "🔴 Majoré (€/m²)",
                         "Minoré (€/m²)", f"Loyer max. {surface:g} m² (€)"],
                localize=True,
                sticky=True,
                style=("background-color:white; border:1px solid #999;"
                       "border-radius:5px; padding:8px; font-size:12px;"),
            ),
            popup=folium.GeoJsonPopup(
                fields=["quartier", "maj", "mensuel_max"],
                aliases=["Quartier", "Majoré €/m²", "Plafond mensuel €"],
                localize=True,
            ),
        ).add_to(m)

        # Contours d'arrondissements (habillage)
        folium.GeoJson(
            gj,
            name="Contours quartiers",
            style_function=lambda x: {
                "fillOpacity": 0, "color": "#666", "weight": 0.4
            },
            control=True,
        ).add_to(m)

        if show_labels:
            for f in feats:
                p = f["properties"]
                try:
                    from shapely.geometry import shape
                    c = shape(f["geometry"]).representative_point()
                    folium.Marker(
                        [c.y, c.x],
                        icon=folium.DivIcon(html=(
                            f'<div style="font-size:9px;color:#111;'
                            f'text-shadow:0 0 3px #fff;font-weight:600">'
                            f'{p["maj"]:.1f}</div>')),
                    ).add_to(m)
                except Exception:
                    pass

        colormap.add_to(m)
        Fullscreen().add_to(m)
        MiniMap(toggle_display=True, minimized=True).add_to(m)
        folium.LayerControl(collapsed=True).add_to(m)

        out = st_folium(m, width=None, height=650, returned_objects=["last_object_clicked_tooltip"])

    with right:
        st.subheader("🏆 Classement")
        top = agg.nlargest(12, "loyer_max")[["quartier", "loyer_max"]]
        st.dataframe(
            top.rename(columns={"quartier": "Quartier", "loyer_max": "€/m²"}),
            hide_index=True, use_container_width=True,
            column_config={"€/m²": st.column_config.ProgressColumn(
                "€/m² majoré", min_value=float(agg["loyer_max"].min()),
                max_value=float(agg["loyer_max"].max()), format="%.1f")},
        )
        st.subheader("💸 Les plus abordables")
        bot = agg.nsmallest(8, "loyer_max")[["quartier", "loyer_max"]]
        st.dataframe(
            bot.rename(columns={"quartier": "Quartier", "loyer_max": "€/m²"}),
            hide_index=True, use_container_width=True,
        )

# ══════════════════════════════════════════════════════════════════════
# SIMULATEUR
# ══════════════════════════════════════════════════════════════════════
with tab_simu:
    st.subheader("🧮 Vérifier la conformité d'un loyer")

    cA, cB = st.columns([1, 1])

    with cA:
        quartier_sel = st.selectbox(
            "Quartier du logement",
            sorted(agg["quartier"].unique()),
        )
        loyer_pratique = st.number_input(
            "Loyer hors charges pratiqué (€/mois)",
            min_value=0.0, value=1200.0, step=10.0,
        )

    rec = agg[agg["quartier"] == quartier_sel].iloc[0]
    plafond_base = rec["loyer_max"] * surface
    plafond_total = plafond_base + complement
    plancher = rec["loyer_min"] * surface

    with cB:
        st.markdown(f"""
        **Quartier :** {quartier_sel}  
        **Zone :** {rec.get('id_zone', '—')} · **{int(pieces)} p.** · {epoque} · {meuble}
        """)
        st.metric("Loyer de référence", f"{rec['loyer_ref']:.1f} €/m²",
                  f"{rec['loyer_ref']*surface:,.0f} € pour {surface:g} m²".replace(",", " "))
        st.metric("🔴 Loyer de référence MAJORÉ", f"{rec['loyer_max']:.1f} €/m²",
                  f"{plafond_base:,.0f} € plafond".replace(",", " "))

    st.divider()

    k1, k2, k3 = st.columns(3)
    k1.metric("Plancher (minoré)", f"{plancher:,.0f} €".replace(",", " "))
    k2.metric("Plafond légal (majoré + complément)",
              f"{plafond_total:,.0f} €".replace(",", " "))
    ecart = loyer_pratique - plafond_total
    k3.metric("Écart au plafond", f"{ecart:+,.0f} €".replace(",", " "),
              delta=f"{ecart/plafond_total*100:+.1f} %" if plafond_total else None,
              delta_color="inverse")

    # Jauge
    fig = px.bar(
        x=[plancher, rec["loyer_ref"] * surface, plafond_total, loyer_pratique],
        y=["Minoré", "Référence", "Plafond majoré", "Loyer pratiqué"],
        orientation="h",
        color=["Minoré", "Référence", "Plafond majoré", "Loyer pratiqué"],
        color_discrete_map={
            "Minoré": "#a5d6a7", "Référence": "#64b5f6",
            "Plafond majoré": "#ef9a9a",
            "Loyer pratiqué": "#c62828" if ecart > 0 else "#2e7d32",
        },
        text=[f"{v:,.0f} €".replace(",", " ")
              for v in [plancher, rec["loyer_ref"] * surface,
                        plafond_total, loyer_pratique]],
    )
    fig.update_layout(showlegend=False, height=280,
                      margin=dict(l=0, r=0, t=10, b=0),
                      xaxis_title="€ / mois", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    if ecart > 0:
        st.error(f"""
        ### ⚠️ Loyer non conforme
        Le loyer dépasse le plafond légal de **{ecart:,.0f} € / mois**
        (soit **{ecart*12:,.0f} € / an**).

        Le locataire peut saisir la **Commission départementale de conciliation (CDC)**
        de Paris, puis le juge des contentieux de la protection.
        Le bailleur s'expose à une amende administrative
        (jusqu'à 5 000 € personne physique / 15 000 € personne morale).
        """.replace(",", " "))
    else:
        st.success(f"""
        ### ✅ Loyer conforme
        Marge disponible avant le plafond : **{-ecart:,.0f} € / mois**.
        """.replace(",", " "))

    st.markdown("""
    <div class="legal-box">
    <b>Rappels juridiques</b><br>
    • Le loyer de base ne peut excéder le <b>loyer de référence majoré</b> (= référence + 20 %).<br>
    • Un <b>complément de loyer</b> n'est possible que pour des caractéristiques
      de <i>confort ou de localisation exceptionnelles</i>, absentes des logements
      comparables, et doit être <b>explicitement mentionné au bail</b>.<br>
    • Les annonces doivent afficher le loyer de référence majoré et, le cas échéant,
      le complément de loyer (art. 3 loi du 6 juillet 1989).<br>
    • Contestation : 3 ans à compter de la signature du bail.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# DONNÉES
# ══════════════════════════════════════════════════════════════════════
with tab_data:
    st.subheader("📋 Table de référence")

    show = agg[["quartier", "arrondissement", "id_zone",
                "loyer_min", "loyer_ref", "loyer_max"]].copy()
    show["mensuel_max"] = (show["loyer_max"] * surface + complement).round(0)
    show = show.sort_values("loyer_max", ascending=False)

    st.dataframe(
        show.rename(columns={
            "quartier": "Quartier", "arrondissement": "Arr.",
            "id_zone": "Zone", "loyer_min": "Minoré €/m²",
            "loyer_ref": "Référence €/m²", "loyer_max": "Majoré €/m²",
            "mensuel_max": f"Plafond {surface:g} m² (€)",
        }),
        hide_index=True, use_container_width=True, height=520,
    )

    st.download_button(
        "⬇️ Télécharger (CSV)",
        show.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
        file_name=f"encadrement_loyers_paris_{annee}_{pieces}p.csv",
        mime="text/csv",
    )

# ══════════════════════════════════════════════════════════════════════
# ANALYSES
# ══════════════════════════════════════════════════════════════════════
with tab_analyse:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Distribution des loyers majorés")
        fig = px.histogram(agg, x="loyer_max", nbins=25,
                           labels={"loyer_max": "€/m² majoré"},
                           color_discrete_sequence=["#d84315"])
        fig.add_vline(x=agg["loyer_max"].median(), line_dash="dash",
                      annotation_text="médiane")
        fig.update_layout(height=380, bargap=.05)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Loyer majoré par arrondissement")
        if "arrondissement" in agg:
            fig = px.box(agg.dropna(subset=["arrondissement"]),
                         x="arrondissement", y="loyer_max",
                         labels={"arrondissement": "Arrondissement",
                                 "loyer_max": "€/m² majoré"},
                         color_discrete_sequence=["#1565c0"])
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Effet des paramètres sur le loyer majoré médian")
    base = d[d["meuble"] == meuble]
    piv = (base.pivot_table(index="epoque", columns="pieces",
                            values="loyer_max", aggfunc="median")
           .reindex(["Avant 1946", "1946-1970", "1971-1990",
                     "Apres 1990", "Après 1990"]).dropna(how="all"))
    fig = px.imshow(piv, text_auto=".1f", aspect="auto",
                    color_continuous_scale="YlOrRd",
                    labels=dict(x="Nb pièces", y="Époque", color="€/m²"))
    fig.update_layout(height=330)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Évolution pluriannuelle (médiane parisienne)")
    evo = (df[(df["pieces"] == pieces) & (df["meuble"] == meuble) &
              (df["epoque"] == epoque)]
           .groupby("annee")[["loyer_min", "loyer_ref", "loyer_max"]]
           .median().reset_index())
    if len(evo) > 1:
        fig = px.line(evo, x="annee", y=["loyer_min", "loyer_ref", "loyer_max"],
                      markers=True,
                      labels={"value": "€/m²", "annee": "Année",
                              "variable": "Indicateur"})
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Un seul millésime disponible dans le jeu de données chargé.")

# ══════════════════════════════════════════════════════════════════════



with tab_invest:
    st.subheader("💰 Rentabilité locative par secteur")

    st.info("""
    **Lecture des résultats** — Le loyer est *plafonné par arrêté* (donnée exacte),
    mais le prix d'acquisition est une *médiane statistique* avec une dispersion
    intra-quartier de ± 20 à 25 %. Les rendements sont donc des **ordres de
    grandeur** : c'est la *hiérarchie entre secteurs* qui est informative,
    pas la valeur absolue à la décimale.
    """)

    # ─────────────────── HYPOTHÈSES ───────────────────
    with st.expander("⚙️ Hypothèses d'investissement", expanded=True):
        h1, h2, h3, h4 = st.columns(4)

        with h1:
            st.markdown("**Acquisition**")
            frais_notaire = st.slider("Frais de notaire (%)", 2.0, 9.0, 7.5, .1)
            frais_agence = st.slider("Frais d'agence (%)", 0.0, 6.0, 4.0, .5)
            travaux_m2 = st.number_input("Travaux (€/m²)", 0, 3000, 300, 50)
            mobilier_m2 = st.number_input(
                "Mobilier (€/m²)", 0, 800,
                250 if meuble == "meublé" else 0, 25,
                help="Obligatoire en meublé : liste minimale réglementaire",
            )

        with h2:
            st.markdown("**Exploitation**")
            charges_m2 = st.number_input("Charges copro non récup. (€/m²/an)",
                                         0, 100, 30, 5)
            tf_pct = st.slider("Taxe foncière (% du loyer annuel)",
                               0.0, 20.0, 8.0, .5)
            vacance = st.slider("Vacance locative (%)", 0.0, 20.0, 4.0, .5)
            gestion = st.slider("Gestion locative (%)", 0.0, 12.0, 0.0, .5)
            provision = st.slider("Provision gros travaux (%)",
                                  0.0, 15.0, 4.0, .5)

        with h3:
            st.markdown("**Financement**")
            apport = st.slider("Apport (%)", 0, 100, 20, 5)
            taux = st.slider("Taux crédit (%)", 0.5, 7.0, 3.60, .05)
            duree = st.select_slider("Durée (ans)",
                                     [10, 15, 20, 22, 25], value=20)
            assur_emp = st.slider("Assurance emprunteur (%)",
                                  0.0, 1.0, 0.34, .02)

        with h4:
            st.markdown("**Fiscalité**")
            regime = st.selectbox(
                "Régime fiscal",
                ["LMNP amortissement", "Micro-BIC", "Réel foncier",
                 "Micro-foncier"],
                index=0 if meuble == "meublé" else 2,
            )
            tmi = st.select_slider("TMI (%)", [0, 11, 30, 41, 45], value=30)
            part_terrain = st.slider("Part terrain non amort. (%)",
                                     5, 30, 15, 1)
            duree_amort = st.slider("Durée amort. bâti (ans)", 20, 40, 30, 1)

        # Cohérence régime / type de location
        if meuble == "non meublé" and regime in ("LMNP amortissement", "Micro-BIC"):
            st.warning(
                "⚠️ Les régimes LMNP / micro-BIC supposent une **location meublée**. "
                "Basculez le type de location dans la barre latérale, "
                "ou choisissez un régime foncier."
            )
        if meuble == "meublé" and regime in ("Réel foncier", "Micro-foncier"):
            st.warning(
                "⚠️ Les régimes fonciers s'appliquent à la location **nue**. "
                "En meublé, les revenus sont des BIC."
            )

    H = HypothesesInvest(
        frais_notaire_pct=frais_notaire, frais_agence_pct=frais_agence,
        travaux_eur_m2=travaux_m2, mobilier_eur_m2=mobilier_m2,
        charges_copro_eur_m2_an=charges_m2, taxe_fonciere_pct_loyer=tf_pct,
        gestion_locative_pct=gestion, vacance_locative_pct=vacance,
        provision_travaux_pct_loyer=provision,
        apport_pct=apport, taux_credit_pct=taux, duree_credit_ans=duree,
        taux_assurance_emprunteur_pct=assur_emp,
        regime=regime, tmi_pct=tmi, part_terrain_pct=part_terrain,
        duree_amort_bati_ans=duree_amort,
    )

    # ─────────────────── PRIX ───────────────────
    cp1, cp2, cp3 = st.columns([1, 1, 2])
    with cp1:
        src_prix = st.radio("Source des prix",
                            ["DVF (transactions réelles)", "Table statique"],
                            index=1)
    with cp2:
        scenario = st.radio("Scénario de prix",
                            ["bas", "median", "haut"], index=1,
                            format_func=lambda s: {
                                "bas": "🟢 Bas (Q1 — bonne affaire)",
                                "median": "🟡 Médian",
                                "haut": "🔴 Haut (Q3)"}[s])

    dvf = load_prix_dvf() if src_prix.startswith("DVF") else None
    prix_q = prix_par_quartier(agg.assign(
        arrondissement=agg.get("arrondissement")), dvf)

    with cp3:
        if dvf is not None and not dvf.empty:
            n_fiables = int(dvf["fiable"].sum())
            st.metric("Arrondissements avec échantillon suffisant",
                      f"{n_fiables}/20",
                      f"{int(dvf['n_ventes'].sum()):,} ventes retenues".replace(",", " "))
        else:
            st.caption("📌 Prix issus de la table de médianes intégrée.")

    # Jointure loyers × prix
    base_inv = prix_q.merge(
        agg[["quartier", "loyer_ref", "loyer_max", "loyer_min"]],
        on="quartier", how="inner",
    )

    if base_inv.empty:
        st.error("Jointure loyers/prix vide.")
        st.stop()

    res = analyser_secteurs(base_inv, surface, complement, H, scenario)

    # ─────────────────── KPI ───────────────────
    st.divider()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rdt brut médian", f"{res['rdt_brut'].median():.2f} %")
    m2.metric("Rdt net de charges", f"{res['rdt_net'].median():.2f} %")
    m3.metric("Rdt net-net (fiscal)", f"{res['rdt_net_net'].median():.2f} %")
    m4.metric("Cashflow médian",
              f"{res['cashflow_mens'].median():+,.0f} €/m".replace(",", " "),
              delta_color="normal")
    n_positif = int((res["cashflow_mens"] > 0).sum())
    m5.metric("Secteurs autofinancés", f"{n_positif}/{len(res)}")

    # ─────────────────── CARTE RENTABILITÉ ───────────────────
    st.divider()
    st.markdown("### 🗺️ Cartographie du rendement")

    ci1, ci2 = st.columns([3, 1])

    with ci2:
        indic_inv = st.radio(
            "Indicateur",
            ["rdt_brut", "rdt_net", "rdt_net_net",
             "cashflow_mens", "rdt_fonds_propres", "prix_m2"],
            format_func=lambda k: {
                "rdt_brut": "Rendement brut (%)",
                "rdt_net": "Rdt net de charges (%)",
                "rdt_net_net": "Rdt net-net après impôt (%)",
                "cashflow_mens": "Cashflow (€/mois)",
                "rdt_fonds_propres": "Rdt fonds propres (%)",
                "prix_m2": "Prix d'achat (€/m²)",
            }[k],
        )
        inverse_palette = st.checkbox(
            "Inverser l'échelle", value=(indic_inv == "prix_m2"),
            help="Pour le prix, le vert doit signaler le moins cher.",
        )

    with ci1:
        vals = res.set_index("quartier")[indic_inv].to_dict()
        vmin, vmax = min(vals.values()), max(vals.values())
        if vmin == vmax:
            vmax = vmin + 1

        cols_pal = ["#d73027", "#fee08b", "#1a9850"]
        if inverse_palette:
            cols_pal = cols_pal[::-1]
        cmap_inv = cm.LinearColormap(cols_pal, vmin=vmin, vmax=vmax).to_step(8)
        cmap_inv.caption = {
            "rdt_brut": "Rendement brut (%)",
            "rdt_net": "Rendement net (%)",
            "rdt_net_net": "Rendement net-net (%)",
            "cashflow_mens": "Cashflow (€/mois)",
            "rdt_fonds_propres": "Rdt fonds propres (%)",
            "prix_m2": "Prix (€/m²)",
        }[indic_inv]

        m2map = folium.Map(location=[48.8566, 2.3400], zoom_start=12,
                           tiles="CartoDB dark_matter")

        lookup_inv = res.set_index("quartier").to_dict("index")
        feats_inv = []
        for f in gj["features"]:
            p = f["properties"]
            name = (p.get("l_qu") or p.get("nom_quartier")
                    or p.get("quartier") or "")
            match = next((k for k in lookup_inv if norm(k) == norm(name)), None)
            if match is None:
                continue
            r = lookup_inv[match]
            feats_inv.append({
                "type": "Feature", "geometry": f["geometry"],
                "properties": {
                    "quartier": match,
                    "arr": r["arrondissement"],
                    "prix": f"{r['prix_m2']:,.0f}".replace(",", " "),
                    "loyer": f"{r['loyer_m2']:.1f}",
                    "brut": f"{r['rdt_brut']:.2f}",
                    "net": f"{r['rdt_net']:.2f}",
                    "netnet": f"{r['rdt_net_net']:.2f}",
                    "cf": f"{r['cashflow_mens']:+,.0f}".replace(",", " "),
                    "cout": f"{r['cout_total']:,.0f}".replace(",", " "),
                    "value": r[indic_inv],
                },
            })

        folium.GeoJson(
            {"type": "FeatureCollection", "features": feats_inv},
            style_function=lambda x: {
                "fillColor": cmap_inv(x["properties"]["value"]),
                "color": "#222", "weight": .6, "fillOpacity": .82,
            },
            highlight_function=lambda x: {"weight": 3, "color": "#fff"},
            tooltip=folium.GeoJsonTooltip(
                fields=["quartier", "arr", "prix", "loyer", "brut",
                        "net", "netnet", "cf", "cout"],
                aliases=["Quartier", "Arr.", "Prix €/m²", "Loyer majoré €/m²",
                         "Rdt brut %", "Rdt net %", "Rdt net-net %",
                         "Cashflow €/m", f"Coût total {surface:g} m² €"],
                sticky=True,
                style=("background:#fff;border:1px solid #999;"
                       "border-radius:5px;padding:8px;font-size:12px;"),
            ),
        ).add_to(m2map)

        cmap_inv.add_to(m2map)
        Fullscreen().add_to(m2map)
        st_folium(m2map, height=560, width=None,
                  returned_objects=[], key="map_invest")

    # ─────────────────── CLASSEMENT ───────────────────
    st.divider()
    st.markdown("### 🏆 Classement des secteurs")

    cl1, cl2 = st.columns(2)
    with cl1:
        tri = st.selectbox("Trier par",
                           ["rdt_net_net", "rdt_brut", "rdt_net",
                            "cashflow_mens", "rdt_fonds_propres"],
                           format_func=lambda k: {
                               "rdt_net_net": "Rendement net-net",
                               "rdt_brut": "Rendement brut",
                               "rdt_net": "Rendement net",
                               "cashflow_mens": "Cashflow mensuel",
                               "rdt_fonds_propres": "Rdt fonds propres",
                           }[k])
    with cl2:
        top_n = st.slider("Nombre de secteurs affichés", 5, 80, 20)

    table = (res.sort_values(tri, ascending=False)
             .head(top_n)[["quartier", "arrondissement", "prix_m2",
                           "loyer_m2", "cout_total", "loyer_mensuel",
                           "charges_an", "mensualite", "impot_an",
                           "rdt_brut", "rdt_net", "rdt_net_net",
                           "cashflow_mens", "rdt_fonds_propres"]])

    st.dataframe(
        table, hide_index=True, use_container_width=True, height=480,
        column_config={
            "quartier": "Quartier",
            "arrondissement": st.column_config.NumberColumn("Arr.", format="%d"),
            "prix_m2": st.column_config.NumberColumn("Prix €/m²", format="%.0f"),
            "loyer_m2": st.column_config.NumberColumn("Loyer €/m²", format="%.1f"),
            "cout_total": st.column_config.NumberColumn("Coût total €", format="%.0f"),
            "loyer_mensuel": st.column_config.NumberColumn("Loyer €/m", format="%.0f"),
            "charges_an": st.column_config.NumberColumn("Charges €/an", format="%.0f"),
            "mensualite": st.column_config.NumberColumn("Mensualité €", format="%.0f"),
            "impot_an": st.column_config.NumberColumn("Impôt €/an", format="%.0f"),
            "rdt_brut": st.column_config.NumberColumn("Brut %", format="%.2f"),
            "rdt_net": st.column_config.NumberColumn("Net %", format="%.2f"),
            "rdt_net_net": st.column_config.ProgressColumn(
                "Net-net %", format="%.2f",
                min_value=float(res["rdt_net_net"].min()),
                max_value=float(res["rdt_net_net"].max())),
            "cashflow_mens": st.column_config.NumberColumn(
                "Cashflow €/m", format="%+.0f"),
            "rdt_fonds_propres": st.column_config.NumberColumn(
                "Rdt FP %", format="%.1f"),
        },
    )

    st.download_button(
        "⬇️ Export analyse investisseur (CSV)",
        res.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
        f"rentabilite_paris_{annee}_{int(pieces)}p_{scenario}.csv",
        "text/csv",
    )

    # ─────────────────── GRAPHIQUES ───────────────────
    st.divider()
    g1, g2 = st.columns(2)

    with g1:
        st.markdown("#### Prix vs Loyer plafonné")
        fig = px.scatter(
            res, x="prix_m2", y="loyer_m2",
            size=res["rdt_brut"].clip(lower=.1),
            color="rdt_net_net", hover_name="quartier",
            color_continuous_scale="RdYlGn",
            labels={"prix_m2": "Prix €/m²", "loyer_m2": "Loyer majoré €/m²",
                    "rdt_net_net": "Rdt net-net %"},
        )
        # Iso-rendement brut
        xs = np.linspace(res["prix_m2"].min(), res["prix_m2"].max(), 50)
        for r_iso in [2.5, 3.5, 4.5, 5.5]:
            fig.add_scatter(x=xs, y=xs * r_iso / 100 / 12, mode="lines",
                            line=dict(dash="dot", width=1, color="grey"),
                            name=f"{r_iso}% brut", hoverinfo="skip")
        fig.update_layout(height=430, legend=dict(font=dict(size=9)))
        st.plotly_chart(fig, use_container_width=True)

    with g2:
        st.markdown("#### Rendement net-net par arrondissement")
        fig = px.box(res.dropna(subset=["arrondissement"]),
                     x="arrondissement", y="rdt_net_net",
                     points="all", hover_name="quartier",
                     color_discrete_sequence=["#2e7d32"],
                     labels={"arrondissement": "Arrondissement",
                             "rdt_net_net": "Rdt net-net %"})
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(height=430)
        st.plotly_chart(fig, use_container_width=True)

    # Décomposition du rendement (waterfall)
    st.markdown("#### Décomposition brut → net-net")
    q_sel = st.selectbox("Secteur analysé",
                         res.sort_values("rdt_net_net", ascending=False)["quartier"])
    r = res[res["quartier"] == q_sel].iloc[0]

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "relative", "relative",
                 "relative", "total"],
        x=["Loyer brut", "Vacance/impayés", "Charges",
           "Intérêts + assur.", "Impôt", "Net après crédit"],
        y=[r["loyer_brut_an"],
           -(r["loyer_brut_an"] - r["loyer_encaisse_an"]),
           -r["charges_an"], -(r["annuite"]), -r["impot_an"], 0],
        text=[f"{v:,.0f} €".replace(",", " ") for v in
              [r["loyer_brut_an"],
               -(r["loyer_brut_an"] - r["loyer_encaisse_an"]),
               -r["charges_an"], -r["annuite"], -r["impot_an"],
               r["cashflow_an"]]],
        connector=dict(line=dict(color="grey")),
        increasing=dict(marker_color="#2e7d32"),
        decreasing=dict(marker_color="#c62828"),
        totals=dict(marker_color="#1565c0"),
    ))
    fig.update_layout(height=400, yaxis_title="€ / an",
                      title=f"{q_sel} — {surface:g} m², {int(pieces)} p., {meuble}")
    st.plotly_chart(fig, use_container_width=True)

    # ─────────────────── MONTE-CARLO ───────────────────
    st.divider()
    st.markdown("### 🎲 Analyse de sensibilité (Monte-Carlo)")
    st.caption(
        "Le prix d'achat est tiré dans la fourchette Q1–Q3 observée, "
        "avec un bruit sur la vacance et les charges. "
        "Montre la **distribution réaliste** du rendement plutôt qu'une valeur unique."
    )

    mc1, mc2 = st.columns([1, 3])
    with mc1:
        q_mc = st.selectbox("Secteur", sorted(res["quartier"]), key="mc_q")
        n_sim = st.select_slider("Simulations", [500, 1000, 2000, 5000],
                                 value=1000)
        lancer = st.button("▶️ Lancer", type="primary")

    if lancer:
        rr = res[res["quartier"] == q_mc].iloc[0]
        with st.spinner("Simulation…"):
            mc = monte_carlo(rr["prix_bas"], rr["prix_haut"], rr["loyer_m2"],
                             surface, complement, H, n=n_sim)
        with mc2:
            f1, f2 = st.columns(2)
            with f1:
                fig = px.histogram(mc, x="rdt_net_net", nbins=40,
                                   color_discrete_sequence=["#1565c0"],
                                   labels={"rdt_net_net": "Rdt net-net %"})
                for q, c in [(.1, "orange"), (.5, "black"), (.9, "green")]:
                    fig.add_vline(x=mc["rdt_net_net"].quantile(q),
                                  line_dash="dash", line_color=c,
                                  annotation_text=f"P{int(q*100)}")
                fig.update_layout(height=320, bargap=.03)
                st.plotly_chart(fig, use_container_width=True)
            with f2:
                fig = px.histogram(mc, x="cashflow_mens", nbins=40,
                                   color_discrete_sequence=["#6a1b9a"],
                                   labels={"cashflow_mens": "Cashflow €/mois"})
                fig.add_vline(x=0, line_color="red", line_width=2)
                fig.update_layout(height=320, bargap=.03)
                st.plotly_chart(fig, use_container_width=True)

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Rdt net-net P10", f"{mc['rdt_net_net'].quantile(.1):.2f} %")
        s2.metric("Médiane", f"{mc['rdt_net_net'].median():.2f} %")
        s3.metric("P90", f"{mc['rdt_net_net'].quantile(.9):.2f} %")
        s4.metric("Proba cashflow > 0",
                  f"{(mc['cashflow_mens'] > 0).mean()*100:.0f} %")

    # ─────────────────── AVERTISSEMENT ───────────────────
    st.divider()
st.divider()
st.caption(
    "⚖️ Outil informatif — ne constitue pas un conseil juridique. "
    "Les valeurs officielles font foi : arrêté préfectoral annuel, "
    "consultable sur www.referenceloyer.drihl.ile-de-france.developpement-durable.gouv.fr"
)
