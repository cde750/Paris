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
tab_carte, tab_simu, tab_data, tab_analyse = st.tabs(
    ["🗺️ Carte", "🧮 Simulateur", "📋 Données", "📈 Analyses"]
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
st.divider()
st.caption(
    "⚖️ Outil informatif — ne constitue pas un conseil juridique. "
    "Les valeurs officielles font foi : arrêté préfectoral annuel, "
    "consultable sur www.referenceloyer.drihl.ile-de-france.developpement-durable.gouv.fr"
)
