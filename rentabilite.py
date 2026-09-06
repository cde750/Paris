"""
Moteur de calcul de rentabilité locative.
Toutes les hypothèses sont explicites et paramétrables.
"""

from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd


@dataclass
class HypothesesInvest:
    # --- Acquisition ---
    frais_notaire_pct: float = 7.5        # ancien ; 2-3 % en neuf
    frais_agence_pct: float = 4.0         # 0 si de particulier à particulier
    travaux_eur_m2: float = 300.0         # rafraîchissement léger
    mobilier_eur_m2: float = 0.0          # si meublé : 150-350 €/m²

    # --- Exploitation (annuel) ---
    charges_copro_eur_m2_an: float = 30.0     # non récupérables : ~25-40 €/m²/an
    taxe_fonciere_pct_loyer: float = 8.0      # ~0,8-1,2 mois de loyer
    assurance_pno_eur_an: float = 180.0
    gestion_locative_pct: float = 0.0         # 6-8 % TTC si agence
    vacance_locative_pct: float = 4.0         # ~2 sem./an en zone tendue
    provision_travaux_pct_loyer: float = 4.0  # gros entretien / ravalement
    impayes_pct: float = 1.0

    # --- Financement ---
    apport_pct: float = 20.0
    taux_credit_pct: float = 3.60
    duree_credit_ans: int = 20
    taux_assurance_emprunteur_pct: float = 0.34

    # --- Fiscalité ---
    regime: str = "LMNP amortissement"   # ou "Micro-foncier", "Réel foncier", "Micro-BIC"
    tmi_pct: float = 30.0
    ps_pct: float = 17.2                 # prélèvements sociaux
    duree_amort_bati_ans: int = 30
    part_terrain_pct: float = 15.0        # non amortissable


# ══════════════════════════════════════════════════════════════════
def cout_acquisition(prix_m2: float, surface: float, h: HypothesesInvest) -> dict:
    prix = prix_m2 * surface
    notaire = prix * h.frais_notaire_pct / 100
    agence = prix * h.frais_agence_pct / 100
    travaux = h.travaux_eur_m2 * surface
    mobilier = h.mobilier_eur_m2 * surface
    total = prix + notaire + agence + travaux + mobilier
    return dict(prix=prix, notaire=notaire, agence=agence,
                travaux=travaux, mobilier=mobilier, total=total)


def loyer_annuel(loyer_m2: float, surface: float, complement: float,
                 h: HypothesesInvest) -> dict:
    brut = (loyer_m2 * surface + complement) * 12
    perte_vacance = brut * h.vacance_locative_pct / 100
    perte_impayes = brut * h.impayes_pct / 100
    encaisse = brut - perte_vacance - perte_impayes
    return dict(brut=brut, vacance=perte_vacance,
                impayes=perte_impayes, encaisse=encaisse)


def charges_annuelles(surface: float, loyer_brut: float,
                      h: HypothesesInvest) -> dict:
    copro = h.charges_copro_eur_m2_an * surface
    tf = loyer_brut * h.taxe_fonciere_pct_loyer / 100
    pno = h.assurance_pno_eur_an
    gestion = loyer_brut * h.gestion_locative_pct / 100
    prov = loyer_brut * h.provision_travaux_pct_loyer / 100
    return dict(copro=copro, taxe_fonciere=tf, pno=pno,
                gestion=gestion, provision=prov,
                total=copro + tf + pno + gestion + prov)


def mensualite_credit(capital: float, taux_pct: float, duree_ans: int,
                      taux_assur_pct: float) -> dict:
    if capital <= 0:
        return dict(mensualite=0.0, interets_an1=0.0, assurance_an=0.0)
    t = taux_pct / 100 / 12
    n = duree_ans * 12
    m = capital * t / (1 - (1 + t) ** -n) if t > 0 else capital / n
    assur_an = capital * taux_assur_pct / 100
    # Intérêts année 1 (approximation par amortissement réel)
    solde, interets = capital, 0.0
    for _ in range(12):
        i = solde * t
        interets += i
        solde -= (m - i)
    return dict(mensualite=m, interets_an1=interets, assurance_an=assur_an)


def impot_annuel(loyer_encaisse: float, charges_deduct: float,
                 interets: float, prix_bien: float, surface: float,
                 h: HypothesesInvest) -> dict:
    """Calcul simplifié — 1re année, sans report de déficit pluriannuel."""
    tx = (h.tmi_pct + h.ps_pct) / 100

    if h.regime == "Micro-foncier":
        base = loyer_encaisse * 0.70            # abattement 30 %
        return dict(base=base, impot=base * tx, amortissement=0.0)

    if h.regime == "Micro-BIC":
        base = loyer_encaisse * 0.50            # abattement 50 % (meublé)
        return dict(base=base, impot=base * tx, amortissement=0.0)

    if h.regime == "Réel foncier":
        base = max(0.0, loyer_encaisse - charges_deduct - interets)
        return dict(base=base, impot=base * tx, amortissement=0.0)

    # LMNP au réel avec amortissement
    amort_bati = (prix_bien * (1 - h.part_terrain_pct / 100)
                  / h.duree_amort_bati_ans)
    amort_mob = (h.mobilier_eur_m2 * surface / 7) if h.mobilier_eur_m2 else 0.0
    amort = amort_bati + amort_mob
    base = max(0.0, loyer_encaisse - charges_deduct - interets - amort)
    return dict(base=base, impot=base * tx, amortissement=amort)


# ══════════════════════════════════════════════════════════════════
def analyser(prix_m2: float, loyer_m2: float, surface: float,
             complement: float, h: HypothesesInvest) -> dict:
    """Calcul complet pour un couple (prix, loyer)."""
    acq = cout_acquisition(prix_m2, surface, h)
    loy = loyer_annuel(loyer_m2, surface, complement, h)
    chg = charges_annuelles(surface, loy["brut"], h)

    emprunt = acq["total"] * (1 - h.apport_pct / 100)
    apport = acq["total"] - emprunt
    cred = mensualite_credit(emprunt, h.taux_credit_pct,
                             h.duree_credit_ans, h.taux_assurance_emprunteur_pct)

    charges_fisc = chg["total"] - chg["provision"]  # provision non déductible
    fisc = impot_annuel(loy["encaisse"], charges_fisc,
                        cred["interets_an1"], acq["prix"], surface, h)

    # --- Indicateurs ---
    rdt_brut = loy["brut"] / acq["prix"] * 100 if acq["prix"] else 0
    rdt_brut_fai = loy["brut"] / acq["total"] * 100 if acq["total"] else 0
    net_charges = loy["encaisse"] - chg["total"]
    rdt_net = net_charges / acq["total"] * 100 if acq["total"] else 0
    net_net = net_charges - fisc["impot"]
    rdt_net_net = net_net / acq["total"] * 100 if acq["total"] else 0

    annuite = cred["mensualite"] * 12 + cred["assurance_an"]
    cashflow_an = net_net - annuite
    cashflow_mens = cashflow_an / 12

    # Rendement sur fonds propres (effet de levier)
    rdt_fp = cashflow_an / apport * 100 if apport > 0 else np.nan

    # Effort d'épargne / point mort
    seuil_autofin = (annuite + chg["total"] + fisc["impot"]) / 12 / surface \
        if surface else np.nan

    return dict(
        # Acquisition
        prix_m2=prix_m2, prix=acq["prix"], cout_total=acq["total"],
        apport=apport, emprunt=emprunt,
        # Loyers
        loyer_m2=loyer_m2, loyer_mensuel=loy["brut"] / 12,
        loyer_brut_an=loy["brut"], loyer_encaisse_an=loy["encaisse"],
        # Charges
        charges_an=chg["total"], detail_charges=chg,
        # Crédit
        mensualite=cred["mensualite"], interets_an1=cred["interets_an1"],
        annuite=annuite,
        # Fiscal
        base_imposable=fisc["base"], impot_an=fisc["impot"],
        amortissement=fisc["amortissement"],
        # Rendements
        rdt_brut=rdt_brut, rdt_brut_fai=rdt_brut_fai,
        rdt_net=rdt_net, rdt_net_net=rdt_net_net, rdt_fonds_propres=rdt_fp,
        # Cashflow
        cashflow_mens=cashflow_mens, cashflow_an=cashflow_an,
        loyer_equilibre_m2=seuil_autofin,
    )


def analyser_secteurs(df: pd.DataFrame, surface: float, complement: float,
                      h: HypothesesInvest, scenario: str = "median") -> pd.DataFrame:
    """
    Applique l'analyse à tous les quartiers.
    scenario : 'bas' (prix Q1 = opportunité), 'median', 'haut' (prix Q3)
    """
    col_prix = {"bas": "prix_m2_bas", "median": "prix_m2_median",
                "haut": "prix_m2_haut"}[scenario]

    out = []
    for _, r in df.iterrows():
        res = analyser(r[col_prix], r["loyer_max"], surface, complement, h)
        res.update(quartier=r["quartier"], arrondissement=r["arrondissement"],
                   id_zone=r.get("id_zone"),
                   prix_bas=r["prix_m2_bas"], prix_haut=r["prix_m2_haut"],
                   loyer_ref=r.get("loyer_ref"), n_ventes=r.get("n_ventes"))
        out.append(res)
    return pd.DataFrame(out)


def monte_carlo(prix_m2_bas: float, prix_m2_haut: float, loyer_m2: float,
                surface: float, complement: float, h: HypothesesInvest,
                n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """
    Simulation : tire le prix dans la fourchette observée + bruit sur
    vacance et charges. Donne une distribution de rendement net-net.
    """
    rng = np.random.default_rng(seed)
    prix = rng.triangular(prix_m2_bas, (prix_m2_bas + prix_m2_haut) / 2,
                          prix_m2_haut, n)
    vac = np.clip(rng.normal(h.vacance_locative_pct, 2.0, n), 0, 20)
    chg = np.clip(rng.normal(h.charges_copro_eur_m2_an, 8.0, n), 10, 70)

    rows = []
    for p, v, c in zip(prix, vac, chg):
        hh = HypothesesInvest(**{**asdict(h),
                                 "vacance_locative_pct": float(v),
                                 "charges_copro_eur_m2_an": float(c)})
        r = analyser(p, loyer_m2, surface, complement, hh)
        rows.append(dict(prix_m2=p, rdt_net_net=r["rdt_net_net"],
                         cashflow_mens=r["cashflow_mens"],
                         rdt_brut=r["rdt_brut"]))
    return pd.DataFrame(rows)
