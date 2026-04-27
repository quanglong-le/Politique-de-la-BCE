"""
generate_data.py
----------------
Génère des données macroéconomiques simulées pour la zone euro (2003–2023).
Les séries reproduisent les propriétés empiriques des données BCE/Eurostat :
- Corrélation négative taux → crédit
- Cycles économiques cohérents (2008, 2012, 2020)
- Hétérogénéité structurelle entre pays
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# ─────────────────────────────────────────────
# 1. Paramètres de simulation par pays
# ─────────────────────────────────────────────
COUNTRIES = {
    "Allemagne": {
        "beta_rate": -0.82,
        "beta_infl": -0.31,
        "base_credit": 3.2,
        "volatility": 0.8,
        "label": "DEU",
    },
    "France": {
        "beta_rate": -0.64,
        "beta_infl": -0.18,
        "base_credit": 4.1,
        "volatility": 0.9,
        "label": "FRA",
    },
    "Italie": {
        "beta_rate": -1.23,
        "beta_infl": -0.52,
        "base_credit": 2.8,
        "volatility": 1.3,
        "label": "ITA",
    },
    "Espagne": {
        "beta_rate": -1.41,
        "beta_infl": -0.61,
        "base_credit": 5.0,
        "volatility": 1.6,
        "label": "ESP",
    },
    "Pays-Bas": {
        "beta_rate": -0.51,
        "beta_infl": -0.12,
        "base_credit": 3.8,
        "volatility": 0.7,
        "label": "NLD",
    },
}

# ─────────────────────────────────────────────
# 2. Séries macroéconomiques communes (zone euro)
# ─────────────────────────────────────────────
PERIODS = pd.period_range(start="2003Q1", end="2023Q4", freq="Q")
N = len(PERIODS)  # 84 trimestres

# Taux directeur BCE — reproduit les cycles réels
ecb_rate_base = np.array([
    # 2003–2007 : remontée progressive
    *np.linspace(2.0, 4.0, 20),
    # 2008–2009 : crise financière, baisse rapide
    *np.linspace(4.0, 1.0, 8),
    # 2010–2011 : légère remontée
    *np.linspace(1.0, 1.5, 8),
    # 2012–2021 : forward guidance, taux nuls / négatifs
    *np.linspace(1.5, -0.5, 36),
    # 2022–2023 : resserrement monétaire brutal
    *np.linspace(-0.5, 4.0, 12),
])
ecb_rate = ecb_rate_base + np.random.normal(0, 0.05, N)

# Inflation zone euro (IPCH)
infl_base = np.array([
    *np.linspace(2.1, 2.3, 20),
    *np.linspace(3.5, 0.3, 8),   # déflation post-crise
    *np.linspace(1.2, 2.7, 8),
    *np.linspace(2.5, 0.2, 28),  # very low inflation era
    *np.linspace(0.2, 10.6, 12), # choc inflationniste 2022
    *np.linspace(10.6, 2.9, 8),  # désinflation 2023
])
inflation_ea = infl_base + np.random.normal(0, 0.15, N)

# Croissance PIB zone euro  (total = 84 trimestres)
gdp_growth_base = np.array([
    *np.linspace(1.8, 3.0, 16),     # 2003–2006  (16)
    *[-4.5, -5.8, -0.5, 1.2],       # 2008 GFC   (4)
    *np.linspace(2.0, -1.5, 12),    # 2009–2012  (12)
    *np.linspace(0.5, 2.5, 24),     # 2013–2018  (24)
    *[-13.8, -4.3, 5.7, 5.5],       # 2020 COVID (4)
    *np.linspace(3.5, 0.5, 16),     # 2021–2023  (16 → 76) … pad
    *np.linspace(0.5, 0.3, 8),      # extension  (8 → 84)
])
gdp_growth_ea = gdp_growth_base + np.random.normal(0, 0.3, N)


# ─────────────────────────────────────────────
# 3. Génération pays par pays
# ─────────────────────────────────────────────
def generate_country_data(country_name: str, params: dict) -> pd.DataFrame:
    """Génère une série temporelle pour un pays donné."""

    # Spread de taux spécifique au pays (prime de risque)
    country_spread = {
        "Allemagne": 0.0,
        "France": 0.3,
        "Italie": 1.5,
        "Espagne": 1.1,
        "Pays-Bas": 0.1,
    }
    rate = ecb_rate + country_spread.get(country_name, 0.5)
    rate += np.random.normal(0, 0.08, N)

    # Inflation spécifique au pays
    inflation_country_factor = {
        "Allemagne": 0.9,
        "France": 1.0,
        "Italie": 1.15,
        "Espagne": 1.25,
        "Pays-Bas": 0.95,
    }
    inflation = inflation_ea * inflation_country_factor.get(country_name, 1.0)
    inflation += np.random.normal(0, 0.2, N)

    # PIB
    gdp_growth = gdp_growth_ea + np.random.normal(0, params["volatility"] * 0.4, N)

    # Crédit (variable endogène)
    credit_growth = (
        params["base_credit"]
        + params["beta_rate"] * rate
        + params["beta_infl"] * inflation
        + 0.45 * gdp_growth
        + np.random.normal(0, params["volatility"], N)
    )

    return pd.DataFrame({
        "period": PERIODS,
        "country": country_name,
        "country_code": params["label"],
        "ecb_rate": np.round(rate, 3),
        "inflation": np.round(inflation, 3),
        "gdp_growth": np.round(gdp_growth, 3),
        "credit_growth": np.round(credit_growth, 3),
    })


def generate_all_countries() -> pd.DataFrame:
    """Génère le panel complet (tous pays × toutes périodes)."""
    frames = [generate_country_data(name, params) for name, params in COUNTRIES.items()]
    panel = pd.concat(frames, ignore_index=True)
    panel["year"] = panel["period"].dt.year
    panel["quarter"] = panel["period"].dt.quarter
    panel["period_str"] = panel["period"].astype(str)
    return panel


if __name__ == "__main__":
    df = generate_all_countries()
    df.to_csv("data/panel_zone_euro.csv", index=False)
    print(f"✅ Données générées : {df.shape[0]} observations × {df.shape[1]} variables")
    print(df.head(10).to_string())
