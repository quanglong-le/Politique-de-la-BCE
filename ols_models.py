"""
ols_models.py
-------------
Estimation des modèles MCO (OLS) :
  - Pays par pays (5 régressions individuelles)
  - Modèle poolé (toutes observations)
  - Récapitulatif des coefficients et R²
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS, RegressionResultsWrapper


REGRESSORS = ["ecb_rate", "inflation", "gdp_growth"]
DEPENDENT  = "credit_growth"


def run_ols_country(df_country: pd.DataFrame, country: str) -> RegressionResultsWrapper:
    """MCO pour un seul pays."""
    Y = df_country[DEPENDENT]
    X = sm.add_constant(df_country[REGRESSORS])
    model = OLS(Y, X).fit(cov_type="HC3")   # erreurs robustes à l'hétéroscédasticité
    return model


def run_ols_all_countries(df: pd.DataFrame) -> dict[str, RegressionResultsWrapper]:
    """Lance une régression MCO pour chaque pays."""
    results = {}
    countries = df["country"].unique()
    for country in countries:
        df_c = df[df["country"] == country].copy()
        results[country] = run_ols_country(df_c, country)
    return results


def run_pooled_ols(df: pd.DataFrame) -> RegressionResultsWrapper:
    """MCO poolé — toutes observations, sans effets pays."""
    Y = df[DEPENDENT]
    X = sm.add_constant(df[REGRESSORS])
    model = OLS(Y, X).fit(cov_type="HC3")
    return model


def summarize_results(country_results: dict, pooled_result: RegressionResultsWrapper) -> pd.DataFrame:
    """
    Construit un tableau récapitulatif :
    Pays | β_taux | β_inflation | β_PIB | R² | N | Significativité
    """
    rows = []
    for country, res in country_results.items():
        rows.append(_extract_row(country, res))

    rows.append(_extract_row("Pooled (all)", pooled_result))

    summary = pd.DataFrame(rows)
    return summary


def _extract_row(label: str, res: RegressionResultsWrapper) -> dict:
    params = res.params
    pvals  = res.pvalues

    def sig_stars(p):
        if p < 0.01:  return "***"
        if p < 0.05:  return "**"
        if p < 0.10:  return "*"
        return ""

    return {
        "Pays": label,
        "β_taux": round(params.get("ecb_rate", np.nan), 3),
        "p_taux": round(pvals.get("ecb_rate", np.nan), 3),
        "sig_taux": sig_stars(pvals.get("ecb_rate", 1)),
        "β_inflation": round(params.get("inflation", np.nan), 3),
        "p_infl": round(pvals.get("inflation", np.nan), 3),
        "sig_infl": sig_stars(pvals.get("inflation", 1)),
        "β_PIB": round(params.get("gdp_growth", np.nan), 3),
        "p_pib": round(pvals.get("gdp_growth", np.nan), 3),
        "sig_pib": sig_stars(pvals.get("gdp_growth", 1)),
        "R²": round(res.rsquared, 3),
        "R²_adj": round(res.rsquared_adj, 3),
        "N": int(res.nobs),
    }


def print_summary(df_summary: pd.DataFrame) -> None:
    """Affiche le tableau de résultats formaté."""
    print("\n" + "=" * 75)
    print("RÉSULTATS MCO — TRANSMISSION MONÉTAIRE PAR PAYS")
    print("=" * 75)
    display = df_summary[["Pays", "β_taux", "sig_taux", "β_inflation",
                           "sig_infl", "β_PIB", "sig_pib", "R²", "N"]].copy()
    display.columns = ["Pays", "β Taux", "***", "β Inflation", "***",
                       "β PIB", "***", "R²", "N"]
    print(display.to_string(index=False))
    print("\nSignificativité : *** p<0.01  ** p<0.05  * p<0.10")
    print("Erreurs standard robustes (HC3 — White)")
    print("=" * 75)
