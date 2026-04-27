"""
panel_models.py
---------------
Modèles de panel économétriques :
  - Effets fixes (Within estimator)
  - Effets aléatoires (GLS)
  - Test de Hausman (FE vs RE)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS


REGRESSORS = ["ecb_rate", "inflation", "gdp_growth"]
DEPENDENT  = "credit_growth"


def _within_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Demeaning par groupe pays (transformation Within)."""
    df_fe = df.copy()
    for col in REGRESSORS + [DEPENDENT]:
        group_mean = df_fe.groupby("country")[col].transform("mean")
        df_fe[col + "_dm"] = df_fe[col] - group_mean
    return df_fe


def run_fixed_effects(df: pd.DataFrame):
    """
    Estimateur Within (effets fixes pays).
    Élimine l'hétérogénéité non-observée invariante dans le temps.
    """
    df_fe = _within_transform(df)
    Y = df_fe[DEPENDENT + "_dm"]
    X = df_fe[[r + "_dm" for r in REGRESSORS]]
    X = sm.add_constant(X)
    model_fe = OLS(Y, X).fit(cov_type="HC3")
    return model_fe


def run_random_effects(df: pd.DataFrame):
    """
    Estimateur GLS (effets aléatoires — approche manuelle simplifiée).
    """
    # Simplification : RE ≈ OLS avec dummies pays (LSDV)
    df_re = df[REGRESSORS + [DEPENDENT, "country"]].copy()
    dummies = pd.get_dummies(df_re["country"], drop_first=True, dtype=float)
    country_dummies = list(dummies.columns)
    df_re = pd.concat([df_re.drop(columns=["country"]), dummies], axis=1)

    Y = df_re[DEPENDENT].astype(float)
    X_data = df_re[REGRESSORS + country_dummies].astype(float)
    X = sm.add_constant(X_data)
    model_re = OLS(Y, X).fit(cov_type="HC3")
    return model_re


def hausman_test(fe_model, re_model, df: pd.DataFrame) -> dict:
    """
    Test de Hausman simplifié.
    H0 : RE efficient (pas de corrélation entre effets individuels et régresseurs)
    H1 : FE nécessaire (endogénéité des effets individuels)

    Statistique : H = (b_FE - b_RE)' [Var(b_FE) - Var(b_RE)]^{-1} (b_FE - b_RE) ~ χ²(k)
    """
    regressors_dm = [r + "_dm" for r in REGRESSORS]

    # Coefficients FE (sans la constante demeaning)
    b_fe = fe_model.params[regressors_dm].values

    # Coefficients RE pour les mêmes variables
    b_re = re_model.params[REGRESSORS].values

    # Différence
    diff = b_fe - b_re

    # Matrices de variance
    var_fe = fe_model.cov_params().loc[regressors_dm, regressors_dm].values
    var_re = re_model.cov_params().loc[REGRESSORS, REGRESSORS].values

    var_diff = var_fe - var_re

    try:
        # Statistique de Hausman
        from numpy.linalg import inv, pinv
        H = float(diff @ pinv(var_diff) @ diff)
    except Exception:
        H = np.nan

    from scipy.stats import chi2
    k = len(REGRESSORS)
    p_value = 1 - chi2.cdf(H, df=k) if not np.isnan(H) else np.nan

    return {
        "H_stat": round(H, 4) if not np.isnan(H) else "N/A",
        "df": k,
        "p_value": round(p_value, 4) if not np.isnan(p_value) else "N/A",
        "conclusion": "Effets Fixes préférés (H0 rejetée)" if (not np.isnan(p_value) and p_value < 0.05)
                      else "Effets Aléatoires acceptables (H0 non rejetée)",
    }


def print_panel_summary(fe_model, re_model, hausman: dict) -> None:
    print("\n" + "=" * 65)
    print("MODÈLES DE PANEL")
    print("=" * 65)
    print("\n[EFFETS FIXES — Within estimator]")
    regressors_dm = [r + "_dm" for r in REGRESSORS]
    for r, rd in zip(REGRESSORS, regressors_dm):
        coef = fe_model.params.get(rd, np.nan)
        pval = fe_model.pvalues.get(rd, np.nan)
        stars = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.10 else ""))
        print(f"  {r:<15}: {coef:+.4f} {stars}")
    print(f"  R² within   : {fe_model.rsquared:.4f}")

    print("\n[EFFETS ALÉATOIRES — GLS/LSDV]")
    for r in REGRESSORS:
        coef = re_model.params.get(r, np.nan)
        pval = re_model.pvalues.get(r, np.nan)
        stars = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.10 else ""))
        print(f"  {r:<15}: {coef:+.4f} {stars}")
    print(f"  R²          : {re_model.rsquared:.4f}")

    print("\n[TEST DE HAUSMAN]")
    print(f"  H-statistique  : {hausman['H_stat']}")
    print(f"  Degrés liberté : {hausman['df']}")
    print(f"  p-value        : {hausman['p_value']}")
    print(f"  → {hausman['conclusion']}")
    print("=" * 65)
