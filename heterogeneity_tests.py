"""
heterogeneity_tests.py
----------------------
Tests économétriques d'hétérogénéité entre pays :

  1. Test de Chow           → rupture structurelle entre groupes de pays
  2. Test de Breusch-Pagan  → hétéroscédasticité des résidus MCO
  3. Test F d'homogénéité   → égalité des vecteurs de coefficients
  4. Test de Pesaran CD     → dépendance inter-sectionnelle (simplifié)
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


REGRESSORS = ["ecb_rate", "inflation", "gdp_growth"]
DEPENDENT  = "credit_growth"


# ─────────────────────────────────────────────
# 1. Test de Chow (rupture structurelle)
# ─────────────────────────────────────────────
def chow_test(df: pd.DataFrame, group1: list[str], group2: list[str]) -> dict:
    """
    Test de Chow pour comparer deux groupes de pays.
    H0 : β_groupe1 = β_groupe2 (pas de rupture structurelle)

    F = [(RSS_R - RSS_U) / k] / [RSS_U / (N - 2k)]
    """
    def _rss_ols(subset: pd.DataFrame) -> tuple[float, int]:
        Y = subset[DEPENDENT]
        X = sm.add_constant(subset[REGRESSORS])
        res = sm.OLS(Y, X).fit()
        return res.ssr, int(res.nobs)

    # Modèle restreint (tous pays ensemble)
    df_all = df[df["country"].isin(group1 + group2)].copy()
    Y_R = df_all[DEPENDENT]
    X_R = sm.add_constant(df_all[REGRESSORS])
    RSS_R = sm.OLS(Y_R, X_R).fit().ssr

    # Modèles non-restreints (séparés)
    RSS_1, N1 = _rss_ols(df[df["country"].isin(group1)])
    RSS_2, N2 = _rss_ols(df[df["country"].isin(group2)])
    RSS_U = RSS_1 + RSS_2

    k = len(REGRESSORS) + 1  # nb de paramètres
    N = N1 + N2

    numerator   = (RSS_R - RSS_U) / k
    denominator = RSS_U / (N - 2 * k)

    F_stat  = numerator / denominator if denominator != 0 else np.nan
    p_value = 1 - stats.f.cdf(F_stat, k, N - 2 * k) if not np.isnan(F_stat) else np.nan

    return {
        "test": "Chow",
        "groupes": f"{group1} vs {group2}",
        "F_stat": round(F_stat, 4),
        "df1": k,
        "df2": N - 2 * k,
        "p_value": round(p_value, 5),
        "conclusion": _conclude(p_value, "Hétérogénéité structurelle significative"),
    }


# ─────────────────────────────────────────────
# 2. Test de Breusch-Pagan (hétéroscédasticité)
# ─────────────────────────────────────────────
def breusch_pagan_test(df: pd.DataFrame, country: str) -> dict:
    """
    Test de Breusch-Pagan pour un pays.
    H0 : homoscédasticité (σ² constant)
    """
    from statsmodels.stats.diagnostic import het_breuschpagan

    df_c = df[df["country"] == country].copy()
    Y = df_c[DEPENDENT]
    X = sm.add_constant(df_c[REGRESSORS])
    res = sm.OLS(Y, X).fit()

    lm_stat, lm_p, f_stat, f_p = het_breuschpagan(res.resid, X)

    return {
        "test": "Breusch-Pagan",
        "pays": country,
        "LM_stat": round(lm_stat, 4),
        "p_value": round(lm_p, 5),
        "conclusion": _conclude(lm_p, "Hétéroscédasticité détectée"),
    }


# ─────────────────────────────────────────────
# 3. Test F global d'homogénéité des coefficients
# ─────────────────────────────────────────────
def f_homogeneity_test(df: pd.DataFrame, country_results: dict) -> dict:
    """
    Test F d'égalité des vecteurs de coefficients entre pays.
    Compare la somme des RSS individuels au RSS du modèle poolé.

    F = [(RSS_pooled - Σ RSS_i) / (K*(G-1))] / [Σ RSS_i / (N - G*K)]
    Où G = nb pays, K = nb paramètres, N = nb observations total
    """
    # RSS poolé
    Y_pool = df[DEPENDENT]
    X_pool = sm.add_constant(df[REGRESSORS])
    RSS_pooled = sm.OLS(Y_pool, X_pool).fit().ssr

    # Somme des RSS individuels
    total_rss = sum(res.ssr for res in country_results.values())
    total_n   = sum(int(res.nobs) for res in country_results.values())

    G = len(country_results)
    K = len(REGRESSORS) + 1

    num   = (RSS_pooled - total_rss) / (K * (G - 1))
    denom = total_rss / (total_n - G * K)

    F_stat  = num / denom if denom != 0 else np.nan
    p_value = 1 - stats.f.cdf(F_stat, K * (G - 1), total_n - G * K) if not np.isnan(F_stat) else np.nan

    return {
        "test": "F homogénéité",
        "F_stat": round(F_stat, 4),
        "df1": K * (G - 1),
        "df2": total_n - G * K,
        "p_value": round(p_value, 6),
        "conclusion": _conclude(p_value, "Hétérogénéité entre pays confirmée ✅"),
    }


# ─────────────────────────────────────────────
# 4. Test de dépendance inter-sectionnelle de Pesaran (CD)
# ─────────────────────────────────────────────
def pesaran_cd_test(df: pd.DataFrame) -> dict:
    """
    Test CD de Pesaran — dépendance transversale des résidus.
    H0 : résidus indépendants entre pays
    """
    countries = list(df["country"].unique())
    residuals = {}

    for c in countries:
        df_c = df[df["country"] == c]
        Y = df_c[DEPENDENT]
        X = sm.add_constant(df_c[REGRESSORS])
        res = sm.OLS(Y, X).fit()
        residuals[c] = res.resid.values

    T = min(len(r) for r in residuals.values())
    G = len(countries)

    # Matrice de corrélations des résidus
    corr_sum = 0
    pairs = 0
    for i in range(G):
        for j in range(i + 1, G):
            r_i = residuals[countries[i]][:T]
            r_j = residuals[countries[j]][:T]
            rho_ij = np.corrcoef(r_i, r_j)[0, 1]
            corr_sum += rho_ij
            pairs += 1

    CD_stat = np.sqrt(2 * T / (G * (G - 1))) * corr_sum
    p_value = 2 * (1 - stats.norm.cdf(abs(CD_stat)))

    return {
        "test": "Pesaran CD",
        "CD_stat": round(CD_stat, 4),
        "p_value": round(p_value, 5),
        "conclusion": _conclude(p_value, "Dépendance cross-sectionnelle significative"),
    }


def _conclude(p_value: float, reject_msg: str, alpha: float = 0.05) -> str:
    if np.isnan(p_value):
        return "Indéterminé"
    if p_value < alpha:
        return f"H0 rejetée — {reject_msg} (p={p_value:.4f})"
    return f"H0 non rejetée — Résultat non significatif (p={p_value:.4f})"


def run_all_tests(df: pd.DataFrame, country_results: dict) -> None:
    """Lance l'ensemble des tests et affiche les résultats."""
    print("\n" + "=" * 65)
    print("TESTS D'HÉTÉROGÉNÉITÉ")
    print("=" * 65)

    # Test de Chow : Cœur vs Périphérie
    core = ["Allemagne", "France", "Pays-Bas"]
    periphery = ["Italie", "Espagne"]
    chow = chow_test(df, core, periphery)
    _print_test(chow)

    # Test F global
    f_test = f_homogeneity_test(df, country_results)
    _print_test(f_test)

    # Breusch-Pagan par pays
    print("\n[BREUSCH-PAGAN — Hétéroscédasticité par pays]")
    for country in df["country"].unique():
        bp = breusch_pagan_test(df, country)
        _print_test(bp, short=True)

    # Pesaran CD
    cd = pesaran_cd_test(df)
    _print_test(cd)

    print("=" * 65)


def _print_test(result: dict, short: bool = False) -> None:
    test_name = result.get("test", "")
    if short:
        pays = result.get("pays", "")
        print(f"  {pays:<12}: p={result['p_value']} — {result['conclusion']}")
    else:
        print(f"\n[{test_name.upper()}]")
        for k, v in result.items():
            if k not in ("test",):
                print(f"  {k:<18}: {v}")
