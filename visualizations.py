"""
visualizations.py
-----------------
Production de tous les graphiques de l'analyse :
  1. Séries temporelles taux BCE + crédit par pays
  2. Coefficients β par pays (forest plot)
  3. Heatmap des corrélations
  4. Résidus MCO et diagnostic
  5. Transmission monétaire — scatter taux vs crédit
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

os.makedirs("outputs/figures", exist_ok=True)

# ─── Style global ──────────────────────────────────────────
PALETTE = {
    "Allemagne": "#1f4e79",
    "France":    "#2e75b6",
    "Italie":    "#c00000",
    "Espagne":   "#e36c09",
    "Pays-Bas":  "#538135",
}
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right": False,
})
SHADE_EVENTS = [
    ("2008Q3", "2009Q2", "Crise 2008"),
    ("2011Q3", "2012Q4", "Crise souv."),
    ("2020Q1", "2020Q3", "COVID"),
]


def _shade_crises(ax, df, alpha=0.12):
    """Ajoute des zones grisées pour les périodes de crise."""
    periods = df["period_str"].unique()
    periods_sorted = sorted(periods)
    for start, end, label in SHADE_EVENTS:
        if start in periods_sorted and end in periods_sorted:
            x0 = periods_sorted.index(start)
            x1 = periods_sorted.index(end)
            ax.axvspan(x0, x1, color="gray", alpha=alpha, label=label)


# ─────────────────────────────────────────────
# Fig. 1 — Séries temporelles taux & crédit
# ─────────────────────────────────────────────
def plot_time_series(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        "Transmission Monétaire BCE — Zone Euro (2003–2023)\n"
        "Taux directeur · Inflation · Crédit bancaire",
        fontsize=14, fontweight="bold", y=1.01,
    )

    periods = sorted(df["period_str"].unique())
    x_ticks = range(0, len(periods), 8)
    x_labels = [periods[i] for i in x_ticks]

    # Panel A — Taux BCE (identique pour tous les pays)
    df_deu = df[df["country"] == "Allemagne"].sort_values("period_str")
    axes[0].plot(df_deu["ecb_rate"].values, color="#1f4e79", lw=2.2, label="Taux BCE")
    axes[0].axhline(0, color="gray", ls="--", lw=0.8)
    axes[0].set_ylabel("Taux directeur (%)", fontsize=10)
    axes[0].legend(loc="upper right", fontsize=9)

    # Panel B — Inflation par pays
    for country, color in PALETTE.items():
        df_c = df[df["country"] == country].sort_values("period_str")
        axes[1].plot(df_c["inflation"].values, color=color, lw=1.6,
                     label=country, alpha=0.85)
    axes[1].set_ylabel("Inflation IPCH (%)", fontsize=10)
    axes[1].legend(loc="upper left", fontsize=8, ncol=2)

    # Panel C — Crédit par pays
    for country, color in PALETTE.items():
        df_c = df[df["country"] == country].sort_values("period_str")
        axes[2].plot(df_c["credit_growth"].values, color=color, lw=1.6,
                     label=country, alpha=0.85)
    axes[2].axhline(0, color="gray", ls="--", lw=0.8)
    axes[2].set_ylabel("Croissance du crédit (%)", fontsize=10)
    axes[2].legend(loc="upper left", fontsize=8, ncol=2)

    for ax in axes:
        ax.set_xticks(list(x_ticks))
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
        for start, end, label in SHADE_EVENTS:
            if start in periods and end in periods:
                x0 = periods.index(start)
                x1 = periods.index(end)
                ax.axvspan(x0, x1, color="gray", alpha=0.10)

    plt.tight_layout()
    path = "outputs/figures/fig1_time_series.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {path}")


# ─────────────────────────────────────────────
# Fig. 2 — Forest plot des β par pays
# ─────────────────────────────────────────────
def plot_coefficients(country_results: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Coefficients MCO par pays — Intervalles de confiance à 95%",
                 fontsize=13, fontweight="bold")

    variables = [
        ("ecb_rate",    "β Taux directeur BCE",  "_dm"),
        ("inflation",   "β Inflation IPCH",       "_dm"),
        ("gdp_growth",  "β Croissance PIB",       "_dm"),
    ]

    countries = list(country_results.keys())
    colors = [PALETTE.get(c, "#555") for c in countries]
    y_pos  = np.arange(len(countries))

    for ax, (var, title, suffix) in zip(axes, variables):
        coefs  = []
        ci_low = []
        ci_high = []

        for country in countries:
            res = country_results[country]
            # Tentative avec le nom demeaned, sinon original
            var_key = var + suffix if (var + suffix) in res.params.index else var
            if var_key not in res.params.index:
                coefs.append(np.nan); ci_low.append(np.nan); ci_high.append(np.nan)
                continue
            c  = res.params[var_key]
            ci = res.conf_int().loc[var_key]
            coefs.append(c)
            ci_low.append(ci[0])
            ci_high.append(ci[1])

        err_low  = [c - l for c, l in zip(coefs, ci_low)]
        err_high = [h - c for c, h in zip(coefs, ci_high)]

        ax.barh(y_pos, coefs, xerr=[err_low, err_high],
                color=colors, alpha=0.8, capsize=4, ecolor="#333", height=0.55)
        ax.axvline(0, color="black", lw=1.0, ls="--")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(countries, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Coefficient (pp)", fontsize=9)

    plt.tight_layout()
    path = "outputs/figures/fig2_coefficients.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {path}")


# ─────────────────────────────────────────────
# Fig. 3 — Heatmap des corrélations
# ─────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    fig.suptitle("Matrices de corrélation par pays",
                 fontsize=13, fontweight="bold")

    vars_to_corr = ["credit_growth", "ecb_rate", "inflation", "gdp_growth"]
    labels = ["Crédit", "Taux BCE", "Inflation", "PIB"]

    for idx, country in enumerate(PALETTE.keys()):
        df_c = df[df["country"] == country][vars_to_corr]
        corr = df_c.corr()
        corr.index   = labels
        corr.columns = labels
        sns.heatmap(corr, ax=axes[idx], annot=True, fmt=".2f",
                    cmap="RdBu_r", vmin=-1, vmax=1, center=0,
                    square=True, linewidths=0.5,
                    cbar_kws={"shrink": 0.8})
        axes[idx].set_title(country, fontweight="bold")

    # Masquer l'axe vide
    axes[-1].set_visible(False)

    plt.tight_layout()
    path = "outputs/figures/fig3_correlations.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {path}")


# ─────────────────────────────────────────────
# Fig. 4 — Scatter : Taux BCE vs Crédit
# ─────────────────────────────────────────────
def plot_transmission_scatter(df: pd.DataFrame, country_results: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    fig.suptitle("Transmission taux directeur → crédit bancaire\n(régressions MCO par pays)",
                 fontsize=13, fontweight="bold")

    import statsmodels.api as sm

    for idx, (country, color) in enumerate(PALETTE.items()):
        ax = axes[idx]
        df_c = df[df["country"] == country]

        x = df_c["ecb_rate"]
        y = df_c["credit_growth"]

        ax.scatter(x, y, color=color, alpha=0.55, s=28, edgecolors="white", lw=0.5)

        # Droite de régression simple (taux uniquement pour le scatter)
        import statsmodels.formula.api as smf
        df_c2 = df[df["country"] == country]
        res_simple = smf.ols("credit_growth ~ ecb_rate", data=df_c2).fit()
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = res_simple.params["Intercept"] + res_simple.params["ecb_rate"] * x_fit
        ax.plot(x_fit, y_fit, color=color, lw=2.0, ls="--")

        beta = country_results[country].params.get("ecb_rate", np.nan)
        r2   = country_results[country].rsquared
        ax.set_title(f"{country}  (β={beta:+.2f}, R²={r2:.2f})", fontweight="bold", fontsize=10)
        ax.set_xlabel("Taux BCE (%)", fontsize=9)
        ax.set_ylabel("Crédit (%)", fontsize=9)

    axes[-1].set_visible(False)
    plt.tight_layout()
    path = "outputs/figures/fig4_transmission_scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {path}")


# ─────────────────────────────────────────────
# Fig. 5 — Diagnostic résidus MCO
# ─────────────────────────────────────────────
def plot_residuals(country_results: dict) -> None:
    import statsmodels.api as smapi

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    fig.suptitle("Diagnostic des résidus MCO par pays (Q-Q plot)",
                 fontsize=13, fontweight="bold")

    for idx, (country, color) in enumerate(PALETTE.items()):
        res = country_results[country]
        sm.qqplot(res.resid, line="45", ax=axes[idx], alpha=0.6,
                  markerfacecolor=color, markeredgewidth=0.3)
        axes[idx].set_title(country, fontweight="bold", fontsize=10)
        axes[idx].set_xlabel("Quantiles théoriques", fontsize=8)
        axes[idx].set_ylabel("Quantiles observés", fontsize=8)

    axes[-1].set_visible(False)
    plt.tight_layout()
    path = "outputs/figures/fig5_residuals_qqplot.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {path}")


def plot_all(df: pd.DataFrame, country_results: dict) -> None:
    """Lance toutes les visualisations."""
    print("\n📊 Génération des graphiques...")
    plot_time_series(df)
    plot_coefficients(country_results)
    plot_correlation_heatmap(df)
    plot_transmission_scatter(df, country_results)

    try:
        plot_residuals(country_results)
    except Exception as e:
        print(f"  ⚠️  Q-Q plot ignoré : {e}")
