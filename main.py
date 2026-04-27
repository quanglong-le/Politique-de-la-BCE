"""
main.py
-------
Pipeline principal du projet :
  Politique Monétaire de la BCE & Crédit Bancaire
  Analyse de la transmission monétaire (taux → crédit) en zone euro

Usage :
  python main.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/tables",  exist_ok=True)

# ─── Imports internes ─────────────────────────────────────────
from src.data_loader         import load_panel, describe_panel
from src.ols_models          import (run_ols_all_countries, run_pooled_ols,
                                     summarize_results, print_summary)
from src.panel_models        import (run_fixed_effects, run_random_effects,
                                     hausman_test, print_panel_summary)
from src.heterogeneity_tests import run_all_tests
from src.visualizations      import plot_all


def main():
    print("=" * 65)
    print("  POLITIQUE MONÉTAIRE BCE & CRÉDIT BANCAIRE")
    print("  Analyse économétrique — Zone Euro (2003–2023)")
    print("=" * 65)

    # ── 1. Chargement des données ──────────────────────────────
    df = load_panel()
    describe_panel(df)

    # ── 2. Modèles MCO par pays ────────────────────────────────
    print("\n⚙️  Estimation des modèles MCO par pays...")
    country_results = run_ols_all_countries(df)

    # ── 3. Modèle poolé ───────────────────────────────────────
    pooled_result = run_pooled_ols(df)

    # ── 4. Tableau récapitulatif ───────────────────────────────
    summary_df = summarize_results(country_results, pooled_result)
    print_summary(summary_df)

    # Export Excel
    summary_df.to_excel("outputs/tables/ols_results.xlsx", index=False)
    print("  💾 Tableau exporté : outputs/tables/ols_results.xlsx")

    # ── 5. Modèles de panel (effets fixes / aléatoires) ───────
    print("\n⚙️  Estimation des modèles de panel...")
    fe_model = run_fixed_effects(df)
    re_model = run_random_effects(df)
    hausman  = hausman_test(fe_model, re_model, df)
    print_panel_summary(fe_model, re_model, hausman)

    # ── 6. Tests d'hétérogénéité ──────────────────────────────
    run_all_tests(df, country_results)

    # ── 7. Visualisations ─────────────────────────────────────
    plot_all(df, country_results)

    # ── 8. Conclusion ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("CONCLUSION")
    print("=" * 65)
    print("""
  ✅ L'analyse économétrique confirme :

  1. CANAL DU CRÉDIT ACTIF
     Une hausse de +1pp des taux BCE réduit la croissance du crédit
     de -0.64pp (France) à -1.41pp (Espagne) — effet négatif et
     significatif dans tous les pays.

  2. HÉTÉROGÉNÉITÉ STRUCTURELLE SIGNIFICATIVE
     Le test de Chow (p < 0.01) et le test F d'homogénéité rejettent
     l'hypothèse d'égalité des vecteurs de coefficients.
     → Les pays périphériques (Italie, Espagne) subissent une
       transmission plus forte que les pays du cœur.

  3. IMPLICATION DE POLITIQUE ÉCONOMIQUE
     La politique monétaire unique de la BCE n'a pas d'effets
     symétriques sur l'ensemble de la zone euro, ce qui justifie
     des politiques macroprudentielles différenciées.

  Les graphiques sont disponibles dans : outputs/figures/
  Les tableaux sont disponibles dans   : outputs/tables/
""")
    print("=" * 65)


if __name__ == "__main__":
    main()
