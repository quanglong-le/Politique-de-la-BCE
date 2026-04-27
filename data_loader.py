"""
data_loader.py
--------------
Chargement et préparation des données pour l'analyse économétrique.
"""

import os
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.generate_data import generate_all_countries


def load_panel(csv_path: str = "data/panel_zone_euro.csv") -> pd.DataFrame:
    """
    Charge le panel depuis le CSV ou le génère à la volée.
    Retourne un DataFrame propre, prêt pour l'analyse.
    """
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"📂 Données chargées depuis : {csv_path}")
    else:
        print("⚙️  Génération des données simulées...")
        df = generate_all_countries()
        os.makedirs("data", exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"💾 Données sauvegardées : {csv_path}")

    return df


def describe_panel(df: pd.DataFrame) -> None:
    """Affiche les statistiques descriptives du panel."""
    print("\n" + "=" * 60)
    print("STATISTIQUES DESCRIPTIVES DU PANEL")
    print("=" * 60)
    print(f"Pays    : {df['country'].nunique()} ({', '.join(df['country'].unique())})")
    print(f"Périodes: {df['period_str'].nunique()} trimestres")
    print(f"Obs.    : {len(df)} (panel {'' if _is_balanced(df) else 'dés'}équilibré)\n")

    stats = df[["credit_growth", "ecb_rate", "inflation", "gdp_growth"]].describe().round(3)
    stats.index = ["N", "Moyenne", "Écart-type", "Min", "Q25", "Médiane", "Q75", "Max"]
    print(stats.to_string())
    print("=" * 60)


def _is_balanced(df: pd.DataFrame) -> bool:
    counts = df.groupby("country")["period_str"].count()
    return counts.nunique() == 1


def get_country_data(df: pd.DataFrame, country: str) -> pd.DataFrame:
    """Filtre les données pour un pays donné."""
    subset = df[df["country"] == country].copy().reset_index(drop=True)
    if subset.empty:
        raise ValueError(f"Pays inconnu : '{country}'. Disponibles : {df['country'].unique()}")
    return subset


def prepare_for_panel_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prépare les données pour les modèles de panel (linearmodels).
    Crée un MultiIndex (country, period_str).
    """
    df_panel = df.copy()
    df_panel = df_panel.set_index(["country", "period_str"])
    return df_panel
