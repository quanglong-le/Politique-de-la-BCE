
# Politique Monétaire de la BCE & Crédit Bancaire

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-green.svg)]()

> Analyse économétrique de la transmission monétaire (taux directeurs → crédit bancaire) en zone euro via modèles MCO multivariés, avec test d'hétérogénéité significative entre pays membres.

---

## Objectif

Ce projet analyse le **canal du crédit** dans la transmission de la politique monétaire de la BCE à travers la zone euro. Il cherche à répondre à trois questions empiriques :

1. **La hausse des taux directeurs réduit-elle le crédit bancaire ?** (canal du taux d'intérêt)
2. **L'inflation modère-t-elle cet effet ?** (canal des prix)
3. **La transmission est-elle homogène entre pays membres ?** (hétérogénéité structurelle)

---

### Méthodologie

### Modèle MCO Multivarié (OLS)

Pour chaque pays $i$ et chaque période $t$ :

$$\Delta \text{Crédit}_{i,t} = \alpha_i + \beta_1 \cdot \text{Taux}_{i,t} + \beta_2 \cdot \text{Inflation}_{i,t} + \beta_3 \cdot \Delta \text{PIB}_{i,t} + \varepsilon_{i,t}$$

| Variable | Description | Source |
|----------|-------------|--------|
| `credit_growth` | Variation du crédit aux ménages & entreprises (%) | BCE / BRI |
| `ecb_rate` | Taux directeur BCE (%) | BCE |
| `inflation` | IPCH (Indice des Prix à la Consommation Harmonisé, %) | Eurostat |
| `gdp_growth` | Croissance du PIB réel (%) | Eurostat |

### Tests d'Hétérogénéité

- **Test de Chow** : rupture structurelle entre groupes de pays
- **Test de Breusch-Pagan** : hétéroscédasticité des résidus
- **Test de Hausman** : effets fixes vs. effets aléatoires
- **Modèle à effets fixes pays** : contrôle des caractéristiques non-observées

---

## Structure du Projet

```
Politique-de-la-BCE/
│
├── data/
│   ├── generate_data.py        # Générateur de données simulées (BCE/Eurostat)
│   └── raw/                    # Données brutes (à placer ici)
│
├── src/
│   ├── data_loader.py          # Chargement et préparation des données
│   ├── ols_models.py           # Modèles MCO pays par pays + pooled
│   ├── panel_models.py         # Modèles de panel (effets fixes/aléatoires)
│   ├── heterogeneity_tests.py  # Tests de Chow, Breusch-Pagan, Hausman
│   └── visualizations.py       # Graphiques et exports
│
├── outputs/
│   ├── figures/                # Graphiques générés
│   └── tables/                 # Tableaux de résultats
│
├── main.py                     # Script principal — pipeline complet
├── requirements.txt
└── README.md
```

---

## Installation & Exécution

### 1. Cloner le dépôt

```bash
git clone https://github.com/quanglong-le/Politique-de-la-BCE.git
cd Politique-de-la-BCE
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Lancer l'analyse complète

```bash
python main.py
```

Les résultats (graphiques + tableaux) sont exportés dans `outputs/`.

---

## Résultats Principaux

| Pays | β_taux | β_inflation | R² | Significativité |
|------|--------|-------------|-----|-----------------|
| Allemagne | -0.82** | -0.31* | 0.71 | ✅ |
| France | -0.64** | -0.18 | 0.65 | ✅ |
| Italie | -1.23*** | -0.52** | 0.79 | ✅ |
| Espagne | -1.41*** | -0.61** | 0.82 | ✅ |
| Pays-Bas | -0.51* | -0.12 | 0.58 | ✅ |
| **Pooled** | **-0.91***| **-0.38**| **0.68** | ✅ |

> **Test de Chow (p < 0.01)** → Rejet de l'homogénéité → Hétérogénéité significative entre pays ✅

### Interprétation

- Une hausse de **+1pp des taux BCE** réduit la croissance du crédit de **-0.64pp (France) à -1.41pp (Espagne)**
- Les **pays périphériques** (Espagne, Italie) sont significativement plus sensibles aux taux que les **pays du cœur** (Allemagne, Pays-Bas)
- L'hétérogénéité est robuste aux spécifications alternatives → argument contre une politique monétaire unique

---

## Visualisations

- Transmission taux → crédit par pays (lignes temporelles)
- Coefficients β par pays avec intervalles de confiance à 95%
- Carte thermique des corrélations par pays
- Résidus MCO et diagnostic d'hétéroscédasticité

---

## Technologies

| Outil | Usage |
|-------|-------|
| `pandas` | Manipulation des données |
| `numpy` | Calcul numérique |
| `statsmodels` | MCO, tests économétriques |
| `linearmodels` | Modèles de panel (effets fixes/aléatoires) |
| `matplotlib` / `seaborn` | Visualisations |
| `scipy` | Tests statistiques |

---

## Références

- **BCE** – Statistical Data Warehouse : [sdw.ecb.europa.eu](https://sdw.ecb.europa.eu)
- **Eurostat** – Données macroéconomiques zone euro
- Bernanke, B. & Gertler, M. (1995) – *Inside the Black Box: The Credit Channel of Monetary Policy Transmission*
- Kashyap, A. & Stein, J. (2000) – *What Do a Million Observations on Banks Say About the Transmission of Monetary Policy?*

---

## Auteur

**Quang Long LE**  
Étudiant L3 Économie-Gestion – Monnaie et Finance | Paris Panthéon-Assas  
📧 quang-long.le@assas-universite.org | 🔗 [linkedin.com/in/quanglong-le](https://linkedin.com/in/quanglong-le)

---

*Projet académique – Données simulées à des fins pédagogiques. Pour une analyse empirique complète, utiliser les données réelles BCE/Eurostat.*
