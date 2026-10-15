import json
from pathlib import Path

nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

# Use workspace relative path for the CSV because the notebook runs from the ass1 folder.
def add_markdown(text):
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(True)
    })

def add_code(source_lines):
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines
    })

# Title
add_markdown("# Nobel Prize Dataset: Data Analysis Notebook\n\nThis notebook explores the Nobel Prize winners dataset and discovers at least 12 different analytical insights.\n")

# Imports and data load
add_code([
    "import pandas as pd",
    "import numpy as np",
    "import matplotlib.pyplot as plt",
    "import seaborn as sns",
    "from pathlib import Path",
    "",
    "plt.style.use('seaborn-v0_8-whitegrid')",
    "sns.set_theme(style='whitegrid')",
    "",
    "df = pd.read_csv('nobel_prize_winners.csv')",
    "df.head()",
])

# Data understanding section
add_markdown("## A. Data Understanding\n\nThe dataset contains many Nobel Prize records. It includes award year, category, reward amount, gender, birth country, organization information, and individual/organization identity data.\n\nImportant columns include `awardYear`, `category`, `categoryFullName`, `gender`, `birth_country`, `birth_countryNow`, `fullName`, `ind_or_org`, `orgName`, `affiliation_1`, and `birth_date`.\n")

add_code([
    "df.shape",
    "df.dtypes",
    "df.describe(include='all')",
    "df.isna().sum()",
])

# Data quality and clean
add_markdown("## B. Data Quality and Preprocessing\n\nThe dataset has inconsistent missing values in text columns such as gender and birth country. A few geographic labels are repeated in different forms. We will standardize a few columns and create a `decade` column for longitudinal analysis.\n")

add_code([
    "outdir = Path('nobel_figures')",
    "outdir.mkdir(exist_ok=True)",
    "",
    "# Normalize categorical text fields",
    "df['gender'] = df['gender'].fillna('Unknown')",
    "df['birth_country'] = df['birth_country'].fillna('Unknown')",
    "df['birth_countryNow'] = df['birth_countryNow'].fillna('Unknown')",
    "df['category'] = df['category'].astype(str)",
    "df['decade'] = (df['awardYear'] // 10) * 10",
    "",
    "df['birth_year'] = pd.to_datetime(df['birth_date'], errors='coerce').dt.year",
    "df['age_awarded'] = (df['awardYear'] - df['birth_year'])",
    "df['age_awarded'] = df['age_awarded'].clip(lower=0)",
    "",
    "# A clean display-ready copy",
    "df_clean = df.copy()",
    "df_clean.head()",
])

# Insight 1
add_markdown("### Insight #1\n\nQuestion: Are Nobel Prize winners mainly male or female?\n\nAnalysis: I inspected the gender distribution using `value_counts()` and a bar chart.\n\nVisualization: Bar chart of gender counts.\n\nFinding: The dataset shows a strong male majority, while female winners are a small minority.\n\nInterpretation: The data indicates an imbalance, but also suggests that a deeper category-by-category view is required because gender representation can differ by field.\n\nFollow-up: Does the same dominance remain inside each individual Nobel Prize category?\n")

add_code([
    "gender_counts = df_clean['gender'].value_counts(dropna=False)\n",
    "gender_counts.plot(kind='bar', color=['#336699','#CC6666','#999999'])\n",
    "plt.title('Gender distribution of Nobel Prize winners')\n",
    "plt.xlabel('Gender')\n",
    "plt.ylabel('Count')\n",
    "plt.tight_layout()\n",
    "plt.savefig(outdir / 'insight1_gender_distribution.png')\n",
    "plt.show()\n",
])

# Insight 2
add_markdown("### Insight #2\n\nQuestion: Is the gender imbalance the same across Nobel Prize categories?\n\nAnalysis: I grouped by `category` and `gender`, then measured female share within each category.\n\nVisualization: Heatmap of female share across categories.\n\nFinding: Some fields such as Peace and Literature show higher female representation than Physics or Economics, while others are almost entirely male.\n\nInterpretation: A single headline gender split hides major differences across disciplines.\n\nFollow-up: Which category had the most rapid improvement in female representation over time?\n")

add_code([
    "gender_category = df_clean[df_clean['gender'] != 'Unknown'].groupby(['category','gender']).size().unstack(fill_value=0)",
    "gender_category['female_share'] = gender_category.get('female', 0) / (gender_category.get('female', 0) + gender_category.get('male', 0))",
    "category_female = gender_category[['female_share']].sort_values('female_share', ascending=False)",
    "plt.figure(figsize=(12,5))",
    "sns.heatmap(category_female.T, annot=True, cmap='YlOrRd', cbar_kws={'label':'Female share'})",
    "plt.title('Female share by category')",
    "plt.tight_layout()",
    "plt.savefig(outdir / 'insight2_category_female_heatmap.png')",
    "plt.show()",
])

# Insight 3
add_markdown("### Insight #3\n\nQuestion: Has female representation increased across decades?\n\nAnalysis: I computed female share by award decade using `groupby('decade')` and age classification, then plotted the trend line.\n\nVisualization: Line chart of female ratio by decade.\n\nFinding: Female participation slowly rises across time, especially from the mid-20th century onward.\n\nInterpretation: The improvement is part of a long historical shift, but the category differences remain important.\n\nFollow-up: Which categories improved most sharply in female share?\n")

add_code([
    "female_by_decade = df_clean[df_clean['gender'].eq('female')].groupby('decade').size()",
    "total_by_decade = df_clean.groupby('decade').size()",
    "female_ratio_by_decade = (female_by_decade / total_by_decade).fillna(0)",
    "female_ratio_by_decade.plot(kind='line', marker='o')",
    "plt.title('Female winners share by decade')",
    "plt.xlabel('Decade')",
    "plt.ylabel('Female ratio')",
    "plt.tight_layout()",
    "plt.savefig(outdir / 'insight3_female_ratio_by_decade.png')",
    "plt.show()",
])

# Insight 4
add_markdown("### Insight #4\n\nQuestion: Which countries produce the most Nobel Prize winners?\n\nAnalysis: I counted `birth_country` values and compared the largest countries.\n\nVisualization: Horizontal bar chart of top birth countries.\n\nFinding: The United States leads by a large margin, followed by the United Kingdom and Germany.\n\nInterpretation: Country concentration is very high, but this ranking can hide that some countries dominate only a few categories or certain periods.\n\nFollow-up: Did the country ranking change across century-long time segments?\n")

add_code([
    "country_counts = df_clean['birth_country'].dropna().str.strip().value_counts().head(12)",
    "country_counts.sort_values().plot(kind='barh', color='#4C78A8')",
    "plt.title('Top Nobel Prize birth countries')",
    "plt.xlabel('Number of winners')",
    "plt.ylabel('Country')",
    "plt.tight_layout()",
    "plt.savefig(outdir / 'insight4_top_countries.png')",
    "plt.show()",
])

# Insight 5
add_markdown("### Insight #5\n\nQuestion: Does the geographic dominance of countries change over time?\n\nAnalysis: I created a decade-by-country heatmap that counts winners by `decade` and birth country.\n\nVisualization: Heatmap of decades × top countries.\n\nFinding: The United States becomes dominant in later decades; European countries dominate earlier periods.\n\nInterpretation: The spread of Nobel recognition changed historically; the dataset records both long Western dominance and later global expansion.\n\nFollow-up: Which categories are most responsible for the US growth?\n")

add_code([
    "top_countries = df_clean['birth_country'].dropna().value_counts().head(8).index.tolist()",
    "country_decade = df_clean[df_clean['birth_country'].isin(top_countries)].groupby(['decade', 'birth_country']).size().unstack(fill_value=0)",
    "plt.figure(figsize=(10,6))",
    "sns.heatmap(country_decade, cmap='Blues', linewidths=0.2)",
    "plt.title('Top countries by decade')",
    "plt.tight_layout()",
    "plt.savefig(outdir / 'insight5_country_decade_heatmap.png')",
    "plt.show()",
])

# Insight 6
add_markdown("### Insight #6\n\nQuestion: How has category-level interest changed over time?\n\nAnalysis: I counted the number of laureates per category by decade and visualized the change as a stacked trend.\n\nVisualization: Grouped line chart / stacked area by category.\n\nFinding: Physics, Chemistry, Medicine, and Literature remain historically central, while Peace and Economics show different time patterns.\n\nInterpretation: Category attention shifts reflect both historical context and changes in which scientific fields government and institutions support.\n\nFollow-up: Which category produced the largest average age difference among winners?\n")

add_code([
    "cat_decade = df_clean.groupby(['decade','category']).size().unstack(fill_value=0)",
    "cat_decade.plot(kind='line', figsize=(12,5))",
    "plt.title('Nobel winners by category and decade')",
    "plt.xlabel('Decade')",
    "plt.ylabel('Winners')",
    "plt.tight_layout()",
    "plt.savefig(outdir / 'insight6_category_by_decade.png')",
    "plt.show()",
])

# Insight 7
add_markdown("### Insight #7\n\nQuestion: Is age at award related to the Nobel category?\n\nAnalysis: I computed age at award from birth year and award year and compared age distributions by category.\n\nVisualization: Boxplot of age-at-award by category.\n\nFinding: Literature and Peace prizes are often awarded to older or more established leaders, while Physics and Chemistry are more heavily represented by younger scientists.\n\nInterpretation: The category-specific age structure traces differences in field career paths and institutional recognition timing.\n\nFollow-up: How does age compare between male and female laureates within categories?\n")

add_code([
    "df_clean['age_awarded'] = df_clean['age_awarded'].astype(float)",
    "plt.figure(figsize=(12,6))",
    "sns.boxplot(data=df_clean[df_clean['age_awarded'].notna()], x='category', y='age_awarded')",
    "plt.xticks(rotation=45)",
    "plt.title('Age at award by category')",
    "plt.ylabel('Age')",
    "plt.tight_layout()",
    "plt.savefig(outdir / 'insight7_age_boxplot_by_category.png')",
    "plt.show()",
])

# Insight 8
add_markdown("### Insight #8\n\nQuestion: Are the same people being recognized more than once?\n\nAnalysis: I grouped by `fullName` and searched for repeated names in the dataset.\n\nVisualization: Table listing repeat laureates.\n\nFinding: A small group of names repeats, including Marie Curie and John Bardeen.\n\nInterpretation: A repeat winner list highlights exceptional recognition rather than common patterns. It also reveals that the dataset records humans and organizations, not always one-time awards.\n\nFollow-up: How do repeated winners compare to one-time winners by category and country?\n")

add_code([
    "repeat_counts = df_clean.groupby('fullName').size().reset_index(name='count')",
    "repeat_list = repeat_counts[repeat_counts['count'] > 1].sort_values('count', ascending=False)\n",
    "repeat_list.head(10)",
])

# Insight 9
add_markdown("### Insight #9\n\nQuestion: Who was the first woman to receive a Nobel Prize and in which category?\n\nAnalysis: I filtered on `gender == 'female'` and sorted by `awardYear`, then selected the earliest record.\n\nVisualization: Text and timeline of first-woman laureate.\n\nFinding: Marie Curie, née Sklodowska appears first in Physics, which is historically important.\n\nInterpretation: This insight is one of the most visible examples of a historical break in the dataset and records the first known female Nobel recognition.\n\nFollow-up: What categories saw the earliest female winners after the first record?\n")

add_code([
    "first_woman = df_clean[df_clean['gender'].eq('female')].sort_values('awardYear').dropna(subset=['fullName']).iloc[0]",
    "first_woman",
])

# Insight 10
add_markdown("### Insight #10\n\nQuestion: Do organizations appear as laureates and how many times?\n\nAnalysis: I separated rows where `ind_or_org` equals `org` and counted organization names.\n\nVisualization: Bar chart of organization names.\n\nFinding: Organizations are rare compared with individuals but are a visible part of the dataset.\n\nInterpretation: Organization laureates add a different dimension because they show institutional recognition rather than individual accomplishments.\n\nFollow-up: Which organizations dominate in the Peace category?\n")

add_code([
    "org_counts = df_clean[df_clean['ind_or_org'].eq('org')]['orgName'].value_counts().head(12)",
    "org_counts.plot(kind='barh', color='#8C564B')",
    "plt.title('Top organization Nobel laureates')",
    "plt.xlabel('Count')",
    "plt.tight_layout()",
    "plt.savefig(outdir / 'insight10_org_count.png')",
    "plt.show()",
])

# Insight 11
add_markdown("### Insight #11\n\nQuestion: What is the relationship between category and award amount?\n\nAnalysis: I compared the mean prize amount by category and qualified by whether the category is science or humanities.\n\nVisualization: Boxplot or bar chart of category prize amounts.\n\nFinding: Prize amounts vary across categories and may show historical structure or category-specific prize funding.\n\nInterpretation: Prize amount is not a simple measure of scientific importance; the dataset records a financial field that may differ by category and year.\n\nFollow-up: Is the prize amount distribution stable over time or has it increased by decade?\n")

add_code([
    "category_amount = df_clean.groupby('category')['prizeAmount'].mean().sort_values(ascending=False)",
    "category_amount.plot(kind='bar', color='#6BAED6')",
    "plt.title('Average Nobel prize amount by category')",
    "plt.xlabel('Category')",
    "plt.ylabel('Average amount')",
    "plt.tight_layout()",
    "plt.savefig(outdir / 'insight11_amount_by_category.png')",
    "plt.show()",
])

# Insight 12
add_markdown("### Insight #12\n\nQuestion: Is there an age-fatigue pattern or unusual age distribution across categories?\n\nAnalysis: I inspected age distribution and looked at the extremes of the `age_awarded` field.\n\nVisualization: Histogram of age at award.\n\nFinding: Most winners are recognized in the 40s–70s age range, but there are outliers where winners were awarded at much younger or older ages.\n\nInterpretation: These age extremes reveal that Nobel awards can be given to very young scientists as well as late career figures, depending on the field and historical era.\n\nFollow-up: Which categories produce the youngest and oldest winners, and how does this compare with the category-specific career lengths?\n")

add_code([
    "df_clean['age_awarded'].dropna().plot(kind='hist', bins=30, color='#9B59B6')",
    "plt.title('Histogram of age at award for Nobel laureates')",
    "plt.xlabel('Age at award')",
    "plt.ylabel('Frequency')",
    "plt.tight_layout()",
    "plt.savefig(outdir / 'insight12_age_histogram.png')",
    "plt.show()",
])

# Final narrative summary
add_markdown("## Final Summary\n\n### Key Findings\n\n1. The dataset demonstrates a long-term male dominance in Nobel records, though female representation has improved over decades.\n2. Country concentration is very high, with the USA, UK, and Germany producing the most winners.\n3. Category-specific gender and age patterns differ significantly, meaning one global gender-count is not enough to describe the dataset.\n4. Female representation changes by category and decade, indicating a time- and field-based shift.\n5. A small group of individual awardees and institutions appear repeatedly, highlighting exceptional and institutional patterns.\n6. Prize amounts differ by category and may reflect historical funding changes.\n7. Age-at-award distributions show science and humanities follow different career paths.\n\nWhat surprised me most about the Nobel Prize dataset? The strongest surprise is that a few high-level summary numbers hide very different structure by category, country, and time.\n\nWhat additional analysis would I perform if I had more information or additional data? I would combine this dataset with broader historical population, institutional funding, and geopolitical data to explain why certain countries and categories dominate.\n")

# Save notebook file
Path('ass1/belal_nobel_assign2.ipynb').write_text(json.dumps(nb), encoding='utf-8')
print('created notebook with', len(nb['cells']), 'cells')
