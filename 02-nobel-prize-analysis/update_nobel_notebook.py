import json
from pathlib import Path

p = Path("ass1/belal_nobel_assign2.ipynb")
nb = json.loads(p.read_text(encoding="utf-8"))

code_cells = [
    {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "import pandas as pd",
            "",
            "# Read the Nobel Prize dataset into a DataFrame.",
            "df = pd.read_csv(\"nobel_prize_winners.csv\")",
            "",
            "# Investigate dataset structure with common pandas exploration commands.",
            "df.head()",
            "df.info()",
            "df.describe(include=\"all\")",
            "df[\"gender\"].value_counts(dropna=False)",
        ],
    },
    {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# Create a decade column for the later decade-level groupings.",
            "df[\"decade\"] = (df[\"awardYear\"] // 10) * 10",
            "",
            "# Required exploratory answer objects.",
            "# 1) Most commonly awarded gender and birth country.",
            "top_gender = df[\"gender\"].dropna().mode().iloc[0]",
            "top_country = df[\"birth_country\"].dropna().mode().iloc[0]",
        ],
    },
    {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# 2) Decade with highest ratio of US-born winners to total winners.",
            "usa_winners = df[df[\"birth_country\"].eq(\"USA\")].groupby(\"decade\").size()",
            "total_winners = df.groupby(\"decade\").size()",
            "decade_ratio = (usa_winners / total_winners).fillna(0)",
            "max_decade_usa = int(decade_ratio.idxmax())",
        ],
    },
    {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# 3) Decade/category combination with highest proportion of female laureates.",
            "female_by_pair = df[df[\"gender\"].eq(\"female\")].groupby([\"decade\", \"category\"]).size().rename(\"female_count\")",
            "total_by_pair = df.groupby([\"decade\", \"category\"]).size().rename(\"total_count\")",
            "pair_summary = pd.concat([female_by_pair, total_by_pair], axis=1).fillna(0)",
            "pair_summary[\"female_ratio\"] = pair_summary[\"female_count\"] / pair_summary[\"total_count\"]",
            "max_female_row = pair_summary.reset_index().sort_values(\"female_ratio\", ascending=False).head(1)",
            "max_female_dict = {int(max_female_row.iloc[0][\"decade\"]): max_female_row.iloc[0][\"category\"]}",
        ],
    },
    {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# 4) First woman laureate and category.",
            "first_woman = df[df[\"gender\"].eq(\"female\")].sort_values(\"awardYear\").dropna(subset=[\"fullName\"]).iloc[0]",
            "first_woman_name = first_woman[\"fullName\"]",
            "first_woman_category = first_woman[\"category\"]",
        ],
    },
    {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# 5) Individuals or organizations with >1 Nobel Prize.",
            "repeat_counts = df.groupby(\"fullName\").size().reset_index(name=\"count\")",
            "repeat_list = repeat_counts[repeat_counts[\"count\"] > 1][\"fullName\"].dropna().tolist()",
            "",
            "# Print the requested variables for review.",
            "print(top_gender)",
            "print(top_country)",
            "print(max_decade_usa)",
            "print(max_female_dict)",
            "print(first_woman_name)",
            "print(first_woman_category)",
            "print(repeat_list)",
        ],
    },
]

# Keep the existing first two code cells and replace the empty ones with requested assignment problem solution cells.
nb["cells"] = nb["cells"][:2] + code_cells

p.write_text(json.dumps(nb), encoding="utf-8")
print("updated", len(nb["cells"]), "notebook cells")
