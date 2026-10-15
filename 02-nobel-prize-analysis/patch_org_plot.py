import json
from pathlib import Path

p = Path('ass1/belal_nobel_assign2.ipynb')
nb = json.loads(p.read_text(encoding='utf-8'))

for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        source = ''.join(cell.get('source', []))
        if "org_counts = df_clean[df_clean['ind_or_org'].eq('org')]['orgName'].value_counts().head(12)" in source:
            cell['source'] = [
                "org_counts = df_clean[df_clean['ind_or_org'].eq('org')]['orgName'].value_counts().head(12)",
                "",
                "org_counts.plot(kind='barh', color='#8C564B')",
                "",
                "plt.title('Top organization Nobel laureates')",
                "plt.xlabel('Count')",
                "plt.ylabel('Organization')",
                "plt.tight_layout()",
                "plt.savefig(outdir / 'insight10_org_count.png')",
                "plt.show()",
            ]
            break

p.write_text(json.dumps(nb), encoding='utf-8')
print('updated organization plot cell with y-axis label')
