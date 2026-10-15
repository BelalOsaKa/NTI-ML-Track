import pandas as pd

# Read dataset
# In notebook the file sits beside the notebook, so the same relative path works there.
df = pd.read_csv("ass1/nobel_prize_winners.csv")

df["decade"] = (df["awardYear"] // 10) * 10

top_gender = df["gender"].dropna().mode().iloc[0]
top_country = df["birth_country"].dropna().mode().iloc[0]

usa_winners = df[df["birth_country"].eq("USA")].groupby("decade").size()
total_winners = df.groupby("decade").size()
decade_ratio = (usa_winners / total_winners).fillna(0)
max_decade_usa = int(decade_ratio.idxmax())

female_by_pair = df[df["gender"].eq("female")].groupby(["decade", "category"]).size().rename("female_count")
total_by_pair = df.groupby(["decade", "category"]).size().rename("total_count")
pair_summary = pd.concat([female_by_pair, total_by_pair], axis=1).fillna(0)
pair_summary["female_ratio"] = pair_summary["female_count"] / pair_summary["total_count"]
max_female_row = pair_summary.reset_index().sort_values("female_ratio", ascending=False).head(1)
max_female_dict = {int(max_female_row.iloc[0]["decade"]): max_female_row.iloc[0]["category"]}

first_woman = df[df["gender"].eq("female")].sort_values("awardYear").dropna(subset=["fullName"]).iloc[0]
first_woman_name = first_woman["fullName"]
first_woman_category = first_woman["category"]

repeat_counts = df.groupby("fullName").size().reset_index(name="count")
repeat_list = repeat_counts[repeat_counts["count"] > 1]["fullName"].dropna().tolist()

print("top_gender =", top_gender)
print("top_country =", top_country)
print("max_decade_usa =", max_decade_usa)
print("max_female_dict =", max_female_dict)
print("first_woman_name =", first_woman_name)
print("first_woman_category =", first_woman_category)
print("repeat_list =", repeat_list)
