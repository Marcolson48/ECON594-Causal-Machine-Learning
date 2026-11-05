import os
import re
import unicodedata

import pandas as pd

# step 3: a review sheet for the labels outside the top 100. for each label, the AIC
# candidates whose name shares a word with the label and whose dates are close
# (birth + death years within 6 in total), plus same-name records that have no dates.
# writes aic_tail_matches.csv, which was read by hand to choose the ids in step 6

folder = os.path.dirname(os.path.abspath(__file__))
labels = pd.read_csv(os.path.join(folder, "all_labels.csv")).iloc[100:]
candidates = pd.read_csv(os.path.join(folder, "aic_candidates_tail.csv"))
roman = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix"]


# lowercase without accents ("Shōson" -> "shoson")
def fold(s):
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(ch for ch in s if not unicodedata.combining(ch)).lower()


rows = []
for r in labels.itertuples():
    c = candidates[(candidates["label"] == r.label) & candidates["id"].notna()].copy()
    years = [int(y) for y in re.findall(r"\d{4}", str(r.dates))]
    first = re.sub(r"\(.*?\)", "", r.label.split(" and ")[0]).replace("After ", "")
    words = [w for w in re.split(r"[\s,/]+", first.lower()) if w and w not in roman]
    c["name_ok"] = c["title"].map(fold).apply(lambda s: any(re.search(r"\b" + re.escape(w) + r"\b", s) for w in words))
    if len(years) >= 2:
        c["dist"] = (c["birth"] - years[0]).abs() + (c["death"] - years[1]).abs()
    elif len(years) == 1:
        year_col = c["death"] if "died" in str(r.dates) else c["birth"]
        c["dist"] = (year_col - years[0]).abs() * 2
    else:
        c["dist"] = float("nan")
    good = c[c["name_ok"] & (c["dist"] <= 6)]
    undated = c[c["name_ok"] & c["dist"].isna()]
    rows.append((r.label, r.dates, r.lots,
                 "; ".join(f"{int(i)} {t} {b}-{d}" for i, t, b, d in good[["id", "title", "birth", "death"]].itertuples(index=False)),
                 "; ".join(f"{int(i)} {t}" for i, t in undated[["id", "title"]].itertuples(index=False))))

out = pd.DataFrame(rows, columns=["label", "dates", "lots", "date_match", "undated_name_match"])
out.to_csv(os.path.join(folder, "aic_tail_matches.csv"), index=False)
print((out["date_match"] != "").sum() + (out["undated_name_match"] != "").sum(), "candidate matches to review")
