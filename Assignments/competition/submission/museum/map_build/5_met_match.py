import os
import re
import unicodedata

import pandas as pd

# step 5: first-pass match of every label to Met artists: a Met name that shares a
# word with the label's first artist, with dates close to the label's (birth + death
# years within 6 in total). these were then checked by hand in step 6.
# writes met_matches.csv

folder = os.path.dirname(os.path.abspath(__file__))
roman = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix"]


# romanized name without accents or kanji ("Utagawa Kunisada 歌川国貞" -> "Utagawa Kunisada")
def fold(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^\x00-\x7f]", "", s)
    return " ".join(s.split())


met = pd.read_csv(os.path.join(folder, "met_japanese_artists.csv"), dtype={"begin": str, "end": str})
met["name"] = met["name"].map(fold)
met["b"] = pd.to_numeric(met["begin"], errors="coerce")
met["e"] = pd.to_numeric(met["end"], errors="coerce")
met = met.groupby(["name", "b", "e"], as_index=False, dropna=False)["works"].sum()
met["low"] = met["name"].str.lower().str.replace(r"[()',.]", " ", regex=True)

labels = pd.read_csv(os.path.join(folder, "all_labels.csv"))
out = []
for r in labels.itertuples():
    if "artist (" in r.label:
        out.append({"label": r.label, "lots": r.lots, "dates": r.dates, "met_names": "", "met_works": None, "status": "placeholder"})
        continue
    years = [int(y) for y in re.findall(r"\d{4}", str(r.dates))]
    # the label's dates belong to its first artist
    first = re.sub(r"\(.*?\)", "", r.label.split(" and ")[0]).replace("After ", "")
    words = [w for w in re.split(r"[\s,/]+", first.lower()) if w and w not in roman]
    cand = met[met["low"].apply(lambda s: any(re.search(r"\b" + re.escape(w) + r"\b", s) for w in words))].copy()
    if len(years) >= 2:
        cand["dist"] = (cand["b"] - years[0]).abs() + (cand["e"] - years[1]).abs()
    elif len(years) == 1:
        year_col = cand["e"] if "died" in str(r.dates) else cand["b"]
        cand["dist"] = (year_col - years[0]).abs() * 2
    else:
        cand["dist"] = 99
    best = cand[cand["dist"] <= 6]
    names = " | ".join(f"{n} ({'' if pd.isna(b) else int(b)}-{'' if pd.isna(e) else int(e)}: {w})"
                       for n, b, e, w in best[["name", "b", "e", "works"]].itertuples(index=False))
    out.append({"label": r.label, "lots": r.lots, "dates": r.dates, "met_names": names,
                "met_works": best["works"].sum() if len(best) else 0, "status": "match" if len(best) else "none"})

out = pd.DataFrame(out)
out.to_csv(os.path.join(folder, "met_matches.csv"), index=False)
print(out["status"].value_counts())
