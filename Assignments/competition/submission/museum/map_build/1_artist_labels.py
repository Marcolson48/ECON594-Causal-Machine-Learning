import os

import pandas as pd

# step 1: every artist label in train and test, with its lot count and most common dates.
# labels use the same rule as clean.R: artist_info with any (...) holding a year removed
# writes all_labels.csv (most lots first; the first 100 rows are the "top 100")

folder = os.path.dirname(os.path.abspath(__file__))
release_folder = os.path.join(folder, "..", "..", "..", "release")

train = pd.read_csv(os.path.join(release_folder, "train.csv"))
test = pd.read_csv(os.path.join(release_folder, "test.csv"))
lots = pd.concat([train, test])

info = lots["artist_info"].str.replace(chr(160), " ").str.split().str.join(" ")  # chr(160): non-breaking space
lots["label"] = info.str.replace(r"\s*\([^)]*\d[^)]*\)?", "", regex=True).str.split().str.join(" ")
lots["dates"] = info.str.extract(r"\(([^)]*)\)?")[0]


def most_common(dates):
    if dates.notna().any():
        return dates.mode().iloc[0]
    return ""


labels = lots.groupby("label").agg(lots=("label", "size"), dates=("dates", most_common))
labels = labels.sort_values("lots", ascending=False)
labels.to_csv(os.path.join(folder, "all_labels.csv"))
print(len(labels), "labels")
