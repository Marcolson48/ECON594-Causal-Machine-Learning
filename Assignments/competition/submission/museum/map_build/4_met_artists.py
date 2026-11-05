import os
import tempfile
import urllib.request

import pandas as pd

# step 4: Japanese artists in the Met's open access file (github.com/metmuseum/openaccess,
# 341 MB, CC0), with the number of works under each name + birth year + death year.
# an object with several artists lists them separated by "|"; objects whose artist
# lists don't line up are skipped.
# writes met_japanese_artists.csv

folder = os.path.dirname(os.path.abspath(__file__))

# downloaded once to a temporary folder, so it stays out of the project
met_file = os.path.join(tempfile.gettempdir(), "MetObjects.csv")
if not os.path.exists(met_file):
    urllib.request.urlretrieve("https://media.githubusercontent.com/media/metmuseum/openaccess/master/MetObjects.csv", met_file)

cols = ["Artist Display Name", "Artist Nationality", "Artist Begin Date", "Artist End Date"]
met = pd.read_csv(met_file, low_memory=False, dtype=str, usecols=["Object ID"] + cols)
met = met.dropna(subset=["Artist Display Name"])

rows = []
for obj, name, nat, begin, end in met[["Object ID"] + cols].fillna("").itertuples(index=False):
    names, nats, begins, ends = name.split("|"), nat.split("|"), begin.split("|"), end.split("|")
    if not (len(names) == len(nats) == len(begins) == len(ends)):
        continue
    for i in range(len(names)):
        if "Japan" in nats[i]:
            rows.append((obj, names[i].strip(), begins[i].strip(), ends[i].strip()))

japanese = pd.DataFrame(rows, columns=["obj", "name", "begin", "end"])
artists = japanese.groupby(["name", "begin", "end"], as_index=False).agg(works=("obj", "nunique"))
artists = artists.sort_values("works", ascending=False)
artists.to_csv(os.path.join(folder, "met_japanese_artists.csv"), index=False)
print(len(artists), "Japanese name/date combinations")
