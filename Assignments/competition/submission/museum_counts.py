import json
import os
import tempfile
import time
import urllib.request

import pandas as pd

# museum holdings per artist: how many works the Art Institute of Chicago (AIC)
# and the Metropolitan Museum of Art (Met) hold by each artist.
# every artist label in train and test was matched to the museums' artist records
# by hand in museum/artist_museum_map.csv, checking names and dates (see museum/map_build).
# II and III are separate people with separate records (Hiroshige II is not Hiroshige)

museum_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "museum")
artist_map = pd.read_csv(os.path.join(museum_folder, "artist_museum_map.csv"))


# ---- AIC: public API (api.artic.edu), one request per artist id ----

# the API asks scripts to say who they are
headers = {"AIC-User-Agent": "ECON594-class-project", "User-Agent": "ECON594-class-project"}


# number of artworks credited to one AIC artist id
def aic_works(aic_id):
    time.sleep(3)  # one request every 3 seconds; faster runs hit the API's rate limit
    url = "https://api.artic.edu/api/v1/artworks/search?query%5Bterm%5D%5Bartist_id%5D=" + str(aic_id) + "&limit=0"
    response = urllib.request.urlopen(urllib.request.Request(url, headers=headers))
    return json.load(response)["pagination"]["total"]


aic_ids = pd.concat([artist_map["aic_id_1"], artist_map["aic_id_2"]]).dropna().astype(int).unique()
print(len(aic_ids), "AIC artists, about", round(len(aic_ids) * 3 / 60), "minutes")
aic = {}
for aic_id in aic_ids:
    aic[aic_id] = aic_works(aic_id)


# ---- Met: open access file (github.com/metmuseum/openaccess, 341 MB, CC0) ----

# downloaded to a temporary folder, so it stays out of the project
met_file = os.path.join(tempfile.gettempdir(), "MetObjects.csv")
if not os.path.exists(met_file):
    urllib.request.urlretrieve("https://media.githubusercontent.com/media/metmuseum/openaccess/master/MetObjects.csv", met_file)

# one row per object and artist; several artists on one object are separated by "|".
# an artist is "display name | birth year", e.g. "Utagawa Hiroshige | 1797"
met = pd.read_csv(met_file, usecols=["Object ID", "Artist Display Name", "Artist Begin Date"], dtype=str)
met = met.dropna(subset=["Artist Display Name"])
met["name"] = met["Artist Display Name"].str.split("|")
met["begin"] = met["Artist Begin Date"].fillna("").str.split("|")
met = met[met["name"].str.len() == met["begin"].str.len()]
met = met.explode(["name", "begin"])
met["met_key"] = met["name"].str.strip() + " | " + met["begin"].str.strip()
met = met.drop_duplicates(["Object ID", "met_key"])
met_works = met["met_key"].value_counts()


# a label's Met count: works under any of its keys (the same artist is sometimes
# written two ways, e.g. with and without kanji). set() drops repeated keys so
# no works are counted twice
def met_count(keys):
    if pd.isna(keys):
        return 0
    unique_keys = list(set(keys.split("; ")))
    return int(met_works.reindex(unique_keys).fillna(0).sum())


# AIC count for one id; labels without an id get 0
def aic_count(aic_id):
    if pd.isna(aic_id):
        return 0
    return aic[int(aic_id)]


# ---- one count per label ----

# collaborations ("Hiroshige and Toyokuni III/Kunisada") get the larger of the two artists' counts.
# artists the museum doesn't hold get 0, and so do placeholder labels ("Meiji era artist (unsigned)"),
# which have no named artist to look up
museum_counts = pd.DataFrame({"artist_text": artist_map["artist_text"]})
museum_counts["aic_works"] = [max(aic_count(a), aic_count(b)) for a, b in zip(artist_map["aic_id_1"], artist_map["aic_id_2"])]
museum_counts["met_works"] = [max(met_count(a), met_count(b)) for a, b in zip(artist_map["met_keys_1"], artist_map["met_keys_2"])]

museum_counts.to_csv(os.path.join(museum_folder, "museum_counts.csv"), index=False)
print("wrote counts for", len(museum_counts), "artist labels")
