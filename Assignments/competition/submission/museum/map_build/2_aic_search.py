import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

# step 2: search the Art Institute of Chicago's public API (api.artic.edu) for
# artist records matching each label's name. these are only candidates: the
# matches were then checked by hand on names and dates (step 6).
# writes aic_candidates.csv (top 100 labels, 12 results each) and
# aic_candidates_tail.csv (the other labels, 8 results each).
# about 20 minutes: one request every 2 seconds keeps under the API's rate limit.
# the API can change over time, so a re-run may return slightly different candidates
# than the saved files

folder = os.path.dirname(os.path.abspath(__file__))
labels = pd.read_csv(os.path.join(folder, "all_labels.csv"))
headers = {"AIC-User-Agent": "ECON594-class-project", "User-Agent": "ECON594-class-project"}


# one API request; waits and retries if rate limited (HTTP 429)
def get(url):
    for wait in [0, 30, 90]:
        time.sleep(wait)
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=headers)))
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
    raise RuntimeError("rate limited")


# names to search for one label: one per artist, without "After", generation
# numbers (II, III) or the first of two art names ("Toyokuni III/Kunisada" -> "Kunisada").
# placeholder labels ("Meiji era artist (unsigned)") have no name to search
def search_names(label):
    if "artist (" in label or ("artist" in label.lower() and "unsigned" in label.lower()):
        return []
    names = []
    for part in label.split(" and "):
        part = re.sub(r"\(.*?\)", "", part).replace("After ", "")
        part = part.split("/")[-1]
        part = re.sub(r"\b(I|II|III|IV|V|VI|VII|VIII|IX)\b", "", part)
        name = " ".join(p for p in re.split(r"[,\s]+", part) if p)
        if name:
            names.append(name)
    return names


def search(label_list, limit):
    rows = []
    for label in label_list:
        for name in search_names(label):
            url = "https://api.artic.edu/api/v1/agents/search?" + urllib.parse.urlencode(
                {"q": name, "fields": "id,title,birth_date,death_date", "limit": limit})
            results = get(url)["data"]
            if len(results) == 0:
                rows.append({"label": label, "query": name})
            for a in results:
                rows.append({"label": label, "query": name, "id": a["id"], "title": a["title"],
                             "birth": a.get("birth_date"), "death": a.get("death_date")})
            time.sleep(2)
    return pd.DataFrame(rows)


search(labels["label"][:100], 12).to_csv(os.path.join(folder, "aic_candidates.csv"), index=False)
search(labels["label"][100:], 8).to_csv(os.path.join(folder, "aic_candidates_tail.csv"), index=False)
