import os
import re
import unicodedata

import pandas as pd

# step 6: the hand-checked crosswalk from every artist label to the museums' artist records.
# every match below was checked by hand against the candidates from steps 2-5, on names
# and dates. II and III are separate people with separate records (Hiroshige II is not
# Hiroshige). writes ../artist_museum_map.csv, which museum_counts.py reads.
#
# run order: 1_artist_labels.py, 2_aic_search.py, 3_aic_tail_matches.py (review sheet),
# 4_met_artists.py, 5_met_match.py, then this file. steps 2-5 have saved outputs in this
# folder, so this file alone rebuilds the map from them

folder = os.path.dirname(os.path.abspath(__file__))
labels = pd.read_csv(os.path.join(folder, "all_labels.csv"))
top100 = set(labels["label"][:100])


# romanized name without accents or kanji
def fold(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return " ".join(re.sub(r"[^\x00-\x7f]", "", s).split())


# ---- AIC: artist ids per label ----

# AIC's name for each id, from the search results
candidates = pd.concat([pd.read_csv(os.path.join(folder, "aic_candidates.csv")),
                        pd.read_csv(os.path.join(folder, "aic_candidates_tail.csv"))]).dropna(subset=["id"])
aic_name = dict(zip(candidates["id"].astype(int), candidates["title"]))
aic_name[33198] = "Utagawa Yoshiiku"
aic_name[35220] = "Kawase Hasui"

# top 100 labels (90% of lots). top 100 labels missing here have no AIC record
top_aic = {
    "Toyokuni III/Kunisada": [61302], "Chikanobu": [103264], "Kunichika": [31767], "Kuniyoshi": [14488],
    "Hiroshige": [34946], "Imao Keinen": [99061], "Kunisada II": [35352], "Yoshitoshi": [37401],
    "Kiyochika": [35262], "Tsukioka Kogyo": [26646], "Yoshitora": [37400], "Toshikata": [26395],
    "Hokusai": [31492], "Tokuriki": [112858], "Toyokuni III/Kunisada and Hiroshige II": [61302, 34950],
    "Kyosai": [35361], "Kunisada III": [12042], "Hiroshige III": [66945], "Yoshiiku": [33198],
    "Ginko": [9739], "Gekko": [19572], "Ito Shinsui": [36653], "Eisen": [34380], "Utamaro": [37074],
    "Yoshikazu": [37396], "Munakata": [35935], "Junichiro Sekino": [35168], "Hirosada": [11629],
    "Kuniteru II": [14480], "Hiroshige II": [34950], "Toyokuni I": [26428], "Kokunimasa": [103267],
    "Nobukazu": [113693], "Endo Kyozo": [14549], "Yoshisuke Funasaka": [101566], "Toshi Yoshida": [36997],
    "Kuniyasu": [14483], "Yoshitsuya": [60944], "Hajime Namiki": [104124], "Tsuchiya Koitsu": [14227],
    "Bakufu Ohno": [112859], "Eizan": [34385], "Toshihide": [113691], "Yoshifuji": [104265],
    "Fusatane": [106030], "Kiyoshi Saito": [14032], "Yoshimori": [60941], "Sadahide": [22718],
    "Ishikawa Toraji": [26364], "Hiroshige and Toyokuni III/Kunisada": [34946, 61302], "Kunitoshi": [42734],
    "Sharaku": [36624], "Harunobu": [36845], "Fumio Kitaoka": [35261], "Toyokuni II": [37006],
    "Kunitsuna II": [14480], "Hashiguchi Goyo": [34861], "Torii Kotondo": [36995],
    "Shotei (Takahashi Hiroaki)": [34943],
}

# judgment calls on the top 100, kept in the map's note column
top_notes = {
    "Toyokuni III/Kunisada": "AIC files him as Kunisada I (Toyokuni III)",
    "Kunitsuna II": "AIC: Kuniteru II (Kunitsuna II), same person as Kuniteru II",
    "Toyonobu": "AIC 'Utagawa Toyonobu' record holds an 18th c. Ishikawa Toyonobu work, not this artist",
    "Sadanobu III": "AIC has only Hasegawa Sadanobu I (1809-1879)",
    "Sadanobu II": "AIC has only Hasegawa Sadanobu I (1809-1879)",
    "Tomioka Eisen": "different artist from Keisai Eisen",
    "Miyagawa Shuntei": "different artist from Katsukawa Shuntei",
    "Masao Ido": "different artist from Maeda Masao",
    "Torii Kiyotada VII": "AIC has only Torii Kiyotada I",
    "Torii Kiyotada VII and Torii Kiyosada": "AIC has only Torii Kiyotada I",
    "Kyosai": "a second AIC record (13660) holds 1 work, left out",
    "Nobukazu": "AIC dates 1872-1944",
}

# labels outside the top 100, chosen from the review sheet (step 3)
tail_aic = {
    "Hasui": [35220], "Tsunetomi Kitano": [26653], "Koichi Maeda": [104213], "Noda Kyuho": [112855],
    "Yamamura Toyonari": [81691], "Kawakami Sumio": [36835], "Shiro Kasamatsu": [36657], "Shinsai": [22656],
    "Ohara Shoson": [36665], "Kiyokata Kaburagi": [31670], "Kamisaka Sekka": [42579], "Goro Kumagai": [14431],
    "Aoyama Masaharu Inoue": [112851], "Yoshiharu": [114265], "Mitsuhiro Unno": [32986], "Zeshin": [37416],
    "Maekawa Senpan": [36605], "Eishi": [52271], "Kaoru Kawano": [13584], "Kunihisa II": [35347],
    "Bertha Lum": [15726], "Sadanobu": [22731], "Gyokusho Kawabata": [10540], "Kiyonaga": [35265],
    "Uemura Shoen": [32970], "Yamakawa Shuho": [91529], "Yoshitoshi Mori": [17707], "Yoshimune": [12520],
    "Nakao Yoshitaka": [18963], "Aoyama Masaharu": [112851], "Shunsho": [36681], "Tokuriki Tomikichiro": [112858],
    "Koryusai": [35327], "Toyohiro": [37003], "Eiichi Kotozuka": [112853], "Clifton Karhu": [101568],
    "Sadanobu I": [22731], "Shigemasa": [23776], "Gakutei": [9879], "Watanabe Seitei": [36666],
    "Takeuchi Seiho": [23459], "Shunman": [23940], "Kiyomine": [35272], "Tsuchida Bakusen": [59323],
    "After Maruyama Okyo": [35633], "Mitsuoki Tosa": [26388], "Kiyonobu": [35275], "Toyoharu": [37002],
    "Shigenobu I": [22657], "Kunimaru": [14451], "Masanobu, Okumura": [61758], "Ikeda Terukata": [91593],
    "Ide Gakusui": [111128], "Moromasa": [35837], "Masanobu": [61760], "Kunimasa": [35349],
    "Ikeda Shoen": [91595], "Tosa Mitsuoki": [26388], "Koizumi Kishio": [14229], "Toyokuni III/Kunisada I": [61302],
    "Kogyo, Tsukioka": [26646], "Kiyomitsu": [26368], "Shunsen": [23981],
    # AIC records without dates, same artist by name and period
    "Shiro Takagi": [25428], "Azechi Umetaro": [33488], "Koho Shoda": [119363], "Kunisato": [87505],
    "Yoshikata": [29337], "Choki": [4737], "Ashiyuki": [1161], "Mishima Shoso": [17448], "Shoso Mishima": [17448],
    "Seiko Kawachi": [116802], "Shuncho": [36670], "Kaigetsudo Dohan": [54717], "Ikuhide": [104170],
    "Eisho": [52275], "Shunko": [115103], "Hokuei": [11815], "Shucho": [36669], "Hokuju": [34972],
    "Shunzan": [24033], "Toyomasa": [12519], "Eisui": [34384], "Kiyohiro": [35264], "Kiyotsune": [35281],
    "Kiyotomo": [35279], "Kiyoshige": [35278], "Tadashi Nakayama": [35953],
    # collaborations: both artists
    "Toyokuni III/Kunisada and Hiroshige": [61302, 34946], "Toyokuni III/Kunisada and Kunisada II": [61302, 35352],
    "Kyosai and Hiroshige III": [35361, 66945], "Kunichika and Ikkei": [31767],
    "Hiroshige II and Toyokuni III/Kunisada": [34950, 61302], "Chikanobu and Kunichika": [103264, 31767],
    "Yoshitoshi and Hiroshige III": [37401, 66945],
}


# ---- Met: exact "display name | birth year" keys per label ----

met_raw = pd.read_csv(os.path.join(folder, "met_japanese_artists.csv"), dtype=str)
met_raw["f"] = met_raw["name"].map(fold)
met_matches = pd.read_csv(os.path.join(folder, "met_matches.csv"))


# every Met spelling (with or without kanji) of the given (romanized name, birth year) pairs
def keys_for(name_year_pairs):
    keys = []
    for name, year in name_year_pairs:
        hit = met_raw[(met_raw["f"] == name) & (met_raw["begin"].str.strip() == str(year))]
        keys += [f"{n} | {b.strip()}" for n, b in hit[["name", "begin"]].itertuples(index=False)]
    return keys


# step 5's matches as (name, birth year) pairs
def parse(met_names):
    if pd.isna(met_names) or met_names == "":
        return []
    pairs = []
    for part in met_names.split(" | "):
        m = re.match(r"(.*) \((\d*)-(\d*): \d+\)$", part.strip())
        if m:
            pairs.append((m.group(1), m.group(2)))
    return pairs


met_pairs = {r.label: parse(r.met_names) for r in met_matches.itertuples()}

# false matches from step 5: a shared word and close dates, but a different person
for label in ["Gijin Okuyama", "Hiyoshi Mamoru", "Takahashi Biho", "Yamaguchi Kayo", "Kayo Yamaguchi",
              "Tadamasa Ueno", "Okada Koichi", "Shusei"]:
    met_pairs[label] = []
met_pairs["Takeuchi Keishu"] = [p for p in met_pairs["Takeuchi Keishu"] if p[0] != "Takeuchi Seiho"]
met_pairs["Takeuchi Seiho"] = [p for p in met_pairs["Takeuchi Seiho"] if p[0] != "Takeuchi Keishu"]
met_pairs["Toyomasa"] = [p for p in met_pairs["Toyomasa"] if p[0] != "Toyomasa"]

# matches step 5 missed (the Met gives different dates than Fuji Arts)
met_pairs.update({
    "Yoshitora": [("Utagawa Yoshitora", "1845")], "Ginko": [("Adachi Ginko", "1853")],
    "Chikashige": [("Morikawa Chikashige", "1850")], "Fusatane": [("Utagawa Fusatane", "1849")],
    "Hirosada": [("Gosotei Hirosada", "1819"), ("Utagawa Hirosada", "1825")],
    "Shinsai": [("Ryuryukyo Shinsai", "1799")], "Choki": [("Eishosai Choki", "1700")],
    "Toshimine": [("Tsutsui Toshimine", "1863")], "Kunitsuna II": met_pairs["Kuniteru II"],
})

# second artist of a collaboration -> the label holding that artist's own keys
second_artist = {
    "Toyokuni III/Kunisada and Hiroshige II": "Hiroshige II", "Hiroshige and Toyokuni III/Kunisada": "Toyokuni III/Kunisada",
    "Toyokuni III/Kunisada and Hiroshige": "Hiroshige", "Toyokuni III/Kunisada and Kunisada II": "Kunisada II",
    "Kyosai and Hiroshige III": "Hiroshige III", "Kunichika and Ikkei": "Ikkei",
    "Hiroshige II and Toyokuni III/Kunisada": "Toyokuni III/Kunisada", "Chikanobu and Kunichika": "Kunichika",
    "Yoshitoshi and Hiroshige III": "Hiroshige III",
}


# "Meiji era artist (unsigned)" and the like: no named artist
def is_placeholder(label):
    low = label.lower()
    return "artist" in low and ("unsigned" in low or "not read" in low)


# ---- one row per label ----

rows = []
for r in labels.itertuples():
    label = r.label
    if label in top100:
        ids = top_aic.get(label, [])
        if "artist (" in label:
            note = "placeholder label, no artist named"
        elif len(ids) == 0:
            note = top_notes.get(label, "no AIC artist record")
        else:
            note = top_notes.get(label, "")
    else:
        ids = tail_aic.get(label, [])
        note = ""
    keys_1 = keys_for(met_pairs.get(label, []))
    keys_2 = keys_for(met_pairs.get(second_artist[label], [])) if label in second_artist else []
    rows.append({"artist_text": label, "fuji_dates": r.dates, "lots": r.lots, "placeholder": is_placeholder(label),
                 "aic_id_1": ids[0] if len(ids) > 0 else None,
                 "aic_name_1": aic_name.get(ids[0]) if len(ids) > 0 else None,
                 "aic_id_2": ids[1] if len(ids) > 1 else None,
                 "aic_name_2": aic_name.get(ids[1]) if len(ids) > 1 else None,
                 "met_keys_1": "; ".join(keys_1), "met_keys_2": "; ".join(keys_2), "note": note})

artist_map = pd.DataFrame(rows)
artist_map["aic_id_1"] = artist_map["aic_id_1"].astype("Int64")
artist_map["aic_id_2"] = artist_map["aic_id_2"].astype("Int64")
artist_map.to_csv(os.path.join(folder, "..", "artist_museum_map.csv"), index=False, encoding="utf-8")
print(len(artist_map), "labels |", artist_map["placeholder"].sum(), "placeholders |",
      artist_map["aic_id_1"].notna().sum(), "with an AIC record |", (artist_map["met_keys_1"] != "").sum(), "with Met keys")
