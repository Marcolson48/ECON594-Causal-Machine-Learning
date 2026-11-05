import os
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, normalize

# other scripts call make_model_data():
#   cv_round = None: final run (all of train -> test)
#   cv_round = 1-4: fit on time blocks 1..r, validate on block r + 1 (used by cv.py)
# n_terms: top lasso terms per text field

# this file sits in submission/, the data in release/
submission_folder = os.path.dirname(os.path.abspath(__file__))
release_folder = os.path.join(submission_folder, "..", "release")


# year or year range ("1840 - 1842"), with "circa" and "s" for year estimates
year_pat = r"(?:circa\s+)?\d{4}s?(?:\s*[^\d,;()]{1,3}?\s*\d{4}s?)?"

# clearance levels, $25 indicator is farther down
clearance_titles = {
    "clearance1": "Drastic Price Reduction Moved to Clearance, Act Fast!",
    "clearance2": "A Clearance Opportunity! Meiji or Edo era Original",
    "clearance3": "Fine Old Reprint Clearance! A Fuji Arts Value",
    "clearance4": "A Clearance Opportunity! Shin-hanga or Modern era Original",
}

# my own stop word list
articles = ["a", "an", "the"]


#################### functions ######################


# collapse repeated spaces and trim the ends
def squish(x):
    return x.str.replace(r"\s+", " ", regex=True).str.strip()


# the print date written in a title or series, e.g. "circa 1840 - 1842":
# the last date in the text from 1650 on (earlier years are depicted events).
# dates in (...) are skipped
def last_date(text):
    if pd.isna(text):
        return np.nan
    text = re.sub(r"\([^)]*\)", "", text)
    dates = re.findall(year_pat, text)
    dates = [d for d in dates if int(re.search(r"\d{4}", d).group()) >= 1650]
    if len(dates) == 0:
        return np.nan
    return dates[-1]


# a date to one number: decade -> midpoint ("1930s" -> 1935),
# range -> midpoint, single year -> that year
def date_to_year(dates):
    first_year = dates.str.extract(r"(\d{4})")[0].astype(float)
    second_year = dates.str.extract(r"\d{4}s?\D+(\d{4})")[0].astype(float)
    is_decade = dates.str.contains(r"^(?:circa\s+)?\d{4}s", na=False)
    year = np.select([is_decade, second_year.notna()],
                     [first_year + 5, (first_year + second_year) / 2],
                     default=first_year)
    return pd.Series(year, index=dates.index)


# drop years, ranges, decades and "circa"
def strip_years(x):
    x = x.str.replace(r"\b(?:circa|ca\.|c\.)\s*", "", regex=True, case=False)
    x = x.str.replace(r"\b\d{4}s?\b(?:\s*[^\w,;()]{1,3}\s*\d{4}s?\b)?", "", regex=True)
    x = x.str.replace(r"\(\W*\)", "", regex=True)
    x = x.str.replace(r"(\s*,)+", ",", regex=True)
    x = x.str.replace(r"[\s,;:-]+$", "", regex=True)
    return squish(x)


# all row-level features, applied to train and test
def prep(df):
    # rows numbered 0, 1, 2, ... so the term columns added later line up by position
    df = df.reset_index(drop=True)
    for col in ["artist_info", "series_info", "link_text", "condition"]:
        df[col] = squish(df[col].str.replace(chr(160), " "))  # chr(160) is a non-breaking space

    # artist dates: first (...) only; midpoint of a range, else the single year.
    # the closing ")" is optional: 10 lots have it missing
    artist_dates = df["artist_info"].str.extract(r"\(([^)]*)\)?")[0]
    df["artist_active"] = artist_dates.str.contains("active", na=False)
    df["artist_circa"] = artist_dates.str.contains(r"circa|c\.", na=False)
    artist_year1 = artist_dates.str.extract(r"(\d{4})")[0].astype(float)
    artist_year2 = artist_dates.str.extract(r"\d{4}\D+(\d{4})")[0].astype(float)
    df["artist_year"] = np.where(artist_year2.isna(), artist_year1, (artist_year1 + artist_year2) / 2)

    # print dates from the title and the series
    # astype(object) keeps .str working even if no row has a date
    title_date = df["link_text"].apply(last_date).astype(object)
    series_date = df["series_info"].apply(last_date).astype(object)
    title_year = date_to_year(title_date)
    series_year = date_to_year(series_date)
    df["title_circa"] = title_date.str.contains("circa", na=False)
    df["series_circa"] = series_date.str.contains("circa", na=False)

    # comparisons with a missing year come out False
    df["posthumous"] = ~df["artist_active"] & (title_year > artist_year2)
    # title year first, the series year only if title year isn't there
    df["made_year"] = np.ceil(title_year.fillna(series_year))
    # origin of made_year as indicators, insight into made_year's accuracy
    df["year_from_title"] = title_year.notna()
    df["year_from_series"] = title_year.isna() & series_year.notna()
    df["year_from_artist"] = title_year.isna() & series_year.isna() & df["artist_year"].notna()

    # is "$25" in the series entry indicator
    df["sale_25"] = df["series_info"].fillna("").str.contains("$25", regex=False)
    for name, title in clearance_titles.items():
        df[name] = df["link_text"] == title

    # missing series is specified for the lasso score so rows aren't dropped
    df["series_present"] = df["series_info"].notna()

    # text for the lasso scores, years removed, clearance statements are replaced with blanks (not nas)
    df["artist_text"] = squish(df["artist_info"].str.replace(r"\s*\([^)]*\d[^)]*\)?", "", regex=True))
    is_clearance = df["link_text"].isin(clearance_titles.values())
    df["title_text"] = strip_years(df["link_text"]).where(~is_clearance, "")
    df["series_text"] = strip_years(df["series_info"])
    return df


# split lowercase text into words and drop articles
def tokenize(text):
    words = re.findall(r"\w+(?:'\w+)*", text)
    return [w for w in words if w not in articles]


# lasso on text tokens, pick the
# largest lambda whose CV error is within one standard error of the min lambda
def fit_lasso_1se(x, y):
    scaler = StandardScaler(with_mean=False)
    x_scaled = scaler.fit_transform(x) # standardize
    cv_fit = LassoCV(cv=KFold(10, shuffle=True, random_state=594), eps=0.01, alphas=50,
                     tol=0.001, n_jobs=-1).fit(x_scaled, y)
    mean_mse = cv_fit.mse_path_.mean(axis=1)
    se_mse = cv_fit.mse_path_.std(axis=1, ddof=1) / np.sqrt(10)
    best = np.argmin(mean_mse)
    alpha_1se = cv_fit.alphas_[mean_mse <= mean_mse[best] + se_mse[best]].max()
    lasso = Lasso(alpha=alpha_1se).fit(x_scaled, y)
    return scaler, lasso


# 0/1 columns: does the text contain each term
def presence(x, top, terms, prefix):
    columns = (x[:, top] > 0).toarray().astype(int)
    names = [prefix + "_" + terms[i].replace(" ", "_") for i in top]
    return pd.DataFrame(columns, columns=names)


# for each text column: lasso score and columns for top terms based on the lasso
# terms kept are the top N in terms of predictive power
def text_lasso(train_text, test_text, y, fold, prefix, n_terms, use_tfidf=True):
    train_text = train_text.fillna("")
    test_text = test_text.fillna("")

    # count every 1-2 word phrase, keep the ones used 5+ times in train
    vectorizer = CountVectorizer(tokenizer=tokenize, token_pattern=None, ngram_range=(1, 2))
    counts = vectorizer.fit_transform(train_text)
    keep = np.asarray(counts.sum(axis=0)).ravel() >= 5
    terms = vectorizer.get_feature_names_out()[keep]
    vectorizer = CountVectorizer(tokenizer=tokenize, token_pattern=None, ngram_range=(1, 2), vocabulary=terms)
    x_train = vectorizer.transform(train_text)
    x_test = vectorizer.transform(test_text)

    if use_tfidf: # use TF-IDF for all except artist column
        tfidf = TfidfTransformer(norm=None)
        x_train = tfidf.fit_transform(normalize(x_train, norm="l1"))
        x_test = tfidf.transform(normalize(x_test, norm="l1"))
    else:
        x_train = (x_train > 0).astype(float)
        x_test = (x_test > 0).astype(float)

    oof = np.zeros(len(y))
    for k in range(1, 6):
        holdout = fold == k
        scaler, lasso = fit_lasso_1se(x_train[~holdout], y[~holdout])
        oof[holdout] = lasso.predict(scaler.transform(x_train[holdout]))

    scaler, lasso = fit_lasso_1se(x_train, y)
    # rank by size, since they were standardized, rare tokens aren't an overshadowing concern
    importance = np.abs(lasso.coef_)
    order = np.argsort(-importance)[:n_terms]
    top = [i for i in order if importance[i] > 0]

    return {
        "train_score": oof,
        "test_score": lasso.predict(scaler.transform(x_test)),
        "train_terms": presence(x_train, top, terms, prefix),
        "test_terms": presence(x_test, top, terms, prefix),
        "ssr_removed": 1 - np.sum((y - oof) ** 2) / np.sum((y - np.mean(y)) ** 2),
        # phrases the lasso kept for each N token value so CV knows how many terms to use
        "kept": pd.Series(lasso.coef_ / scaler.scale_, index=terms)[lasso.coef_ != 0].sort_values(),
    }


# 5 time-ordered blocks for CV
def time_blocks(df):
    order = pd.to_datetime(df["pulled_at"]).rank(method="first")
    return np.ceil(5 * order / len(df)).astype(int).to_numpy()


# put it all into one function to use in the CV script
def make_model_data(cv_round=None, n_terms=25):
    # natural log of final_bid is the outcome
    train_raw = pd.read_csv(os.path.join(release_folder, "train.csv"))
    test_raw = pd.read_csv(os.path.join(release_folder, "test.csv"))

    # in a cv round the held-out block is the test set
    # the below code applies the clean in only the training set
    y_val = None
    if cv_round is not None:
        block = time_blocks(train_raw)
        test_raw = train_raw[block == cv_round + 1].reset_index(drop=True)
        train_raw = train_raw[block <= cv_round].reset_index(drop=True)
        y_val = np.log(test_raw["final_bid"]).to_numpy()

    # ---- features ----
    train = prep(train_raw)
    test = prep(test_raw)

    # lots per artist in train, artists not in train get 0
    artist_counts = train["artist_text"].value_counts()
    train["n_artist_entries"] = train["artist_text"].map(artist_counts)
    test["n_artist_entries"] = test["artist_text"].map(artist_counts).fillna(0)

    # works by the artist held by the Art Institute of Chicago and the Met (made by
    # museum_counts.py, one row per artist label). 0 for placeholder labels like
    # "Meiji era artist (unsigned)"
    museum_counts = pd.read_csv(os.path.join(submission_folder, "museum", "museum_counts.csv"))
    train = train.merge(museum_counts, on="artist_text", how="left", validate="many_to_one")
    test = test.merge(museum_counts, on="artist_text", how="left", validate="many_to_one")

    # -1 marks a missing year (fine for trees, not for linear models)
    for col in ["made_year", "artist_year"]:
        train[col] = train[col].fillna(-1)
        test[col] = test[col].fillna(-1)

    # ---- text scores ----
    y = np.log(train["final_bid"]).to_numpy()
    # 5 time-ordered folds: test lots are all later than train
    fold = time_blocks(train)

    cond = text_lasso(train["condition"], test["condition"], y, fold, "cond_term", n_terms)
    title = text_lasso(train["title_text"], test["title_text"], y, fold, "title_term", n_terms)
    # 0/1 presence for artist: names are short, so tf-idf's length normalization
    # would make "hiroshige" count less in "Hiroshige II" than in "Hiroshige"
    artist = text_lasso(train["artist_text"], test["artist_text"], y, fold, "artist_term", n_terms, use_tfidf=False)
    series = text_lasso(train["series_text"], test["series_text"], y, fold, "series_term", n_terms)

    fields = {"cond": cond, "title": title, "artist": artist, "series": series}
    for name, result in fields.items():
        train[name + "_score"] = result["train_score"]
        test[name + "_score"] = result["test_score"]
    train = pd.concat([train] + [result["train_terms"] for result in fields.values()], axis=1)
    test = pd.concat([test] + [result["test_terms"] for result in fields.values()], axis=1)

    # share of SSR each score removes on its own
    for name, result in fields.items():
        print(name, "score removes", round(result["ssr_removed"], 3), "of SSR")

    # ---- model data ----
    base_vars = [
        "cond_score", "title_score", "artist_score", "series_score",
        "original", "made_year", "year_from_title", "year_from_series", "year_from_artist", "posthumous",
        "artist_year", "artist_circa", "artist_active", "n_artist_entries", "aic_works", "met_works",
        "title_circa", "series_circa", "series_present", "sale_25",
        "clearance1", "clearance2", "clearance3", "clearance4",
    ]
    # term columns per field, best first
    top_cols = {name: list(result["train_terms"].columns) for name, result in fields.items()}
    model_vars = base_vars + top_cols["cond"] + top_cols["title"] + top_cols["artist"] + top_cols["series"]

    # every predictor as a number (True/False -> 1/0)
    model_train = train[model_vars].astype(float)
    model_train.insert(0, "row_id", train["row_id"])
    model_train["log_bid"] = y
    model_test = test[model_vars].astype(float)
    model_test.insert(0, "row_id", test["row_id"])

    return {
        "model_train": model_train,
        "model_test": model_test,
        "y_val": y_val,
        "base_vars": base_vars,
        "top_cols": top_cols,
        "fold": fold,
        "kept": {name: result["kept"] for name, result in fields.items()},
    }


# running this file directly builds the final data and shows its size
if __name__ == "__main__":
    data = make_model_data()
    print(data["model_train"].shape, data["model_test"].shape)
