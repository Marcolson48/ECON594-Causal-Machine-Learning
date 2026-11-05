import os

import numpy as np
import pandas as pd
from clean import make_model_data, submission_folder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import PredefinedSplit
from sklearn.preprocessing import StandardScaler

# tune the dataset and forest hyperparameters together
# each round reruns make_model_data()
cv_rounds = [1, 2, 3, 4]

# text settings for condition, title and series
n_values = [25, 50, 100]
n_grid = [{"text": "scores only", "n_terms": 0}]
for n in n_values:
    n_grid.append({"text": "scores + terms", "n_terms": n})
    n_grid.append({"text": "terms only", "n_terms": n})

score_vars = ["cond_score", "title_score", "series_score"]

# RF tuned within every text setting
# set features and min node size values to iterate over
max_feat = [5, 10, 15, 20, 30, 40, 50, 60]
min_samples = [3, 5, 7, 10]


# share of SSR removed formula
def ssr_share(pred, y):
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - np.mean(y)) ** 2)


results = []
for r in cv_rounds:
    print("round", r)
    # build the largest term set once, use the first N per setting
    data = make_model_data(cv_round=r, n_terms=max(n_values))
    model_train = data["model_train"]
    model_test = data["model_test"]
    y_train = model_train["log_bid"]
    y_val = data["y_val"]
    base_vars = data["base_vars"]
    top_cols = data["top_cols"]

    # for non-RF models -1 fills replaced with median
    lin_train = model_train.copy()
    lin_test = model_test.copy()
    for col in ["made_year", "artist_year"]:
        median = model_train.loc[model_train[col] != -1, col].median()
        lin_train.loc[lin_train[col] == -1, col] = median
        lin_test.loc[lin_test[col] == -1, col] = median

    # lasso and ridge: simpler CV since these pick the N columns to retain
    # no need to iterate over the N token values, given top 100 and that's it
    # standardize first
    all_vars = base_vars + top_cols["cond"] + top_cols["title"] + top_cols["series"]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(lin_train[all_vars])
    x_test = scaler.transform(lin_test[all_vars])
    folds = PredefinedSplit(data["fold"])
    lasso = LassoCV(cv=folds, n_jobs=-1).fit(x_train, y_train)
    ridge = RidgeCV(alphas=np.logspace(-2, 6, 50), cv=folds).fit(x_train, y_train)
    results.append({"round": r, "model": "lasso", "text": "scores + terms", "n_terms": max(n_values),
                    "share_removed": ssr_share(lasso.predict(x_test), y_val)})
    results.append({"round": r, "model": "ridge", "text": "scores + terms", "n_terms": max(n_values),
                    "share_removed": ssr_share(ridge.predict(x_test), y_val)})

    # OLS for every N token value and RF for every N token, mtry x min node combo
    for g in n_grid:
        if g["text"] == "terms only":
            keep = [v for v in base_vars if v not in score_vars]
        else:
            keep = base_vars
        n = g["n_terms"]
        predictors = keep + top_cols["cond"][:n] + top_cols["title"][:n] + top_cols["series"][:n]

        ols = LinearRegression().fit(lin_train[predictors], y_train)
        results.append({"round": r, "model": "ols", "text": g["text"], "n_terms": n,
                        "share_removed": ssr_share(ols.predict(lin_test[predictors]), y_val)})

        for m in max_feat:
            if m > len(predictors):
                continue
            for samp in min_samples:
                rf = RandomForestRegressor(n_estimators=500, max_features=m, min_samples_split=samp,
                                           random_state=594, n_jobs=-1)
                rf.fit(model_train[predictors], y_train)
                results.append({"round": r, "model": "rf", "text": g["text"], "n_terms": n,
                                "features": m, "min_node": samp,
                                "share_removed": ssr_share(rf.predict(model_test[predictors]), y_val)})
        print("  ", g["text"], n, "done")

# mean SSR removed over all CV rounds
results = pd.DataFrame(results)
settings = ["model", "text", "n_terms", "features", "min_node"]
summary = results.groupby(settings, dropna=False)["share_removed"].mean().reset_index()
summary = summary.rename(columns={"share_removed": "mean_share_removed"})
summary = summary.sort_values("mean_share_removed", ascending=False)
summary.to_csv(os.path.join(submission_folder, "cv_results_py.csv"), index=False)

# best forest parameters for each text setting and linear model
best = summary.groupby(["model", "text", "n_terms"]).head(1)
print(best.to_string(index=False))
# print the best model (didn't end up being the one I picked)
print("\nbest forest:")
print(summary[summary["model"] == "rf"].head(1).to_string(index=False))
