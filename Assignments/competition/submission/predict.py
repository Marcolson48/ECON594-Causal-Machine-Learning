import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from clean import make_model_data, release_folder, submission_folder
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence

# final model: fit on all of train, predict the real test set

# define which model to use here
text = "terms only"  # "scores only", "scores + terms" or "terms only"
n_terms = 100        # top N terms per text field (0 for scores only)
features = 50        # max_features: columns at each split
min_node = 5         # min_samples_split: smallest node to be split


# final mode in make_model_data builds the variables from all trainig data
# and applies same to test data
data = make_model_data(cv_round=None, n_terms=n_terms)
model_train = data["model_train"]
model_test = data["model_test"]
top_cols = data["top_cols"]

score_vars = ["cond_score", "title_score", "series_score"]
if text == "terms only":
    keep = [v for v in data["base_vars"] if v not in score_vars]
else:
    keep = data["base_vars"]
predictors = keep + top_cols["cond"][:n_terms] + top_cols["title"][:n_terms] + top_cols["series"][:n_terms]

# write the final training and testing dataset (test still without predictions)
model_train[["row_id", "log_bid"] + predictors].to_csv(os.path.join(submission_folder, "final_train.csv"), index=False)
model_test[["row_id"] + predictors].to_csv(os.path.join(submission_folder, "final_test.csv"), index=False)

# fit the final RF
rf = RandomForestRegressor(n_estimators=500, max_features=features, min_samples_split=min_node,
                           random_state=594, n_jobs=-1)
rf.fit(model_train[predictors], model_train["log_bid"])

# format predictions so that the predictions file doesnt contain all the columns I created
# that can be found in final_test.csv
predictions = pd.read_csv(os.path.join(release_folder, "test.csv"), dtype=str, keep_default_na=False)
assert (predictions["row_id"].astype(int) == model_test["row_id"]).all() 
predictions["prediction"] = rf.predict(model_test[predictors])

predictions.to_csv(os.path.join(submission_folder, "predictions.csv"), index=False)


# VIP for the report
importance = pd.Series(rf.feature_importances_, index=predictors).sort_values(ascending=False)
importance.to_csv(os.path.join(submission_folder, "variable_importance.csv"), header=["importance"])

top20 = importance.head(20).sort_values()
fig, ax = plt.subplots(figsize=(7, 6))
ax.barh(top20.index, top20.values)
ax.set_xlabel("variable importance")
ax.set_title("Top 20 predictors in the final forest")
fig.tight_layout()
fig.savefig(os.path.join(submission_folder, "variable_importance.pdf"))

# PDPs for the top 4 most important predictors according to VIP
top4 = list(importance.index[:4])
fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
for ax, v in zip(axes.ravel(), top4):
    is_year = v in ["made_year", "artist_year"]
    is_binary = model_train[v].isin([0, 1]).all()
    if is_year:
        pd_result = partial_dependence(rf, model_train[predictors], [v], percentiles=(0, 0.95), grid_resolution=1000)
    else:
        pd_result = partial_dependence(rf, model_train[predictors], [v])
    grid = pd_result["grid_values"][0]
    average = pd_result["average"][0]

    if is_binary:
        ax.bar(grid, average, width=0.6)
        ax.set_xticks([0, 1])
    elif is_year:
        first_year = model_train.loc[model_train[v] != -1, v].quantile(0.05)
        real = grid >= first_year
        no_date_x = first_year - 0.1 * (grid.max() - first_year)  # where -1 is drawn
        ax.plot(grid[real], average[real])
        year_ticks = list(range(int(np.ceil(first_year / 25) * 25), int(grid.max()) + 1, 25))
        if grid[0] == -1:  # some lots have no date
            ax.plot(no_date_x, average[0], "o")
            ax.set_xticks([no_date_x] + year_ticks)
            ax.set_xticklabels(["-1\n(no date)"] + [str(y) for y in year_ticks])
    else:
        ax.plot(grid, average)
    ax.set_xlabel(v)
for ax in axes[:, 0]:
    ax.set_ylabel("predicted ln(price)")
low = min(line.get_ydata().min() for ax in axes.ravel() for line in ax.get_lines())
high = max(line.get_ydata().max() for ax in axes.ravel() for line in ax.get_lines())
for ax in axes.ravel():
    for bar in ax.patches:
        low = min(low, bar.get_height())
        high = max(high, bar.get_height())
axes[0, 0].set_ylim(low - 0.1, high + 0.1)
fig.suptitle("Partial dependence of predicted ln(price) on the top 4 predictors")
fig.tight_layout()
fig.savefig(os.path.join(submission_folder, "partial_dependence.pdf"))
