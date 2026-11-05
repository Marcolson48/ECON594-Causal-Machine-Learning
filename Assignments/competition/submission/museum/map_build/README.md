# Building `artist_museum_map.csv`

The map links each of the 377 artist labels in train and test to its records at the
Art Institute of Chicago (AIC) and the Metropolitan Museum of Art (Met).
`museum_counts.py` (one folder up) reads it to count each artist's works.

Run with the course `cml` environment, in order:

| Step | Script | Writes | Notes |
|---|---|---|---|
| 1 | `1_artist_labels.py` | `all_labels.csv` | labels, lot counts, dates from `release/` |
| 2 | `2_aic_search.py` | `aic_candidates.csv`, `aic_candidates_tail.csv` | AIC public API, about 20 minutes |
| 3 | `3_aic_tail_matches.py` | `aic_tail_matches.csv` | review sheet for labels outside the top 100 |
| 4 | `4_met_artists.py` | `met_japanese_artists.csv` | downloads the Met open access file (341 MB) to a temp folder |
| 5 | `5_met_match.py` | `met_matches.csv` | first-pass Met matches by name and dates |
| 6 | `6_assemble_map.py` | `../artist_museum_map.csv` | the hand-checked matches |

Steps 2-5 only produce candidates. The matches themselves were chosen by hand, checking
names and dates, and are written out in `6_assemble_map.py`, with judgment calls in the
map's `note` column. The outputs of steps 1-5 are saved here, so `6_assemble_map.py`
alone rebuilds the map exactly. Re-running step 2 queries the live AIC API, which can
return slightly different candidates over time.
