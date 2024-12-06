# %%
import pandas as pd

scenarios = [
    "train_egll_test_3_trk",
    "train_egll_test_3_trk_vr",
    "train_eidw_test_3_trk",
    "train_eidw_test_3_trk_vr",
    "train_3_test_eglc_trk",
    "train_3_test_eglc_trk_vr",
    "train_3_test_eidw_trk",
    "train_3_test_eidw_trk_vr",
    "train_3_test_lfpg_trk",
    "train_3_test_lfpg_trk_vr",
]
# %%

def holding(hp_label, hp_labels):
    for flight_id, f in hp_labels.groupby("flight_id"):
        previous_label, label = False, False
        first, last = None, None
        for start, stop, label, airport in zip(
            f.start, f.stop, f[hp_label], f.airport
        ):
            if label and first is None:
                first, last = start, stop
            if not label and previous_label:
                yield flight_id, first, last, airport
                first, last = None, None
            if label and previous_label:
                last = stop

            previous_label = label
        if label:
            yield flight_id, first, last, airport


def compute_hp_table(scn):
    hp_labels = pd.read_pickle(f"cache/{scn}/hp_labels.pkl")
    hp_table = pd.concat(
        [
            pd.DataFrame.from_records(
                [hp for hp in holding("hp", hp_labels)],
                columns=["flight_id", "start", "stop", "airport"],
            ).assign(label="t"),
            pd.DataFrame.from_records(
                [hp for hp in holding("hp_c", hp_labels)],
                columns=["flight_id", "start", "stop", "airport"],
            ).assign(label="c"),
        ]
    ).sort_values(["flight_id", "start"])
    hp_table = hp_table.assign(d=hp_table.stop - hp_table.start)
    hp_table.to_pickle(f"{scn}/hp_table.pkl")
    return hp_table


# %%
from tqdm.autonotebook import tqdm

cumul = []
pbar = tqdm(scenarios)
for scn in pbar:
    pbar.set_description(f"{scn}")
    cumul.append(compute_hp_table(scn).assign(scn=scn).reset_index(drop=True))
hp_table = pd.concat(cumul)
hp_table.to_parquet("hp_table.parquet.gz")
hp_table


# %%
def detection_metrics(hp_table):
    cumul = []
    for flight_id, hp in tqdm(hp_table.groupby("flight_id")):
        hp_t = hp.query("label=='t'")
        hp_c = hp.query("label=='c'")
        d_t = pd.Timedelta(0) if hp_t.empty else hp_t[["d"]].sum().item()
        d_c = pd.Timedelta(0) if hp_c.empty else hp_c[["d"]].sum().item()
        intersection, union = pd.Timedelta(0), pd.Timedelta(0)
        if not d_t:
            union = d_c
        elif not d_c:
            union = d_t
        else:
            for _, t in hp_t.iterrows():
                for _, c in hp_c.iterrows():
                    intersection += max(
                        pd.Timedelta(0),
                        min(t.stop, c.stop) - max(t.start, c.start),
                    )
                    union += max(t.stop, c.stop) - min(t.start, c.start)
        cumul.append(
            (flight_id, d_t, d_c, intersection, union, intersection / union)
        )
    return pd.DataFrame.from_records(
        cumul,
        columns=["flight_id", "d_t", "d_c", "intersection", "union", "iou"],
    )


# %%
hp_table = pd.read_parquet("data/hp_table.parquet.gz")
cumul = []
pbar = tqdm(scenarios)
for scn in pbar:
    pbar.set_description(f"{scn}")
    cumul.append(
        detection_metrics(hp_table.query(f"scn=='{scn}'")).assign(scn=scn)
    )
metrics = (
    pd.concat(cumul)
    .merge(
        hp_table[["flight_id", "airport"]],
        on="flight_id",
        how="left",
    )
    .drop_duplicates()
    .reset_index(drop=True)
)
metrics.to_parquet("data/detection_metrics.parquet.gz")
metrics.sort_values(["iou"], ascending=False)


# %%
def perf(metrics):
    iou = metrics.iou.mean()
    intersection = metrics.intersection.mean()
    d_t = metrics.d_t.mean()
    d_c = metrics.d_c.mean()
    precision = intersection / d_c
    recall = intersection / d_t
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1, iou


# %%

cumul = []
for (scn, airport), scn_metrics in metrics.groupby(["scn", "airport"]):
    cumul.append((scn, airport, *perf(scn_metrics)))
perf_arpt = pd.DataFrame.from_records(
    cumul, columns=["scn", "airport", "precision", "recall", "f1", "iou"]
).sort_values(["scn", "f1"], ascending=False)
perf_arpt

# %%
cumul = []
for scn, scn_metrics in metrics.groupby("scn"):
    cumul.append((scn, *perf(scn_metrics)))
perf_scn = pd.DataFrame.from_records(
    cumul, columns=["scn", "precision", "recall", "f1", "iou"]
).sort_values("f1", ascending=False)
perf_scn

# %%
def set_ds(x, airports=["eglc", "egll", "eidw", "lfpg"]):
    scn_elements = x.scn.split("_")
    if scn_elements[1] in airports and x.airport == scn_elements[1]:
        return "train"
    if scn_elements[3] in airports and x.airport == scn_elements[3]:
        return "test"
    if scn_elements[1] == "3":
        return "train"
    if scn_elements[3] == "3":
        return "test"


cumul = []
for (scn, ds), scn_metrics in metrics.assign(
    ds=lambda x: x.apply(lambda x: set_ds(x), axis=1)
).groupby(["scn", "ds"]):
    cumul.append((scn, ds, *perf(scn_metrics)))
perf_per_ds = pd.DataFrame.from_records(
    cumul, columns=["scn", "ds", "precision", "recall", "f1", "iou"]
).sort_values(["ds", "f1"], ascending=False)[
    ["ds", "precision", "recall", "f1", "iou", "scn"]
]
perf_per_ds  # .query("ds=='test'")
