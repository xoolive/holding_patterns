# %%

import pandas as pd
from cache_results import cache_pandas
from pathlib import Path
from traffic.core import Traffic
from tqdm.autonotebook import tqdm


@cache_pandas(path=Path("cache/"), pd_varnames=True)
def prepare_data(
    t: Traffic, duration: str, step: str, threshold: str, samples: int
) -> pd.DataFrame:
    flight_ids, track, vertical_rate, start, stop = [], [], [], [], []
    for flight in tqdm(t, leave=False):
        for i, window in enumerate(
            flight.sliding_windows(duration=duration, step=step)
        ):
            if window.duration >= pd.Timedelta(threshold):
                window = window.assign(
                    flight_id=window.flight_id + "_" + str(i)
                )
                resampled = window.resample(samples)
                if resampled.data.eval(
                    "track != track or vertical_rate != vertical_rate"
                ).any():
                    continue
                flight_ids.append(window.flight_id)
                track.append(
                    resampled.data.track_unwrapped
                    - resampled.data.track_unwrapped[0]
                )
                vertical_rate.append(resampled.data.vertical_rate)
                start.append(window.start)
                stop.append(window.stop)

    return pd.DataFrame(
        dict(
            flight_id=flight_ids,
            track=track,
            vertical_rate=vertical_rate,
            start=start,
            stop=stop,
        )
    )


from traffic.data.datasets import landing_amsterdam_2019 as eham
from traffic.data.datasets import landing_cdg_2019 as lfpg
from traffic.data.datasets import landing_dublin_2019 as eidw
from traffic.data.datasets import landing_heathrow_2019 as egll
from traffic.data.datasets import landing_londoncity_2019 as eglc
from traffic.data.datasets import landing_zurich_2019 as lszh

prepare_data(eglc, "6min", "2min", "5min", 30)
prepare_data(egll, "6min", "2min", "5min", 30)
prepare_data(eham, "6min", "2min", "5min", 30)
prepare_data(eidw, "6min", "2min", "5min", 30)
prepare_data(lfpg, "6min", "2min", "5min", 30)
prepare_data(lszh, "6min", "2min", "5min", 30)

# %%
import pyarrow as pa

hp_df = []
for airport in ["eglc", "egll", "eidw", "lfpg"]:
    hp_df.append(
        prepare_data(airport, "6min", "2min", "5min", 30)
        .merge(
            pd.read_pickle(f"labels_manual_{airport}.pkl"),
            left_on="flight_id",
            right_on="full_id",
            how="inner",
            suffixes=("_x", ""),
        )
        .assign(airport=airport)[
            [
                "full_id",
                "flight_id",
                "track",
                "vertical_rate",
                "start",
                "stop",
                "airport",
                "hp",
            ]
        ]
    )

hp_df = pd.concat(hp_df).reset_index(drop=True)
schema = pa.schema(
    [
        ("full_id", pa.string()),
        ("flight_id", pa.string()),
        ("track", pa.list_(pa.float64())),
        ("vertical_rate", pa.list_(pa.float64())),
        ("start", pa.timestamp("ns", tz="UTC")),
        ("stop", pa.timestamp("ns", tz="UTC")),
        ("airport", pa.string()),
        ("hp", pa.bool_()),
    ]
)
hp_df.to_parquet("data/hp_df.parquet.gz", schema=schema)

# %%
hp_df[
    [
        "full_id",
        "flight_id",
        "start",
        "stop",
        "airport",
        "hp",
    ]
].to_parquet("data/hp_labels.parquet.gz")

# %%
import joblib
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import numpy as np


def split_by_airport(hp_df, train=["egll"], test=["eglc", "eidw", "lfpg"]):
    return hp_df.query(f"airport in {train}"), hp_df.query(f"airport in {test}")


def print_stats(train_df, val_df, test_df, hp_df):
    print(
        "len",
        len(train_df),
        len(val_df),
        len(test_df),
        len(train_df) + len(val_df) + len(test_df),
        len(hp_df),
    )
    print("holding_pattern_ratio (all)", len(hp_df.query("hp")) / len(hp_df))
    print(
        "holding_pattern_ratio (train)",
        len(train_df.query("hp")) / len(train_df),
    )
    print(
        "holding_pattern_ratio (val)",
        len(val_df.query("hp")) / len(val_df) if len(val_df) > 0 else 0,
    )
    print(
        "holding_pattern_ratio (test)", len(test_df.query("hp")) / len(test_df)
    )


def scale(scn, train_df, val_df, test_df, hp_df, add_vertical_rate=False):
    def get_x(df):
        return (
            np.concatenate(
                (
                    np.stack(df.track.values),
                    np.stack(df.vertical_rate.values),
                ),
                axis=1,
            )
            if add_vertical_rate
            else np.stack(df.track.values)
        ).astype(np.float32)

    def get_y(df):
        return df.hp.values.astype(np.float32)

    s = MinMaxScaler()
    # train
    x = s.fit_transform(get_x(train_df))
    joblib.dump(s, f"{scn}/scaler.pkl")
    np.save(f"{scn}/x_train.npy", x)
    np.save(f"{scn}/y_train.npy", get_y(train_df))
    # val
    if len(val_df) > 0:
        x = s.transform(get_x(val_df))
        np.save(f"{scn}/x_val.npy", x)
        np.save(f"{scn}/y_val.npy", get_y(val_df))
    # test
    x = s.transform(get_x(test_df))
    np.save(f"{scn}/x_test.npy", x)
    np.save(f"{scn}/y_test.npy", get_y(test_df))
    # all
    x = s.transform(get_x(hp_df))
    np.save(f"{scn}/x.npy", x)
    np.save(f"{scn}/y.npy", get_y(hp_df))


def df_split(df, frac):
    df1 = df.sample(frac=frac, random_state=200)
    df2 = df.drop(df1.index)
    return df1, df2


def prepare(
    scn,
    hp_df,
    train=["egll"],
    test=["eglc", "eidw", "lfpg"],
    frac_val=0.2,
):
    train_df, test_df = split_by_airport(
        hp_df,
        train,
        test,
    )
    val_df = None
    if len(test_df) > len(train_df):
        print("split test")
        test_df, val_df = df_split(test_df, 1 - frac_val)
    else:
        print("split train")
        train_df, val_df = df_split(train_df, 1 - frac_val)
    print_stats(train_df, val_df, test_df, hp_df)
    scn_path = Path(f"{scn}")
    scn_path.mkdir(parents=True, exist_ok=True)

    # np.save(f"{scn}/train_full_ids.npy", train_df.full_id.values)
    # np.save(f"{scn}/test_full_ids.npy", test_df.full_id.values)
    add_vertical_rate = scn.split("_")[-1] == "vr"
    scale(scn, train_df, val_df, test_df, hp_df, add_vertical_rate)


# %%
scn = "train_egll_test_3_trk"
prepare(scn, hp_df, train=["egll"], test=["eglc", "eidw", "lfpg"], frac_val=0.2)

# %%
scn = "train_egll_test_3_trk_vr"
prepare(
    scn,
    hp_df,
    train=["egll"],
    test=["eglc", "eidw", "lfpg"],
    frac_val=0.2,
)

# %%
scn = "train_eidw_test_3_trk"
prepare(scn, hp_df, train=["eidw"], test=["eglc", "egll", "lfpg"])

# %%

scn = "train_eidw_test_3_trk_vr"
prepare(scn, hp_df, train=["eidw"], test=["eglc", "egll", "lfpg"])

# %%
scn = "train_3_test_eidw_trk"
prepare(scn, hp_df, train=["egll", "eglc", "lfpg"], test=["eidw"])

# %%
scn = "train_3_test_eidw_trk_vr"
prepare(scn, hp_df, train=["egll", "eglc", "lfpg"], test=["eidw"])

# %%
scn = "train_3_test_eglc_trk"
prepare(scn, hp_df, train=["egll", "eidw", "lfpg"], test=["eglc"])

# %%
scn = "train_3_test_eglc_trk_vr"
prepare(scn, hp_df, train=["egll", "eidw", "lfpg"], test=["eglc"])

# %%
scn = "train_3_test_lfpg_trk"
prepare(scn, hp_df, train=["egll", "eglc", "eidw"], test=["lfpg"])

# %%
scn = "train_3_test_lfpg_trk_vr"
prepare(scn, hp_df, train=["egll", "eglc", "eidw"], test=["lfpg"])

# %%
