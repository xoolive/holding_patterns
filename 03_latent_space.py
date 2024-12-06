# %%
from pathlib import Path

from traffic.core import Traffic
from traffic.data.datasets import landing_zurich_2019

root = Path("cache/lszh_latent")
if not root.is_dir():
    root.mkdir(parents=True)

if not (root / "t.parquet").exists():
    t = (
        landing_zurich_2019.query("track == track")
        .unwrap()
        .eval(desc="", max_workers=4)
    )
    assert t is not None
    t.to_parquet(root / "t.parquet")
else:
    t = Traffic.from_file(root / "t.parquet")

if not (root / "thp.parquet").exists():
    thp = t.has("holding_pattern").eval(desc="", max_workers=4)
    assert thp is not None
    thp.to_parquet(root / "thp.parquet")
else:
    thp = Traffic.from_file(root / "thp.parquet")

# %%
if not (root / "tga.parquet").exists():
    tga = t.has("go_around").eval(desc="", max_workers=4)
    assert tga is not None
    tga.to_parquet(root / "tga.parquet")
else:
    tga = Traffic.from_file(root / "tga.parquet")

# %%
if not (root / "tns.parquet").exists():
    tns = t.query("not simple")
    assert tns is not None
    tns.to_parquet(root / "tns.parquet")
else:
    tns = Traffic.from_file(root / "tns.parquet")

# %%

# thp - tns | (thp & tns) - tga  # fix __ror__
thp | tga | thp & tga | thp - tga | tga - thp


# %%

from tqdm.autonotebook import tqdm

import numpy as np
import pandas as pd

flight_ids = []
cumul1 = []
cumul2 = []
cumul3 = []

for f in tqdm(t, leave=False):
    for i, sw in enumerate(f.sliding_windows("6 min", "2 min")):
        if sw.duration >= pd.Timedelta("5 min"):
            sw = sw.assign(flight_id=sw.flight_id + f"_{i}")
            flight_ids.append(sw.flight_id)
            swr = sw.resample(30)
            cumul1.append(swr.data)  # bizarre
            cumul2.append(
                swr.data.track_unwrapped - swr.data.track_unwrapped[0]
            )
            cumul3.append(sw.data)  # bizarre

pd.concat(cumul1).to_parquet(root / "twr.parquet")
X = np.vstack(cumul2)
np.save(root / "X.npy", X)
(tw := Traffic(pd.concat(cumul3))).to_parquet(root / "tw.parquet")

# %%
from tqdm.autonotebook import tqdm

flight_ids = []
for f in tqdm(t, leave=False):
    for i, sw in enumerate(f.sliding_windows("6 min", "2 min")):
        if sw.duration >= pd.Timedelta("5 min"):
            flight_ids.append(sw.flight_id + f"_{i}")
flight_ids

# %%
import pickle

(root / "flight_ids.pkl").write_bytes(pickle.dumps(flight_ids))
# %%
tw = Traffic.from_file(root / "tw.parquet")
# %%
twr = Traffic.from_file(root / "twr.parquet")

# %%
from traffic.core import Traffic

if not (root / "twhp.parquet").exists():
    (twhp := tw.has("holding_pattern").eval(desc="", max_workers=6)).to_parquet(
        root / "twhp.parquet"
    )
else:
    twhp = Traffic.from_file(root / "twhp.parquet")

# %%

X = np.load(root / "X.npy")

# %%
from sklearn.preprocessing import MinMaxScaler

s = MinMaxScaler()
x = s.fit_transform(X)

# %%
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import TensorDataset

dataset = TensorDataset(torch.Tensor(x))


class Autoencoder(pl.LightningModule):
    def __init__(self, lr=1e-3, weight_decay=1e-5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(30, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 30),
            nn.Sigmoid(),
        )
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        (x,) = batch

        latent = self.encoder(x)
        x_hat = self.decoder(latent)

        vanilla_loss = F.mse_loss(x, x_hat)
        self.log("vanilla_loss", vanilla_loss)

        return vanilla_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer


model = Autoencoder.load_from_checkpoint(
    root.parent.parent
    / "holding/bootstrap/lightning_logs/6T2T5T/version_2/checkpoints/epoch=49-step=41849.ckpt"
)


# %%
X_ae = model(torch.Tensor(x)).detach().cpu().numpy()
# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(X_ae[:, 0], X_ae[:, 1], alpha=0.025)
fig
# %%
# %%
lat = X_ae
df = pd.DataFrame.from_records(
    [
        {
            "flight_id": id_,
            "x": x,
            "y": y,
        }
        for (id_, x, y) in zip(
            flight_ids,
            lat[:, 0],
            lat[:, 1],  # , ae_res["final_loss"],
        )
    ]
)
t_ae = Traffic(df.merge(twr.data).drop_duplicates())

# %%

t_ae.to_parquet(root / "t_ae.parquet")

# %%
import matplotlib.pyplot as plt
from cartes.crs import EuroPP
from cartes.utils.features import countries, rivers
from matplotlib.figure import Figure

from traffic.data import airports
from traffic.visualize.markers import atc_tower


def plot_lat_trajs(t, thp=None, tga=None, num_trajs=20) -> Figure:
    with plt.style.context("traffic"):
        colors = [
            "779ae3",
            "e77074",
            "74c863",
            "c2c55e",
            "aaaaaa",
        ]  # "d27bcd", "60ceaf",

        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(121)
        m = fig.add_subplot(122, projection=EuroPP())
        m.add_feature(
            countries(
                edgecolor="white",
                facecolor="#d9dadb",
                alpha=1,
                linewidth=2,
                zorder=-2,
            )
        )
        m.spines["geo"].set_visible(False)

        text_style = dict(
            verticalalignment="top",
            horizontalalignment="right",
            fontname="Ubuntu",
            fontsize=18,
            bbox=dict(facecolor="white", alpha=0.6, boxstyle="round"),
        )

        airports["LSZH"].point.plot(
            m,
            shift=dict(units="dots", x=-15, y=-15),
            marker=atc_tower,
            s=300,
            zorder=5,
            text_kw={**text_style},
        )
        t[:30000].groupby("flight_id").first().plot.scatter(
            x="x",
            y="y",
            ax=ax,
            color=colors[0],
            label="various",
            alpha=0.2,
        )
        t[:num_trajs].plot(m, color=colors[0], linewidth=1.5, alpha=0.3)
        if thp is not None:
            thp.groupby("flight_id").first().plot.scatter(
                x="x",
                y="y",
                ax=ax,
                color=colors[1],
                label="holding pattern",
                alpha=0.4,
            )
            thp[:num_trajs].plot(m, color=colors[1], linewidth=1.5, alpha=0.5)
        if tga is not None:
            tga.groupby("flight_id").first().plot.scatter(
                x="x",
                y="y",
                ax=ax,
                color=colors[2],
                label="hp",
                alpha=0.4,
            )
            tga[:num_trajs].plot(m, color=colors[2], linewidth=1.5, alpha=0.5)

        ax.legend(prop=dict(family="Ubuntu", size=24))
        ax.grid(linestyle="solid", alpha=0.5, zorder=-2)

        ax.set_xlabel("1st component on latent space", labelpad=10)
        ax.set_ylabel("2nd component on latent space", labelpad=10)

    return fig


# %%
fig = plot_lat_trajs(t_ae, t_ae & twhp, num_trajs=300)

# %%
fig.savefig(
    "figures/latent_space.png",
    dpi=200,
    transparent=True,
    bbox_inches="tight",
)

# %%
(t_ae & twhp)[:150]
# %%
f = t["UAL52_20691"]
# %%
axes = fig.get_axes()
f = t["UAL52_20691"]
sub = t_ae.query('flight_id.str.startswith("UAL52_20691")')
sub.data.plot(
    x="x",
    y="y",
    color="white",
    marker="o",
    lw=7,
    ax=axes[0],
    legend=None,
    label="UAL52",
)
sub.data.plot(
    x="x",
    y="y",
    color="#e45756",
    marker="o",
    lw=3,
    ax=axes[0],
    legend=None,
    label="UAL52",
)
for i, row in sub.data.iterrows():
    axes[0].text(
        row.x,
        row.y,
        row.flight_id.split("_")[-1],
        fontdict=dict(size=14),
        color="#e45756",
        bbox=dict(facecolor="white", alpha=0.6, boxstyle="round"),
    )

f.plot(axes[1], lw=7, color="white")
f.plot(axes[1], lw=3, color="#e45756")

axes[0].grid(linestyle="solid", alpha=0.5, zorder=-2)
axes[0].get_legend().set_visible(False)

axes[0].set_xlabel("1st component on latent space", labelpad=10)
axes[0].set_ylabel("2nd component on latent space", labelpad=10)
fig.savefig(
    "figures/latent_trajectory.png",
    dpi=200,
    transparent=True,
    bbox_inches="tight",
)
fig

# %%
from sklearn.mixture import GaussianMixture

c = GaussianMixture(n_components=4, covariance_type="full").fit_predict(lat)
# %%
from matplotlib.colors import ListedColormap

cmap = ListedColormap(["#4c78a8", "#54a24b", "#9ecae9", "#f58518"])
with plt.style.context("traffic"):
    fig, ax = plt.subplots(figsize=((7, 7)))
    ax.scatter(lat.T[0], lat.T[1], c=c, s=5, cmap=cmap)
    # ax.set_aspect(1)
    ax.set_xlabel("1st component on latent space", labelpad=10)
    ax.set_ylabel("2nd component on latent space", labelpad=10)

fig.savefig(
    "figures/latent_clustering.png",
    dpi=200,
    transparent=True,
    bbox_inches="tight",
)

# %%
