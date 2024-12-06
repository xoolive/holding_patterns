# %%

from traffic.data.datasets import landing_zurich_2019

landing_zurich_2019.query("not simple")

# %%
f = landing_zurich_2019["EDW59R_7984"]
assert f is not None
f

# %%
f.holding_pattern()
# %%
f.sliding_windows("6 min", "4 min")
# %%
fig = plot_holding(f, airport="LSZH", buffer=0.2, last="30 min")
fig.savefig("figures/sliding_windows_00.png", dpi=200, bbox_inches="tight")
fig
# %%
f = landing_dublin_2019["ICE416_16873"]
f = landing_londoncity_2019["AZA216_4152"]
# %%

import matplotlib.pyplot as plt
from cartes.crs import EuroPP
from matplotlib.transforms import offset_copy

cmap = plt.get_cmap("Reds")
n_colors = 10

with plt.style.context("traffic"):
    fig, ax = plt.subplots(subplot_kw=dict(projection=EuroPP()))
    for i, segment in enumerate(
        f.compute_xy(EuroPP()).sliding_windows("6 min", "2 min")
    ):
        shifted = offset_copy(
            ax.transData, fig, y=10 * i, x=10 * i, units="dots"
        )
        if segment.duration > pd.Timedelta("4 min"):
            segment.data.plot(
                ax=ax,
                x="x",
                y="y",
                # color=cmap((i + 2) / n_colors),
                transform=shifted,
                legend=False,
            )

    ax.yaxis.set_visible(False)
    ax.set_extent(f.last("30 min"), buffer=0.25)
    square_ratio(ax)

fig.savefig("figures/sliding_windows.png", dpi=200, bbox_inches="tight")
fig
# %%
