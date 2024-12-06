# %%
import matplotlib.pyplot as plt
from cartes.crs import OSGB  # type: ignore

import pandas as pd
from traffic.data import navaids
from traffic.data.datasets import holding_EGLL, holding_EIDW

# %%
hp_undirect = (
    (undirect := holding_EIDW["AAL208_17738"]).holding_pattern().next()
)
undirect = undirect.between(
    hp_undirect.start - pd.Timedelta("7T 30s"),
    hp_undirect.stop + pd.Timedelta("60s"),
)

hp_exit_in_loop = (
    (exit_in_loop := holding_EGLL["BAW391_9297"]).holding_pattern().next()
)
exit_in_loop = exit_in_loop.between(
    hp_exit_in_loop.start - pd.Timedelta("2T"),
    hp_exit_in_loop.stop + pd.Timedelta("30s"),
)

# holding_EGLL["BAW815_9117"].holding_pattern()

hp_tear_drop = (
    (tear_drop := holding_EGLL["BAW89P_22228"]).holding_pattern().next()
)
tear_drop = tear_drop.between(
    hp_tear_drop.start - pd.Timedelta("3T"),
    hp_tear_drop.stop + pd.Timedelta("7T"),
)

hp_direct = (direct := holding_EGLL["ACA854_38423"]).holding_pattern().next()
direct = direct.between(
    hp_direct.start - pd.Timedelta("7T"),
    hp_direct.stop + pd.Timedelta("2T"),
)

# %%

dpi = 300
scale = dpi // 100

with plt.style.context("traffic"):
    fig, ax = plt.subplots(
        1, 4, figsize=(12, 4), subplot_kw=dict(projection=OSGB()), dpi=dpi
    )

    direct.first("20T").plot(ax[0])
    direct.first("20T").at_ratio(0.07).plot(ax[0], zorder=5, text_kw=dict(s=""))
    navaids.extent(direct)["BNN"].plot(  # type: ignore
        ax[0],
        marker="^",
        text_kw=dict(font="B612"),
        shift=dict(units="dots", x=scale * 20),
    )

    tear_drop.plot(ax[1])
    tear_drop.at_ratio(0.05).plot(ax[1], zorder=5, text_kw=dict(s=""))
    navaids.extent(tear_drop)["OCK"].plot(  # type: ignore
        ax[1],
        marker="^",
        text_kw=dict(font="B612"),
        shift=dict(units="dots", x=scale * 20),
    )

    exit_in_loop.plot(ax[2])
    exit_in_loop.point.plot(ax[2], zorder=5, text_kw=dict(s=""))
    navaids.extent(exit_in_loop)["LAM"].plot(  # type: ignore
        ax[2],
        marker="^",
        shift=dict(units="dots", x=15 * scale, y=15 * scale),
        text_kw=dict(font="B612"),
    )

    undirect.first("23T").plot(ax[3])
    undirect.first("23T").at_ratio(0.07).plot(
        ax[3], zorder=5, text_kw=dict(s="")
    )
    navaids["EKREN"].plot(  # type:ignore
        ax[3],
        marker="^",
        text_kw=dict(font="B612"),
        shift=dict(units="dots", x=scale * 20),
    )

    ax[0].text(0.1, 0.8, "a.", transform=fig.transFigure, font="B612")
    ax[1].text(0.3, 0.8, "b.", transform=fig.transFigure, font="B612")
    ax[2].text(0.5, 0.8, "c.", transform=fig.transFigure, font="B612")
    ax[3].text(0.75, 0.8, "d.", transform=fig.transFigure, font="B612")

    fig.savefig("../figures/holding_patterns.png", dpi=dpi, transparent=False)

# %%
# map heathrow
# %%
