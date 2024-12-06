# ruff: noqa: E402, F401

# %%
# %config InlineBackend.figure_format = 'retina'

# %%
from traffic.data.datasets import (
    landing_amsterdam_2019,
    landing_cdg_2019,
    landing_dublin_2019,
    landing_heathrow_2019,
    landing_londoncity_2019,
    landing_zurich_2019,
)
from traffic.data.samples import belevingsvlucht

# %%
f = landing_londoncity_2019["DLH936_034"]
f.map_leaflet(highlight=dict(red="holding_pattern"))
# %%
f = landing_londoncity_2019["AZA216_4152"]
f.map_leaflet(highlight=dict(red="holding_pattern"))
# %%
f = landing_londoncity_2019["CFE2213_1434"]
f.map_leaflet(highlight=dict(red="holding_pattern"))

# %%
f = landing_dublin_2019["AAL722_17460"]
f.map_leaflet(highlight=dict(red="holding_pattern"))
# %%
f = landing_dublin_2019["VOR05_1616"]
f.map_leaflet()
# %%
belevingsvlucht.map_leaflet(highlight=dict(red="holding_pattern"))

# %%

import matplotlib.pyplot as plt
from cartes.crs import EuroPP
from cartes.utils.features import countries
from cartopy.mpl.geoaxes import GeoAxesSubplot
from matplotlib.figure import Figure

from traffic.core import Flight
from traffic.data import airports
from traffic.visualize.markers import atc_tower


def square_ratio(ax: GeoAxesSubplot) -> None:
    bbox = ax.get_extent(crs=EuroPP())
    lon_range = bbox[1] - bbox[0]
    lat_range = bbox[3] - bbox[2]
    aspect_ratio = lon_range / lat_range

    # Adjust the extent to make the aspect ratio 1:1
    if aspect_ratio > 1:
        center_lat = (bbox[3] + bbox[2]) / 2
        lat_range = lon_range
        new_extent = [
            bbox[0],
            bbox[1],
            center_lat - lat_range / 2,
            center_lat + lat_range / 2,
        ]
    else:
        center_lon = (bbox[1] + bbox[0]) / 2
        lon_range = lat_range
        new_extent = [
            center_lon - lon_range / 2,
            center_lon + lon_range / 2,
            bbox[2],
            bbox[3],
        ]

    # Set the new extent to achieve a 1:1 aspect ratio
    ax.set_extent(new_extent, EuroPP())


def plot_holding(
    flight: Flight,
    /,
    *,
    airport: str,
    last: str,
    buffer: float = 0,
) -> Figure:
    with plt.style.context("traffic"):
        fig, ax = plt.subplots(subplot_kw=dict(projection=EuroPP()))
        assert isinstance(ax, GeoAxesSubplot)

        ax.add_feature(
            countries(
                edgecolor="white",
                facecolor="#d9dadb",
                alpha=1,
                linewidth=2,
                zorder=-2,
            )
        )
        # ax.outline_patch.set_visible(False)
        # ax.background_patch.set_visible(False)
        text_style = dict(
            verticalalignment="top",
            horizontalalignment="right",
            fontname="Ubuntu",
            fontsize=18,
            bbox=dict(facecolor="white", alpha=0.6, boxstyle="round"),
        )

        airports[airport].point.plot(
            ax,
            shift=dict(units="dots", x=-15, y=-15),
            marker=atc_tower,
            s=300,
            zorder=5,
            text_kw={**text_style},
        )

        flight.plot(ax)

        for hp in flight.holding_pattern():
            hp.plot(ax, color="#f58518", lw=3)

        ax.set_extent(flight.last(last), buffer=buffer)
        square_ratio(ax)

        return fig


# %%
f = landing_dublin_2019["ICE416_16873"]
fig = plot_holding(f, airport="EIDW", buffer=0.1, last="10 min")
fig.savefig("figures/holding_00.png", dpi=200, bbox_inches="tight")
fig

# %%

f = landing_londoncity_2019["DLH936_034"]
fig = plot_holding(f, airport="EGLC", buffer=0.1, last="20 min")
fig.savefig("figures/holding_01.png", dpi=200, bbox_inches="tight")
fig
# %%
f = landing_londoncity_2019["AZA216_4152"]
fig = plot_holding(f, airport="EGLC", buffer=0.1, last="15 min")
fig.savefig("figures/holding_02.png", dpi=200, bbox_inches="tight")
fig
# %%
f = landing_londoncity_2019["CFE2213_1434"]
fig = plot_holding(f, airport="EGLC", buffer=0.15, last="40 min")
fig.savefig("figures/holding_03.png", dpi=200, bbox_inches="tight")
fig
# %%
f = landing_dublin_2019["AAL722_17460"]
fig = plot_holding(f, airport="EIDW", buffer=0.1, last="10 min")
fig.savefig("figures/holding_04.png", dpi=200, bbox_inches="tight")
fig
# %%
fig = plot_holding(belevingsvlucht, airport="EHLE", buffer=0.1, last="4 h")
fig.savefig("figures/holding_05.png", dpi=200, bbox_inches="tight")
fig
# %%
f = landing_dublin_2019["NAX2NU_8064"]
fig = plot_holding(f, airport="EIDW", buffer=0.1, last="10 min")
fig.savefig("figures/holding_06.png", dpi=200, bbox_inches="tight")
fig

# %%
