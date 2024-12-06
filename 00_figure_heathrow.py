# %%
import random

import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
from cartes.utils.features import ocean
from cartopy.crs import OSGB, PlateCarree
from cartopy.feature import ShapelyFeature
from geopandas import GeoDataFrame
from vega_datasets import data

import numpy as np
from pyproj import Geod
from shapely.geometry import Polygon
from traffic.data import airports, navaids
from traffic.data.datasets import holding_EGLL

# %%

subset = holding_EGLL[random.sample(holding_EGLL.flight_ids[:11000], 600)]

# %%

boroughs = GeoDataFrame.from_file(data.londonBoroughs.url)
heathrow = airports["EGLL"]
assert heathrow is not None
geod = Geod(ellps="WGS84")

lon_, lat_, back_ = geod.fwd(
    heathrow.point.longitude * np.ones(360),
    heathrow.point.latitude * np.ones(360),
    np.arange(0, 360),
    1852 * 40 * np.ones(360),
)

asma = ShapelyFeature([Polygon(list(zip(lon_, lat_)))], crs=PlateCarree())

countries_10 = GeoDataFrame.from_file(
    shpreader.natural_earth(
        resolution="10m",
        category="cultural",
        name="admin_0_countries",
    )
)
gb = ShapelyFeature(
    countries_10.loc[
        countries_10.intersects(next(asma.geometries()))
    ].geometry.to_list(),
    crs=PlateCarree(),
)
gb_asma = ShapelyFeature(
    countries_10.loc[countries_10.intersects(next(asma.geometries()))]
    .intersection(next(asma.geometries()))
    .to_list(),
    crs=PlateCarree(),
)

box_params = dict(
    boxstyle="round", edgecolor="none", facecolor="white", alpha=0.7, zorder=5
)
nl = navaids.extent("Greater London area")
b = ShapelyFeature(boroughs.geometry.to_list(), crs=PlateCarree())

# %%

dpi = 100
scale = dpi // 100

with plt.style.context("traffic"):
    fig, ax = plt.subplots(
        figsize=(10, 10), dpi=dpi, subplot_kw=dict(projection=OSGB())
    )

    ax.add_feature(asma, edgecolor="#79706e", facecolor="none", linewidth=2)
    ax.add_feature(gb, edgecolor="#bab0ac", facecolor="none", linewidth=0.5)
    ax.add_feature(gb_asma, edgecolor="#79706e", facecolor="none", linewidth=1)
    ax.add_feature(ocean(scale="10m", alpha=0.1))  # type: ignore
    ax.add_feature(b, edgecolor="#79706e", facecolor="none", alpha=0.5)
    subset.plot(ax, color="#bab0ac", alpha=0.07, linewidth=0.7)

    heathrow.plot(ax, footprint=False, runways=dict(linewidth=1))
    ax.text(
        heathrow.point.longitude,
        heathrow.point.latitude + 0.03,
        "London Heathrow airport",
        ha="center",
        va="bottom",
        font="Ubuntu",
        fontsize=14,
        transform=PlateCarree(),
        bbox=dict(
            boxstyle="round",
            zorder=5,
            edgecolor="none",
            facecolor="white",
            alpha=0.7,
        ),
    )

    for flight in subset:
        for segment in flight.holding_pattern():
            segment.plot(ax, color="#f58518", alpha=0.1)

    nl["OCK"].plot(  # type:ignore
        ax,
        marker="^",
        shift=dict(units="dots", x=-20 * scale, y=-20 * scale),
        text_kw=dict(
            ha="right",
            va="top",
            fontname="B612",
            fontsize=15,
            bbox=box_params,
        ),
    )
    nl["BIG"].plot(  # type:ignore
        ax,
        marker="^",
        shift=dict(units="dots", x=-20 * scale, y=-20 * scale),
        text_kw=dict(
            ha="right",
            va="top",
            fontname="B612",
            fontsize=15,
            bbox=box_params,
        ),
    )
    nl["LAM"].plot(  # type:ignore
        ax,
        marker="^",
        shift=dict(units="dots", x=20 * scale, y=20 * scale),
        text_kw=dict(fontname="B612", fontsize=15, bbox=box_params),
    )
    nl["BNN"].plot(  # type:ignore
        ax,
        marker="^",
        shift=dict(units="dots", x=20 * scale, y=20 * scale),
        text_kw=dict(fontname="B612", fontsize=15, bbox=box_params),
    )
    ax.set_extent((-1.528, 0.605, 50.805, 52.137))
    fig.savefig("../figures/heathrow.png", dpi=dpi, transparent=False)

# %%
