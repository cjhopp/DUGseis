# DUGSeis
# Copyright (C) 2021 DUGSeis Authors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Plotting functions for EGS Collab
"""

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec


def plot_3D(locs, boreholes, axes):
    x, y, z = zip(*locs)
    for bh in boreholes:
        axes.plot(bh[0], bh[1], color='k', linewidth=0.8)
    axes.scatter(x, y, z, marker='o', color='magenta')
    return


def plot_magtime(times, mags, axes):
    axes.stem(times, mags)
    ax2 = axes.twinx()
    ax2.step(times, np.arange(len(times)))
    return


def plot_mapview(locs, boreholes, axes):
    # Plot boreholes
    for bh in boreholes:
        axes.plot(bh[0], bh[1], color='k', linewidth=0.8)
    x, y, z = zip(*locs)
    axes.scatter(x, y, z, marker='o', color='magenta')
    return


def plot_all(catalog, boreholes, global_to_local, outfile):
    """
    Plot three panels of basic MEQ information to file

    :param catalog: obspy Catalog from dumped
    :param boreholes:
    :return:
    """
    fig = plt.figure(constrained_layout=False, figsize=(14, 11))
    gs = GridSpec(ncols=14, nrows=11, figure=fig)
    axes_map = fig.add_subplot(gs[:7, :7])
    axes_3D = fig.add_subplot(gs[:7, 7:], projection='3d')
    axes_time = fig.add_subplot(gs[7:, :])
    # Convert to HMC system
    catalog = [ev for ev in catalog if len(ev.origins) > 0]
    locs = [(ev.preferred_origin().latitude,
             ev.preferred_origin().longitude,
             ev.preferred_origin().depth) for ev in catalog if
            ev.comments[-1].text == 'Classification: passive']
    hmc_locs = [global_to_local(latitude=pt[0], longitude=pt[1], depth=pt[2])
                for pt in locs]
    times = [ev.preferred_origin().time for ev in catalog]
    mags = []
    for ev in catalog:
        if len(ev.magnitudes) > 0:
            mags.append(ev.preferred_magnitude().mag)
        else:
            mags.append(None)
    plot_mapview(hmc_locs, boreholes, axes_map)
    plot_3D(hmc_locs, boreholes, axes_3D)
    plot_magtime(times, mags, axes_time)
    plt.savefig(outfile, dpi=300)
    return