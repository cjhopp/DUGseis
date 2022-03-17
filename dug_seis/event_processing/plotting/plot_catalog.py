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


def plot_3D(locs, boreholes, colors, axes):
    x, y, z = zip(*locs)
    for bh in boreholes:
        bh = np.array(bh)
        axes.plot(bh[:, 0], bh[:, 1], bh[:, 2], color='k', linewidth=0.8)
    axes.scatter(x, y, z, marker='o', c=colors, s=5)
    axes.set_xlabel('Easting [HMC]')
    axes.set_ylabel('Northing [HMC]')
    axes.set_ylim([-920, -840])
    axes.set_xlim([1200, 1280])
    axes.set_zlim([300, 380])
    axes.view_init(-16., 48.)
    return


def plot_magtime(times, mags, axes):
    mag_inds = np.where(np.array(mags) > -999.)
    mags = np.array(mags)[mag_inds]
    mag_times = np.array(times)[mag_inds]
    axes.stem(mag_times, mags, bottom=-10, basefmt='k-')
    ax2 = axes.twinx()
    ax2.step(times, np.arange(len(times)), color='firebrick')
    axes.set_ylim([-10, 0.])
    axes.set_ylabel('Estimated Mw', fontsize=14)
    ax2.set_ylabel('Cumulative seismic events')
    ax2.tick_params(axis='y', colors='firebrick')
    return


def plot_mapview(locs, boreholes, colors, axes):
    # Plot boreholes
    for bh in boreholes:
        bh= np.array(bh)
        axes.plot(bh[:, 0], bh[:, 1], color='k', linewidth=0.8)
    x, y, z = zip(*locs)
    axes.scatter(x, y, marker='o', c=colors, s=5)
    axes.set_ylim([-920, -840])
    axes.set_xlim([1200, 1280])
    axes.set_xlabel('Easting [HMC]')
    axes.set_ylabel('Northing [HMC]')
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
    catalog.sort(key=lambda x: x.origins[-1].time)
    endtime = catalog[-1].origins[-1].time
    starttime = endtime - 7200
    cat_time = [ev for ev in catalog if ev.origins[-1].time > starttime]
    locs = [(ev.preferred_origin().latitude,
             ev.preferred_origin().longitude,
             ev.preferred_origin().depth) for ev in catalog if
            ev.comments[-1].text == 'Classification: passive']
    hmc_locs = [global_to_local(latitude=pt[0], longitude=pt[1], depth=pt[2])
                for pt in locs]
    colors = ['lightgray' if ev.origins[-1].time < starttime else 'dodgerblue'
              for ev in catalog]
    times = [ev.preferred_origin().time.datetime for ev in cat_time]
    mags = []
    for ev in catalog:
        if len(ev.magnitudes) > 0:
            mags.append(ev.preferred_magnitude().mag)
        else:
            mags.append(-999.)
    plot_mapview(hmc_locs, boreholes, colors, axes_map)
    plot_3D(hmc_locs, boreholes, colors, axes_3D)
    plot_magtime(times, mags, axes_time)
    fig.autofmt_xdate()
    plt.savefig(outfile, dpi=300)
    return