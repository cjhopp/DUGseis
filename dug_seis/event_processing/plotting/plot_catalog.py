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

from datetime import datetime, timedelta

from matplotlib.gridspec import GridSpec


def plot_3D(locs, boreholes, colors, mags, axes):
    x, y, z = zip(*locs)
    mag_inds = np.where(np.array(mags) > -999.)
    mags = np.array(mags)[mag_inds]
    for i, bh in enumerate(boreholes):
        bh = np.array(bh)
        if i == 4:
            color = 'indigo'
            linewidth = 1.3
        else:
            color = 'dimgray'
            linewidth = 0.8
        axes.plot(bh[:, 0], bh[:, 1], bh[:, 2], color=color,
                  linewidth=linewidth)
    sizes = (mags + 9)**2
    axes.scatter(np.array(x)[mag_inds], np.array(y)[mag_inds],
                 np.array(z)[mag_inds], marker='o',
                 c=np.array(colors)[mag_inds], s=sizes,
                 alpha=0.7)
    axes.set_xlabel('Easting [HMC]', fontsize=14)
    axes.set_ylabel('Northing [HMC]', fontsize=14)
    axes.set_zlabel('Elevation [m]', fontsize=14)
    axes.set_ylim([-910, -850])
    axes.set_xlim([1210, 1270])
    axes.set_zlim([300, 360])
    axes.view_init(10., -20.)
    return


def plot_magtime(times, mags, axes):
    mag_inds = np.where(np.array(mags) > -999.)
    mags = np.array(mags)[mag_inds]
    mag_times = np.array(times)[mag_inds]
    axes.stem(mag_times, mags, bottom=-10, basefmt='k-')
    ax2 = axes.twinx()
    ax2.step(times, np.arange(len(times)), color='firebrick')
    axes.set_ylim([-10, 0.])
    axes.set_ylabel('Mw', fontsize=14)
    ax2.set_ylabel('Cumulative seismic events', fontsize=14)
    axes.set_xlabel('Time [UTC]', fontsize=14)
    ax2.tick_params(axis='y', colors='firebrick')
    axes.set_xlim([mag_times[-1] - timedelta(seconds=3600), mag_times[-1]])
    return


def plot_mapview(locs, boreholes, colors, mags, axes):
    hull_pts = np.load('/home/sigmav/chet-collab/drift_pts/4100L_xy_alphashape_pts.npy')
    mag_inds = np.where(np.array(mags) > -999.)
    mags = np.array(mags)[mag_inds]
    # Plot boreholes
    for i, bh in enumerate(boreholes):
        bh = np.array(bh)
        if i == 4:
            color = 'indigo'
            linewidth = 1.3
        else:
            color = 'dimgray'
            linewidth = 0.8
        axes.plot(bh[:, 0], bh[:, 1], color=color, linewidth=linewidth)
    x, y, z = zip(*locs)
    sizes = (mags + 9)**2
    axes.scatter(np.array(x)[mag_inds], np.array(y)[mag_inds],
                 marker='o', c=np.array(colors)[mag_inds], s=sizes)
    axes.plot(hull_pts[:, 0], hull_pts[:, 1], linewidth=0.9, color='k')
    axes.set_ylim([-920, -840])
    axes.set_xlim([1200, 1280])
    axes.set_xlabel('Easting [HMC]', fontsize=14)
    axes.set_ylabel('Northing [HMC]', fontsize=14)
    return


def plot_all(catalog, boreholes, global_to_local, outfile):
    """
    Plot three panels of basic MEQ information to file

    :param catalog: obspy Catalog from dumped
    :param boreholes:
    :return:
    """
    fig = plt.figure(constrained_layout=False, figsize=(18, 13))
    fig.suptitle('Realtime MEQ: {}'.format(datetime.now()), fontsize=20)
    gs = GridSpec(ncols=18, nrows=13, figure=fig)
    axes_map = fig.add_subplot(gs[:9, :9])
    axes_3D = fig.add_subplot(gs[:9, 9:], projection='3d')
    axes_time = fig.add_subplot(gs[9:, :])
    # Convert to HMC system
    catalog = [ev for ev in catalog if len(ev.origins) > 0]
    catalog.sort(key=lambda x: x.origins[-1].time)
    endtime = catalog[-1].origins[-1].time
    starttime = endtime - 3600
    cat_time = [ev for ev in catalog if ev.origins[-1].time > starttime]
    locs = [(ev.preferred_origin().latitude,
             ev.preferred_origin().longitude,
             ev.preferred_origin().depth) for ev in catalog]# if
            # ev.comments[-1].text == 'Classification: passive']
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
    plot_mapview(hmc_locs, boreholes, colors, mags, axes_map)
    plot_3D(hmc_locs, boreholes, colors, mags, axes_3D)
    plot_magtime(times, mags, axes_time)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    return