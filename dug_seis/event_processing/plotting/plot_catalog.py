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
from matplotlib.dates import date2num

# Zone 7 start/end
z7_start = np.array([1250.02033327, -876.09451028, 328.41040807])
z7_end = np.array([1251.6751636, -874.58967042, 327.82095658])
z7 = np.vstack([z7_start, z7_end])

z1_start = np.array([1239.68287212, -885.63997146,  331.55923477])
z1_end = np.array([1241.42805389, -884.04965582,  330.95011331])
z1 = np.vstack([z1_start, z1_end])

TU_start = (1249.3835222592002, -880.1002573728001, 338.5249367664)
TU_end = (1250.0872899144001, -879.5190214512, 338.470134336)
TU_zone = np.vstack([TU_start, TU_end])


def plot_3D(locs, boreholes, colors, mags, stations, axes):
    x, y, z = zip(*locs)
    sx, sy, sz = zip(*stations)
    mag_inds = np.where(np.array(mags) > -999.)
    mags = np.array(mags)[mag_inds]
    for well, xyzd in boreholes.items():
        if well == 'TU':
            color = 'indigo'
            linewidth = 1.3
        else:
            color = 'dimgray'
            linewidth = 0.8
        axes.plot(xyzd[:, 0], xyzd[:, 1], xyzd[:, 2], color=color,
                  linewidth=linewidth)
    # Plot Zone 1, 7
    axes.plot(z7[:, 0], z7[:, 1], z7[:, 2], color='darkgray', linewidth=2.5)
    axes.plot(z1[:, 0], z1[:, 1], z1[:, 2], color='darkgray', linewidth=2.5)
    axes.plot(TU_zone[:, 0], TU_zone[:, 1], TU_zone[:, 2], color='purple',
              linewidth=5)
    # Stations
    axes.scatter(sx, sy, sz, marker='v', color='r')
    sizes = (mags + 9)**2
    mpl = axes.scatter(
        np.array(x)[mag_inds], np.array(y)[mag_inds],
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


def plot_magtime(times, mags, colors, axes):
    mag_inds = np.where(np.array(mags) != -999.)
    mags = np.array(mags)[mag_inds]
    mag_times = np.array(times)[mag_inds]
    colors = np.array(colors)[mag_inds]
    # axes.stem(mag_times, mags, bottom=-10, basefmt='k-', zorder=-1)
    axes.scatter(mag_times, mags, c=colors)
    ax2 = axes.twinx()
    ax2.step(times, np.arange(len(times)), color='firebrick')
    axes.set_ylim([-10, 0.])
    axes.set_ylabel('Mw', fontsize=14)
    ax2.set_ylabel('Cumulative seismic events', fontsize=14)
    axes.set_xlabel('Time [UTC]', fontsize=14)
    ax2.tick_params(axis='y', colors='firebrick')
    # axes.set_xlim([datetime.utcnow() - timedelta(seconds=3600), datetime.utcnow()])
    return


def plot_mapview(locs, boreholes, colors, mags, stations, axes):
    hull_pts = np.load('/media/chet/data/chet-collab/model/4100L_xy_alphashape_pts.npy')
    mag_inds = np.where(np.array(mags) > -999.)
    mags = np.array(mags)[mag_inds]
    # Plot boreholes
    for well, xyzd in boreholes.items():
        if well == 'TU':
            color = 'indigo'
            linewidth = 1.3
        else:
            color = 'dimgray'
            linewidth = 0.8
        axes.plot(xyzd[:, 0], xyzd[:, 1], color=color,
                  linewidth=linewidth)
    axes.plot(z7[:, 0], z7[:, 1], color='darkgray', linewidth=2.5)
    axes.plot(z1[:, 0], z1[:, 1], color='darkgray', linewidth=2.5)
    # TU Injection Zone too
    axes.plot(TU_zone[:, 0], TU_zone[:, 1], color='purple',
              linewidth=2.5)
    # Stations
    sx, sy, sz = zip(*stations)
    axes.scatter(sx, sy, marker='v', color='r')
    x, y, z = zip(*locs)
    sizes = (mags + 9)**2
    axes.scatter(np.array(x)[mag_inds], np.array(y)[mag_inds],
                 marker='o', c=np.array(colors)[mag_inds], s=sizes)
    axes.plot(hull_pts[0, :], hull_pts[1, :], linewidth=0.9, color='k')
    axes.set_ylim([-920, -840])
    axes.set_xlim([1200, 1280])
    axes.set_xlabel('Easting [HMC]', fontsize=14)
    axes.set_ylabel('Northing [HMC]', fontsize=14)
    return


def plot_all(catalog, boreholes, global_to_local, inventory,
             plot_distance=False):
    """
    Plot three panels of basic MEQ information to file

    :param catalog: obspy Catalog from dumped
    :param boreholes:
    :return:
    """
    fig = plt.figure(constrained_layout=False, figsize=(18, 13))
    fig.suptitle('Realtime MEQ: {} UTC'.format(datetime.utcnow()), fontsize=20)
    gs = GridSpec(ncols=18, nrows=13, figure=fig)
    axes_map = fig.add_subplot(gs[:9, :9])
    axes_3D = fig.add_subplot(gs[:9, 9:], projection='3d')
    if plot_distance:
        axes_time = fig.add_subplot(gs[9:11, :])
        axd = fig.add_subplot(gs[11:, :])
    else:
        axes_time = fig.add_subplot(gs[9:, :])
    # Convert to HMC system
    catalog = [ev for ev in catalog if len(ev.origins) > 0]
    catalog.sort(key=lambda x: x.origins[-1].time)
    endtime = datetime.utcnow()
    starttime = endtime - timedelta(seconds=3600)
    stations = [(float(sta.extra.hmc_east.value) * 0.3048,
                 float(sta.extra.hmc_north.value) * 0.3048,
                 float(sta.extra.hmc_elev.value))
                for sta in inventory[0] if sta.code[-2] != 'S']
    try:
        hmc_locs = [(float(ev.preferred_origin().extra.hmc_east.value),
                     float(ev.preferred_origin().extra.hmc_north.value),
                     float(ev.preferred_origin().extra.hmc_elev.value))
                    for ev in catalog]
    except AttributeError:
        hmc_locs = [global_to_local(point=pt) for pt in locs]
    colors = [date2num(ev.picks[0].time) for ev in catalog]
    times = [ev.preferred_origin().time.datetime for ev in catalog]
    with open('meq_locations.csv', 'w') as f:
        for i, l in enumerate(hmc_locs):
            f.write('{},{},{},{}\n'.format(times[i], l[0] / 0.3048,
                                           l[1] / 0.3048, l[2] / 0.3048))
    mags = []
    for ev in catalog:
        if len(ev.magnitudes) > 0:
            mags.append(ev.preferred_magnitude().mag)
        else:
            mags.append(-999.)
    if plot_distance:
        dists = [np.sqrt((l[0] - TU_start[0])**2 +
                         (l[1] - TU_start[1])**2 +
                         (l[2] - TU_start[2])**2)
                 for l in hmc_locs]
        distance = np.array(dists)
        axd.scatter(times, distance, marker='s', s=2.)
        axd.set_ylabel('Dist to injection [m]')
        axd.set_xlabel('Date')
    else:
        dists = None
    plot_mapview(hmc_locs, boreholes, colors, mags, stations, axes_map)
    plot_3D(hmc_locs, boreholes, colors, mags, stations, axes_3D)
    plot_magtime(times, mags, colors, axes_time)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()
    return


def plot_catalog_compare(catalogs, boreholes, inventory):
    """
    Plot location differences between two catalogs

    :param catalogs:
    :param boreholes:
    :param inventory:
    :return:
    """
    fig = plt.figure(constrained_layout=False, figsize=(18, 13))
    fig.suptitle('MEQ Catalog Comparison'.format(datetime.utcnow()),
                 fontsize=20)
    gs = GridSpec(ncols=18, nrows=9, figure=fig)
    axes_map = fig.add_subplot(gs[:9, :9])
    axes_3D = fig.add_subplot(gs[:9, 9:], projection='3d')
    stations = [(float(sta.extra.hmc_east.value) * 0.3048,
                 float(sta.extra.hmc_north.value) * 0.3048,
                 float(sta.extra.hmc_elev.value))
                for sta in inventory[0] if sta.code[-2] != 'S']
    locs_list = []
    for cat in catalogs:
        hmc_locs = [(float(ev.preferred_origin().extra.hmc_east.value),
                     float(ev.preferred_origin().extra.hmc_north.value),
                     float(ev.preferred_origin().extra.hmc_elev.value))
                    for ev in cat]
        locs_list.append(hmc_locs)
    # Mapview
    hull_pts = np.load('/media/chet/data/chet-collab/model/4100L_xy_alphashape_pts.npy')
    # Plot boreholes
    for well, xyzd in boreholes.items():
        if well == 'TU':
            color = 'indigo'
            linewidth = 1.3
        else:
            color = 'dimgray'
            linewidth = 0.8
        axes_map.plot(xyzd[:, 0], xyzd[:, 1], color=color,
                      linewidth=linewidth)
        axes_3D.plot(xyzd[:, 0], xyzd[:, 1], xyzd[:, 2], color=color,
                     linewidth=linewidth)
    # TU Injection Zone too
    axes_map.plot(TU_zone[:, 0], TU_zone[:, 1], color='purple',
                  linewidth=2.5)
    axes_3D.plot(TU_zone[:, 0], TU_zone[:, 1], TU_zone[:, 2], color='purple',
                 linewidth=2.5)
    # Stations
    sx, sy, sz = zip(*stations)
    axes_map.scatter(sx, sy, marker='v', color='r')
    axes_3D.scatter(sx, sy, sz, marker='v', color='r')
    for loc1, loc2 in zip(locs_list[0], locs_list[1]):
        axes_3D.scatter(loc1[0], loc1[1], loc1[2],
                        marker='o', facecolors='none', edgecolors='b')
        axes_3D.scatter(loc2[0], loc2[1], loc2[2],
                        marker='o', facecolors='none', edgecolors='r')
        axes_3D.plot([loc1[0], loc2[0]], [loc1[1], loc2[1]], [loc1[2], loc2[2]],
                     color='r')
        axes_map.scatter(loc1[0], loc1[1], marker='o', facecolors='none',
                         edgecolors='b')
        axes_map.scatter(loc2[0], loc2[1], marker='o', facecolors='none',
                         edgecolors='r')
        axes_map.plot([loc1[0], loc2[0]], [loc1[1], loc2[1]], color='r')
    axes_map.plot(hull_pts[0, :], hull_pts[1, :], linewidth=0.9, color='k')
    axes_map.set_ylim([-920, -840])
    axes_map.set_xlim([1200, 1280])
    axes_map.set_xlabel('Easting [HMC]', fontsize=14)
    axes_map.set_ylabel('Northing [HMC]', fontsize=14)
    axes_map.set_aspect('equal')
    # 3D
    axes_3D.set_xlabel('Easting [HMC]', fontsize=14)
    axes_3D.set_ylabel('Northing [HMC]', fontsize=14)
    axes_3D.set_zlabel('Elevation [m]', fontsize=14)
    axes_3D.set_ylim([-910, -850])
    axes_3D.set_xlim([1210, 1270])
    axes_3D.set_zlim([300, 360])
    axes_3D.view_init(10., -20.)
    plt.show()
    return