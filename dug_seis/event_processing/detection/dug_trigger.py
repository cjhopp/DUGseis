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
Central triggering routine within DUGSeis.
"""

import os
import obspy
import typing

import numpy as np
import matplotlib.pyplot as plt

from .coincidence_trigger import coincidence_trigger


def dug_trigger(
    st: obspy.Stream,
    active_triggering_channel: typing.Optional[str],
    minimum_time_between_events_in_seconds: float,
    max_spread_electronic_interference_in_seconds: float,
    conincidence_trigger_opts: typing.Dict,
    plot: bool,
):
    """
    DUGSeis triggering routine.

    This is largely a wrapper around the coincidence trigger with a few more QA
    steps and some classification heuristics.

    Args:
        st: ObsPy Stream objects with the waveform data.
        active_triggering_channel: Id of the active triggering channel. If
            this channel was amongst the triggering ones, the event will be
            classified as active.
        minimum_time_between_events_in_seconds: Don't allow two events too close
            in time.
        max_spread_electronic_interference_in_seconds: The maximum time between
            the first and the last moment of triggering at different stations
            for which this event is classified as electronic interference.
            Usually set to 0.25e-2.
        coincidence_trigger_opts: Keyword arguments passed on to the coincidence
            trigger.
    """
    triggers = coincidence_trigger(stream=st, **conincidence_trigger_opts)
    # Debug plotting option
    if plot:
        trig_stream = obspy.Stream()
        for tr in st:
            trig_stream += tr.copy().trigger(
                type='recstalta',
                nsta=int(conincidence_trigger_opts['sta'] * tr.stats.sampling_rate),
                nlta=int(conincidence_trigger_opts['lta'] * tr.stats.sampling_rate))
        opts = conincidence_trigger_opts
        triggers = coincidence_trigger(
            trigger_type=None, stream=trig_stream, thr_on=opts['thr_on'],
            thr_off=opts['thr_off'],
            thr_coincidence_sum=opts['thr_coincidence_sum'],
            details=True, trigger_off_extension=opts['trigger_off_extension'])
        plot_triggers(triggers=triggers, st=st, cft_stream=trig_stream,
                      params=conincidence_trigger_opts, outdir=plot)
    events = []
    for trig in triggers:
        event = {"time": min(trig["time"]), "triggered_channels": trig["trace_ids"]}
        # Too close to previous event.
        if (
            events
            and abs(events[-1]["time"] - event["time"])
            < minimum_time_between_events_in_seconds
        ):
            continue

        # Classification.
        # Triggered on active triggering channel == active
        if active_triggering_channel and active_triggering_channel in trig["trace_ids"]:
            classification = "active"
        # Single input trace == passive
        elif len(st) == 1:
            classification = "passive"
        # Spread too small == electronic noise.
        elif (
            max(trig["time"]) - min(trig["time"])
        ) < max_spread_electronic_interference_in_seconds:
            classification = "electronic"
        # Otherwise just treat is as passive.
        else:
            classification = "passive"

        event["classification"] = classification
        events.append(event)

    return events


def plot_triggers(triggers, st, cft_stream, params, outdir):
    """Helper to plot triggers, traces and characteristic funcs"""
    print(triggers)
    for trig in triggers:
        seeds = trig['trace_ids']
        # Clip around trigger time
        st_slice = st.slice(starttime=trig['time'][0] - 0.003,
                            endtime=trig['time'][0] + 0.02)
        cft_slice = cft_stream.slice(starttime=trig['time'][0] - 0.003,
                                     endtime=trig['time'][0] + 0.02)
        fig, ax = plt.subplots(nrows=len(seeds), sharex='col',
                               figsize=(6, len(seeds) / 2.))
        fig.suptitle('Detection: {}'.format(trig['time'][0]))
        fig.subplots_adjust(hspace=0.)
        for i, sid in enumerate(seeds):
            tr_raw = st_slice.select(id=sid)[0]
            tr_cft= cft_slice.select(id=sid)[0].data
            time_vect = np.arange(tr_cft.shape[0]) * tr_raw.stats.delta
            ax[i].plot(time_vect,
                       tr_raw.data / np.max(tr_raw.data) * 0.6 * np.max(tr_cft),
                       color='k')
            ax[i].plot(time_vect, tr_cft.data, color='gray')
            ax[i].axhline(params['thr_on'], linestyle='--', color='r')
            ax[i].axhline(params['thr_off'], linestyle='--', color='b')
            bbox_props = dict(boxstyle="round,pad=0.2", fc="white",
                              ec="k", lw=1)
            ax[i].annotate(text=sid, xy=(0.0, 0.8), xycoords='axes fraction',
                           bbox=bbox_props, ha='center', fontsize=8)
            ax[i].set_yticks([])
        ax[i].set_xlabel('Time [s]', fontsize=12)
        if os.path.isdir(outdir):
            plt.savefig('{}/Trig_{}.png'.format(outdir, trig['time'][0]),
                        dpi=200)
            plt.close('all')
        else:
            plt.show()
    return
