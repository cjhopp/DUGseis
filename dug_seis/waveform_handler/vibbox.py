#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 11:43:55 2018

@author: schoenball

version history:
2018/06/13: first version
2018/11/11: v2: changed channel naming to be in line with miniSEED Guide
2018/12/03: improved the vibbox_checktriggers routine to check if CASSM trigger is contained in stream
            added vibbox_remove_ert_triggers() to remove ERT noise 
2018/12/06: Remove CASSM shorts already in vibbox_trigger; remove it from vibbox_checktriggers
2019/04/16: Corrected the factor scaling the signal by 2**31, propagated change to time signal 
            detection and recognizing CASSM triggers
"""
import os
import pyasdf
import yaml

import numpy as np

from scipy.stats import median_absolute_deviation
from obspy import Stream, Trace, UTCDateTime
from obspy.core.trace import Stats

import matplotlib.pyplot as plt


def vibbox_read(fname, seeds, debug=0):
    network, stations, locations, channels = zip(*[s.split('.') for s in seeds])
    network = network[0]
    # Find channel PPS (pulse per second)
    try:
        clock_channel = np.where(np.array(stations) == 'PPS')[0][0]
    except IndexError:
        print('No PPS channel in file. Not reading')
        return
    # TODO Everything from here to file open should go in config?
    HEADER_SIZE=4
    HEADER_OFFSET=27
    DATA_OFFSET=148
    VOLTAGE_RANGE=10  # +/- Volts
    with open(fname, "rb") as f:
        f.seek(HEADER_OFFSET, os.SEEK_SET)
        # read header
        H = np.fromfile(f, dtype=np.uint32, count=HEADER_SIZE)
        BUFFER_SIZE=H[0]
        FREQUENCY=H[1]
        NUM_OF_BUFFERS=H[2]
        no_channels=H[3]
        # read data
        f.seek(DATA_OFFSET, os.SEEK_SET)
        A = np.fromfile(f, dtype=np.uint32,
                        count=BUFFER_SIZE * NUM_OF_BUFFERS)
        A = A.reshape(int(len(A) / no_channels), no_channels)
    # Sanity check on number of channels provided in yaml
    if len(channels) != no_channels:
        print('Number of channels in config file not equal to number in data')
        return
    A = A / 2**32  # Norm to 32-bit
    A *= (2 * VOLTAGE_RANGE)
    A -= VOLTAGE_RANGE  # Demean
    path, fname = os.path.split(fname)
    try:
        # Use derivative of PPS signal to find pulse start
        dt = np.diff(A[:, clock_channel])
        # Use 70 * MAD threshold
        samp_to_first_full_second = np.where(
            dt > np.mean(dt) + 30 * median_absolute_deviation(dt))[0][0]
        # Condition where PPS not recorded properly
        if samp_to_first_full_second > 101000:
            print('Cannot read time signal')
            return
        # If we start during the time pulse, use end of pulse for timing
        if samp_to_first_full_second > 90000:
            print('Start of data is during time pulse. Using end of pulse.')
            # Negative dt
            samp_to_first_full_second = np.where(
                dt < np.mean(dt) - 30 *
                median_absolute_deviation(dt))[0][0] + 90000
        if debug > 0:
            fig, ax = plt.subplots()
            ax.plot(dt, color='r')
            ax.plot(A[:, clock_channel], color='k')
            ax.axhline(y=np.mean(dt) + 30 * median_absolute_deviation(dt),
                       color='magenta', linestyle='--')
            ax.axvline(x=samp_to_first_full_second, color='magenta',
                       linestyle='--')
            fig.text(x=0.75, y=0.75, s=samp_to_first_full_second,
                     fontsize=14)
            plt.show()
        starttime = UTCDateTime(
            np.int(fname[5:9]), np.int(fname[9:11]), np.int(fname[11:13]),
            np.int(fname[13:15]), np.int(fname[15:17]), np.int(fname[17:19]),
            np.int(1e6 * (1 - (np.float(samp_to_first_full_second) /
                               FREQUENCY))))
    except Exception as e:
        print(e)
        print('Cannot read exact time signal: ' + fname +
              '. Taking an approximate one instead')
        starttime = UTCDateTime(
            np.int(fname[5:9]), np.int(fname[9:11]), np.int(fname[11:13]),
            np.int(fname[13:15]), np.int(fname[15:17]), np.int(fname[17:19]),
            np.int(1e2 * np.int(fname[19:23])))
    # arrange it in an obspy stream
    st = Stream()
    for i, sta in enumerate(stations):
        stats = Stats()
        # stats.sampling_rate = round(H[1], 1)
        stats.delta = 1. / H[1]
        stats.npts = A.shape[0]
        stats.network = network
        stats.station = sta
        stats.channel = channels[i]
        stats.location = locations[i]
        stats.starttime = starttime
        # Create new array to avoid non-contiguous warning in obspy.core.mseed
        st.traces.append(Trace(data=np.array(A[:, i]), header=stats))
    return st


def vibbox_to_asdf(files, inv, param_file):
    """
    Convert a list of vibbox files to ASDF files of the same name

    :param files: List of files to convert
    :param inventory: Inventory object to add to asdf files
    :param param_file: path to yaml config file for DUG-seis

    :return:
    """
    # Load in the parameters
    with open(param_file, 'r') as f:
        param = yaml.load(f, Loader=yaml.FullLoader)
    outdir = param['Acquisition']['asdf_settings']['data_folder']
    for afile in files:
        name = os.path.join(outdir, afile.split('/')[-2],
                            afile.split('/')[-1].replace('.dat', '.h5'))
        if not os.path.isdir(os.path.dirname(name)):
            os.mkdir(os.path.dirname(name))
        print('Writing {} to {}'.format(afile, name))
        st = vibbox_read(afile, param)
        with pyasdf.ASDFDataSet(name, compression='gzip-3') as asdf:
            asdf.add_stationxml(inv)
            asdf.add_waveforms(st, tag='raw_recording')
    return