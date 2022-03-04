"""
Example live processing script. Very similar to normal processing, just written
to monitor certain directories and process newly incoming data.

Parts of this could be wrapped in other functions - that really depends on how
it will be used in the end.
"""
import copy
import logging
import pathlib
import re
import time

import obspy
import tqdm
import yaml

import numpy as np

from glob import glob

# Despiking imports
from eqcorrscan.utils.despike import template_remove
from eqcorrscan.utils.correlate import CorrelationError
from joblib import Parallel, delayed

# Import from the DUGSeis library.
from dug_seis.project.project import DUGSeisProject
from dug_seis.waveform_handler.waveform_handler import FILENAME_REGEX
from dug_seis import util

from dug_seis.event_processing.detection.dug_trigger import dug_trigger
from dug_seis.event_processing.picking.dug_picker import dug_picker
from dug_seis.event_processing.magnitude.estimate_magnitude import (
    est_magnitude_spectra, est_magnitude_energy
)
from dug_seis.event_processing.location.locate_homogeneous import (
    locate_in_homogeneous_background_medium,
)

# The logging is optional, but useful.
util.setup_logging_to_file(
    # folder=".",
    # If folder is not specified it will not log to a file but only to stdout.
    folder=None,
    log_level="info",
)
logger = logging.getLogger(__name__)

"""
n.b. Bad channels in the Vibbox data (commented out in the lists below):

AML2.XNZ,
AMU1.XNY
DML4.*
DMU4.XNZ
"""

trigger_chans = ['CB.TS02..XDH', 'CB.TS04..XDH', 'CB.TS06..XDH', 'CB.TS08..XDH',
                 'CB.TS10..XDH', 'CB.TS12..XDH', 'CB.TS14..XDH', 'CB.TS16..XDH',
                 'CB.TS18..XDH', 'CB.TS20..XDH', 'CB.TS22..XDH', 'CB.TS24..XDH',
                 'CB.AML1..XNX', 'CB.AML1..XNY', 'CB.AML1..XNZ', 'CB.AML2..XNX',
                 'CB.AML2..XNY', #'CB.AML2..XNZ',
                 'CB.AML3..XNX', 'CB.AML3..XNY',
                 'CB.AML3..XNZ', 'CB.AML4..XNX', 'CB.AML4..XNY', 'CB.AML4..XNZ',
                 'CB.AMU1..XNX', #'CB.AMU1..XNY',
                 'CB.AMU1..XNZ', 'CB.AMU2..XNX',
                 'CB.AMU2..XNY', 'CB.AMU2..XNZ', 'CB.AMU3..XNX', 'CB.AMU3..XNY',
                 'CB.AMU3..XNZ', 'CB.AMU4..XNX', 'CB.AMU4..XNY', 'CB.AMU4..XNZ',
                 'CB.DML1..XNX', 'CB.DML1..XNY', 'CB.DML1..XNZ', 'CB.DML2..XNX',
                 'CB.DML2..XNY', 'CB.DML2..XNZ', 'CB.DML3..XNX', 'CB.DML3..XNY',
                 'CB.DML3..XNZ', #'CB.DML4..XNX', 'CB.DML4..XNY', 'CB.DML4..XNZ',
                 'CB.DMU1..XNX', 'CB.DMU1..XNY', 'CB.DMU1..XNZ', 'CB.DMU2..XNX',
                 'CB.DMU2..XNY', 'CB.DMU2..XNZ', 'CB.DMU3..XNX', 'CB.DMU3..XNY',
                 'CB.DMU3..XNZ', 'CB.DMU4..XNX', 'CB.DMU4..XNY', #'CB.DMU4..XNZ',
                 'CB.CTrig..']

mag_chans = ['CB.AML1..XNX', 'CB.AML1..XNY', 'CB.AML1..XNZ', 'CB.AML2..XNX',
             'CB.AML2..XNY', #'CB.AML2..XNZ',
             'CB.AML3..XNX', 'CB.AML3..XNY',
             'CB.AML3..XNZ', 'CB.AML4..XNX', 'CB.AML4..XNY', 'CB.AML4..XNZ',
             'CB.AMU1..XNX', #'CB.AMU1..XNY',
             'CB.AMU1..XNZ', 'CB.AMU2..XNX',
             'CB.AMU2..XNY', 'CB.AMU2..XNZ', 'CB.AMU3..XNX', 'CB.AMU3..XNY',
             'CB.AMU3..XNZ', 'CB.AMU4..XNX', 'CB.AMU4..XNY', 'CB.AMU4..XNZ',
             'CB.DML1..XNX', 'CB.DML1..XNY', 'CB.DML1..XNZ', 'CB.DML2..XNX',
             'CB.DML2..XNY', 'CB.DML2..XNZ', 'CB.DML3..XNX', 'CB.DML3..XNY',
             'CB.DML3..XNZ', #'CB.DML4..XNX', 'CB.DML4..XNY', 'CB.DML4..XNZ',
             'CB.DMU1..XNX', 'CB.DMU1..XNY', 'CB.DMU1..XNZ', 'CB.DMU2..XNX',
             'CB.DMU2..XNY', 'CB.DMU2..XNZ', 'CB.DMU3..XNX', 'CB.DMU3..XNY',
             'CB.DMU3..XNZ', 'CB.DMU4..XNX', 'CB.DMU4..XNY', #'CB.DMU4..XNZ'
             ]


def despike(tr, temp_streams, cc_thresh):
    """
    Remove ERT spikes using template-based removal
    :return:
    """
    new_tr = tr.copy()
    for tst in temp_streams:
        temp_tr = tst.select(id=tr.id)[0]
        window = int(1.5 * (temp_tr.stats.delta * (temp_tr.stats.npts + 1)))
        try:
            template_remove(new_tr, temp_tr, cc_thresh, windowlength=window,
                            interp_len=window)
        except CorrelationError:
            return new_tr
    return new_tr

def launch_processing(project):
    # Helper function to compute intervals over the project.
    intervals = util.compute_intervals(
        project=project, interval_length_in_seconds=30, interval_overlap_in_seconds=0.1
    )
    total_event_count = 0

    for interval_start, interval_end in tqdm.tqdm(intervals):
        # Run the trigger only on a few waveforms.
        print('Interval: {} {}'.format(interval_start, interval_end))
        st_all = project.waveforms.get_waveforms(
            channel_ids=[
                'CB.TS02..XDH', 'CB.TS04..XDH', 'CB.TS06..XDH', 'CB.TS08..XDH',
                'CB.TS10..XDH', 'CB.TS12..XDH', 'CB.TS14..XDH', 'CB.TS16..XDH',
                'CB.TS18..XDH', 'CB.TS20..XDH', 'CB.TS22..XDH', 'CB.TS24..XDH',
                'CB.AML1..XNX', 'CB.AML1..XNY', 'CB.AML1..XNZ', 'CB.AML2..XNX',
                'CB.AML2..XNY', #'CB.AML2..XNZ',
                'CB.AML3..XNX', 'CB.AML3..XNY',
                'CB.AML3..XNZ', 'CB.AML4..XNX', 'CB.AML4..XNY', 'CB.AML4..XNZ',
                'CB.AMU1..XNX', #'CB.AMU1..XNY',
                'CB.AMU1..XNZ', 'CB.AMU2..XNX',
                'CB.AMU2..XNY', 'CB.AMU2..XNZ', 'CB.AMU3..XNX', 'CB.AMU3..XNY',
                'CB.AMU3..XNZ', 'CB.AMU4..XNX', 'CB.AMU4..XNY', 'CB.AMU4..XNZ',
                'CB.DML1..XNX', 'CB.DML1..XNY', 'CB.DML1..XNZ', 'CB.DML2..XNX',
                'CB.DML2..XNY', 'CB.DML2..XNZ', 'CB.DML3..XNX', 'CB.DML3..XNY',
                'CB.DML3..XNZ', #'CB.DML4..XNX', 'CB.DML4..XNY', 'CB.DML4..XNZ',
                'CB.DMU1..XNX', 'CB.DMU1..XNY', 'CB.DMU1..XNZ', 'CB.DMU2..XNX',
                'CB.DMU2..XNY', 'CB.DMU2..XNZ', 'CB.DMU3..XNX', 'CB.DMU3..XNY',
                'CB.DMU3..XNZ', 'CB.DMU4..XNX', 'CB.DMU4..XNY', #'CB.DMU4..XNZ',
                'CB.CMon..', 'CB.CTrig..', 'CB.CEnc..', 'CB.PPS..'],
            start_time=interval_start,
            end_time=interval_end,
        )
        # Separate triggering and magnitude traces
        st_triggering = obspy.Stream(
            traces=[tr for tr in st_all if tr.id in trigger_chans]).copy()
        # Depike triggering trace
        cc_thresh = 0.7
        spike_streams = [obspy.read(s) for s in glob('{}/*.ms'.format(
            project.config['paths']['spike_mseed']))]
        results = Parallel(n_jobs=15, verbose=10)(
            delayed(despike)(tr, spike_streams, cc_thresh)
            for tr in st_triggering)
        st_triggering = obspy.Stream(traces=[r for r in results])
        # Preprocess
        try:
            st_triggering.detrend('linear')
        except NotImplementedError:  # Case of masked arrays...interpolation?
            for tr in st_triggering:
                if isinstance(tr.data, np.ma.masked_array):
                    tr.data = tr.data.filled()
        st_triggering.detrend('demean')
        st_triggering.filter('highpass', freq=2000)
        st_mags = obspy.Stream(
            traces=[tr for tr in st_all if tr.id in mag_chans]).copy()
        # Standard DUGSeis trigger.
        detected_events = dug_trigger(
            st=st_triggering,
            # Helps with classification.
            active_triggering_channel="CB.CTrig..",
            minimum_time_between_events_in_seconds=0.0006,
            max_spread_electronic_interference_in_seconds=2e-5,
            # Passed on the coincidence trigger.
            conincidence_trigger_opts={
                "trigger_type": "recstalta",
                "thr_on": 6.0,
                "thr_off": 2.0,
                "thr_coincidence_sum": 10,
                # The time windows are given in seconds.
                "sta": 0.0007,
                "lta": 0.007,
                "trigger_off_extension": 0.0,
                "details": True,
            },
        )

        logger.info(
            f"Found {len(detected_events)} event candidates in interval "
            f"{interval_start}-{interval_end}."
        )

        if not detected_events:
            continue

        # Now loop over the detected events.
        added_event_count = 0

        for event_candidate in detected_events:
            # Get the waveforms for the event processing. Note that this could
            # use the same channels as for the initial trigger or different ones.
            st_event = st_triggering.slice(
                starttime=event_candidate["time"] - 3e-3,
                endtime=event_candidate["time"] + 1e-2).copy()
            # Remove the active trigger before picking
            st_event.traces.remove(st_event.select(station='CTrig')[0])
            st_event = st_event.select(channel='XNX')
            st_event.detrend('demean')
            st_event.detrend('linear')
            picks = dug_picker(
                    st=st_event,
                    pick_algorithm="aicd",
                    picker_opts={
                        # Here given as samples.
                        "bandpass_f_min": 2000,
                        "bandpass_f_max": 49000,
                        "t_ma": 0.0003,
                        "nsigma": 4,
                        "t_up": 0.00078,
                        "nr_len": 0.002,
                        "nr_coeff": 2,
                        "pol_len": 50,
                        "pol_coeff": 10,
                        "uncert_coeff": 3,
                        "plot": False
                    },
                )

            # We want at least three picks, otherwise we don't designate it an event.
            if len(picks) < 3 or event_candidate['classification'] == 'electronic':
                # Optionally save the picks to the database as unassociated picks.
                # if picks:
                #    project.db.add_object(picks)
                continue

            event = locate_in_homogeneous_background_medium(
                picks=picks,
                coordinates=project.cartesian_coordinates,
                velocity=6900.0,
                damping=0.01,
                local_to_global_coordinates=project.local_to_global_coordinates,
            )

            # Could optionally do a QA step here.
            if event.origins[0].time_errors.uncertainty > 5e-4:
                logger.info(
                    "Rejected event. Time error too large: "
                    f"{event.origins[0].time_errors.uncertainty}"
                )
                continue

            # Try magnitudes only for passive events
            # if event_candidate['classification'] == 'passive':
            # est_magnitude_spectra(
                # event=event, stream=st_mags,
                # coordinates=project.cartesian_coordinates,
                # global_to_local=project.global_to_local_coordinates,
                # Vp=5900, Vs=3300, p=2900, inventory=project.inventory,
                # plot=True)
            # try:
            #     est_magnitude_energy(
            #         event=event, stream=st_mags,
            #         coordinates=project.cartesian_coordinates,
            #         global_to_local=project.global_to_local_coordinates,
            #         Vs=3300, p=2900, G=20, inventory=project.inventory,
            #         plot=True)
            # except ValueError as e:
            #     logger.info(
            #         "Magnitude calculation failed "
            #         f"{event.origins[0].resource_id}"
            #     )

            # Write the classification as a comment.
            event.comments = [
                obspy.core.event.Comment(
                    text=f"Classification: {event_candidate['classification']}"
                )
            ]

            # Add the event to the project.
            added_event_count += 1
            project.db.add_object(event)
        logger.info(
            f"Successfully located {added_event_count} of "
            f"{len(detected_events)} event(s)."
        )
        total_event_count += added_event_count

        logger.info("DONE.")
        logger.info(f"Found {total_event_count} events.")


with open("/home/chet/DUGseis/scripts/mock_live_environment/live_processing_Collab4100.yaml", "r") as fh:
    yaml_template = yaml.load(fh, Loader=yaml.SafeLoader)

all_folders = [
    pathlib.Path(i).absolute() for i in yaml_template["paths"]["asdf_folders"]
]

ping_interval_in_seconds = 2.5

while True:
    try:
        project = DUGSeisProject(config=copy.deepcopy(yaml_template))
        project.waveforms
    except ValueError as err:
        if err.args[0] == "Could not find any waveform data files.":
            logger.info(
                "No data yet - trying again in "
                f"{ping_interval_in_seconds:.2f} seconds."
            )
            time.sleep(ping_interval_in_seconds)
            continue
        raise err
    break

launch_processing(project=project)

# These files have already been processed.
already_processed_files = set([p.absolute() for p in project.waveforms._files.keys()])

# Monitor the folders and launch the processing again.
while True:
    all_available_files = set()
    for p in all_folders:
        all_available_files = all_available_files.union(p.glob("vbox_*.dat"))

    new_files = all_available_files.difference(already_processed_files)
    if not new_files:
        logger.info(
            "No new files yet - trying again in "
            f"{ping_interval_in_seconds:.2f} seconds."
        )
        time.sleep(ping_interval_in_seconds)
        continue

    starttimes = []
    endtimes = []
    for f in new_files:
        m = re.match(FILENAME_REGEX, f.stem)
        if not m:
            continue
        g = m.groups()
        s = f.stat()

        starttimes.append(obspy.UTCDateTime(*[int(i) for i in g]))
        endtimes.append(starttimes[-1] + 32.)

    config = copy.deepcopy(yaml_template)
    config["temporal_range"]["start_time"] = min(starttimes)
    config["temporal_range"]["end_time"] = max(endtimes)
    # Small grace period for everything to finish copying.
    time.sleep(0.25)
    project = DUGSeisProject(config=config)
    launch_processing(project=project)

    already_processed_files = already_processed_files.union(
        set([p.absolute() for p in project.waveforms._files.keys()])
    )
