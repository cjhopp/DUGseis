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

# Import from the DUGSeis library.
from dug_seis.project.project import DUGSeisProject
from dug_seis.waveform_handler.waveform_handler import FILENAME_REGEX
from dug_seis import util

from dug_seis.event_processing.detection.dug_trigger import dug_trigger
from dug_seis.event_processing.picking.dug_picker import dug_picker
from dug_seis.event_processing.location.locate_homogeneous import (
    locate_in_homogeneous_background_medium,
)

# The logging is optional, but useful.
util.setup_logging_to_file(
    # folder=".",
    # If folder is not specified it will not log to a file but only to stdout.
    folder=None,
    log_level="debug",
)
logger = logging.getLogger(__name__)


def launch_processing(project):
    # Helper function to compute intervals over the project.
    intervals = util.compute_intervals(
        project=project, interval_length_in_seconds=10, interval_overlap_in_seconds=0.1
    )
    total_event_count = 0

    for interval_start, interval_end in tqdm.tqdm(intervals):
        # Run the trigger only on a few waveforms.
        print('Interval: {} {}'.format(interval_start, interval_end))
        st_triggering = project.waveforms.get_waveforms(
            channel_ids=[
                'SV.PDT1..XNZ', 'SV.PDB3..XNZ', 'SV.PDB4..XNZ', 'SV.PDB6..XNZ',
                'SV.PSB7..XNZ', 'SV.PSB9..XNZ', 'SV.PST10..XNZ', 'SV.PST12..XNZ',
                'SV.OB13..XNZ', 'SV.OB15..XNZ', 'SV.OT16..XNZ', 'SV.PDB01..XN1',
                'SV.PDB02..XN1', 'SV.PDB03..XN1', 'SV.PDB04..XN1', 'SV.PDB05..XN1',
                'SV.PDB06..XN1', 'SV.PDB07..XN1', 'SV.PDB08..XN1',
                'SV.PDB09..XN1', 'SV.PDB10..XN1', 'SV.PDB11..XN1', 'SV.PDB12..XN1',
                'SV.OT01..XN1', 'SV.OT02..XN1', 'SV.OT03..XN1', 'SV.OT04..XN1',
                'SV.OT05..XN1', 'SV.OT06..XN1', 'SV.OT07..XN1', 'SV.OT08..XN1',
                'SV.OT09..XN1', 'SV.OT10..XN1', 'SV.OT11..XN1', 'SV.OT12..XN1'
                'SV.CTrig..'
            ],
            start_time=interval_start,
            end_time=interval_end,
        )
        st_triggering.filter('highpass', freq=2000)
        # Standard DUGSeis trigger.
        detected_events = dug_trigger(
            st=st_triggering,
            # Helps with classification.
            active_triggering_channel="SV.CTrig..",
            minimum_time_between_events_in_seconds=0.0006,
            max_spread_electronic_interference_in_seconds=2e-5,
            # Passed on the coincidence trigger.
            conincidence_trigger_opts={
                "trigger_type": "recstalta",
                "thr_on": 6.0,
                "thr_off": 2.0,
                "thr_coincidence_sum": 6,
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
            # Optionally remove the instrument response if necessary.
            # Requires StationXML files where this is possible.
            # st_event.remove_response(inventory=project.inventory, output="VEL")
            st_event.detrend('demean')
            st_event.detrend('linear')
            picks = dug_picker(
                    st=st_event,
                    pick_algorithm="aicd",
                    picker_opts={
                        # Here given as samples.
                        "bandpass_f_min": 2000,
                        "bandpass_f_max": 15000,
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
            if len(picks) < 3:
                # Optionally save the picks to the database as unassociated picks.
                # if picks:
                #    project.db.add_object(picks)
                continue

            event = locate_in_homogeneous_background_medium(
                picks=picks,
                coordinates=project.cartesian_coordinates,
                velocity=5900.0,
                damping=0.01,
                local_to_global_coordinates=project.local_to_global_coordinates,
            )

            # If there is a magnitude determination algorithm this could happen
            # here. Same with a moment tensor inversion. Anything really.

            # Write the classification as a comment.
            event.comments = [
                obspy.core.event.Comment(
                    text=f"Classification: {event_candidate['classification']}"
                )
            ]

            # Could optionally do a QA step here.
            if event.origins[0].time_errors.uncertainty > 5e-4:
                logger.info(
                    "Rejected event. Time error too large: "
                    f"{event.origins[0].time_errors.uncertainty}"
                )
                continue

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


with open("/home/chet/DUGseis/scripts/mock_live_environment/live_processing_Collab4850.yaml", "r") as fh:
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
