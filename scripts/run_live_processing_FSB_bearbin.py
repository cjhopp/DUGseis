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
import gc
import time
import shutil
import psutil
import tracemalloc

import obspy
import tqdm
import yaml

import numpy as np

from glob import glob
from tempfile import mkstemp
from datetime import datetime

# Denoise import
from eqcorrscan.utils import clustering
# Despiking imports
from eqcorrscan.utils.timer import Timer
from eqcorrscan.utils.correlate import get_array_xcorr
from eqcorrscan.utils.findpeaks import find_peaks2_short
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
from dug_seis.event_processing.plotting.plot_catalog import plot_all

from sqlite3 import OperationalError

# The logging is optional, but useful.
util.setup_logging_to_file(
    folder="/global/home/users/chopp/chet-FS-B/dug_seis/2021/logs/",
    # If folder is not specified it will not log to a file but only to stdout.
    # folder=None,
    log_level="info",
)
logger = logging.getLogger(__name__)

FILENAME_REGEX = re.compile(
    r"""
^                                                              # Beginning of string
vbox_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{4})         # System creation time as capture groups
.*                                                             # Rest of name
$                                                              # End of string.
""",
    re.VERBOSE,
)

"""
n.b. Bad channels in the Vibbox data (commented out in the lists below):

AML2.XNZ,
AMU1.XNY
DML4.*
DMU4.XNZ
"""

trigger_chans = ['FS.B81..XN1', 'FS.B82..XN1', 'FS.B83..XN1', 'FS.B91..XN1', 'FS.CTrg..']

mag_chans = [
    'FS.B31..XNZ', 'FS.B31..XNX', 'FS.B31..XNY', 'FS.B32..XNZ', 'FS.B32..XNX', 'FS.B32..XNY', 'FS.B42..XNZ',
    'FS.B42..XNX', 'FS.B42..XNY', 'FS.B43..XNZ', 'FS.B43..XNX', 'FS.B43..XNY', 'FS.B551..XNZ', 'FS.B551..XNX',
    'FS.B551..XNY', 'FS.B585..XNZ', 'FS.B585..XNX', 'FS.B585..XNY', 'FS.B647..XNZ', 'FS.B647..XNX', 'FS.B647..XNY',
    'FS.B659..XNZ', 'FS.B659..XNX', 'FS.B659..XNY', 'FS.B748..XNZ', 'FS.B748..XNX', 'FS.B748..XNY', 'FS.B75..XNZ',
    'FS.B75..XNX', 'FS.B75..XNY', 'FS.B301..XN1', 'FS.B303..XN1', 'FS.B305..XN1', 'FS.B307..XN1', 'FS.B309..XN1',
    'FS.B310..XN1', 'FS.B311..XN1', 'FS.B312..XN1', 'FS.B314..XN1', 'FS.B316..XN1', 'FS.B318..XN1', 'FS.B320..XN1',
    'FS.B322..XN1', 'FS.B401..XN1', 'FS.B403..XN1', 'FS.B405..XN1', 'FS.B407..XN1', 'FS.B409..XN1', 'FS.B410..XN1',
    'FS.B411..XN1', 'FS.B412..XN1', 'FS.B414..XN1', 'FS.B416..XN1', 'FS.B418..XN1', 'FS.B420..XN1', 'FS.B422..XN1',
    'FS.CTrg..', 'FS.CEnc..', 'FS.PPS..', 'FS.CMon..', 'FS.B81..XN1', 'FS.B82..XN1', 'FS.B83..XN1', 'FS.B91..XN1']

def edit_config(config):
    """Edit config file start time to current time"""
    fd, abspath = mkstemp()
    with open(fd, 'w') as f1:
        with open(config, 'r') as f2:
            for ln in f2:
                if ln.startswith('  start_time'):
                    f1.write('  start_time: {}\n'.format(obspy.UTCDateTime().now()))
                else:
                    f1.write(ln)
    shutil.move(abspath, config)
    return


def _interp_gap(data, peak_loc, interp_len, mad, mean):
    """
    Internal function for filling gap with linear interpolation

    :type data: numpy.ndarray
    :param data: data to remove peak in
    :type peak_loc: int
    :param peak_loc: peak location position
    :type interp_len: int
    :param interp_len: window to interpolate
    :param mad: MAD of trace
    :param mean: mean of trace

    :returns: Trace works in-place
    :rtype: :class:`obspy.core.trace.Trace`
    """
    start_loc = peak_loc - int(0.5 * interp_len)
    end_loc = peak_loc + int(0.5 * interp_len)
    if start_loc < 0:
        start_loc = 0
    if end_loc > len(data) - 1:
        end_loc = len(data) - 1
    # fill = np.ones(end_loc - start_loc) * ((data[end_loc] + data[start_loc]) / 2)
    fill = np.linspace(data[start_loc], data[end_loc], end_loc - start_loc)
    # Fill with noise
    fill += np.random.normal(0, mad, fill.shape)
    data[start_loc:end_loc] = fill
    return data


def template_remove(tr, template, cc_thresh, windowlength, interp_len, mad, mean):
    """
    Looks for instances of template in the trace and removes the matches.

    :type tr: obspy.core.trace.Trace
    :param tr: Trace to remove spikes from.
    :type template: osbpy.core.trace.Trace
    :param template: Spike template to look for in data.
    :type cc_thresh: float
    :param cc_thresh: Cross-correlation threshold (-1 - 1).
    :type windowlength: float
    :param windowlength: Length of window to look for spikes in in seconds.
    :type interp_len: float
    :param interp_len: Window length to remove and fill in seconds.
    :param mad: Median absolute deviation of the trace
    :param mean: Mean of trace

    :returns: tr, works in place.
    :rtype: :class:`obspy.core.trace.Trace`
    """
    _interp_len = int(tr.stats.sampling_rate * interp_len)
    if _interp_len < len(template.data):
        logger.warning('Interp_len is less than the length of the template, '
                       'will used the length of the template!')
        _interp_len = len(template.data)
    if isinstance(template, obspy.Trace):
        template = np.array([template.data])
    with Timer() as t:
        normxcorr = get_array_xcorr("fftw")
        cc, _ = normxcorr(stream=tr.data.astype(np.float32),
                          templates=template.astype(np.float32), pads=[0])
        peaks = find_peaks2_short(
            arr=cc.flatten(), thresh=cc_thresh,
            trig_int=windowlength * tr.stats.sampling_rate)
        for peak in peaks:
            tr.data = _interp_gap(
                data=tr.data, peak_loc=peak[1] + int(0.5 * _interp_len),
                interp_len=_interp_len, mad=mad, mean=mean)
    logger.info("Despiking took: {0:.4f} s".format(t.secs))
    return tr


def despike(tr, temp_streams, cc_thresh):
    """
    Remove ERT spikes using template-based removal
    :return:
    """
    new_tr = tr.copy()
    mean = np.mean(tr.data)
    mad = np.median(np.abs(tr.data - np.mean(tr.data)))
    for tst in temp_streams:
        temp_tr = tst.select(id=tr.id)[0]
        window = int(1.5 * (temp_tr.stats.delta * (temp_tr.stats.npts + 1)))
        try:
            template_remove(new_tr, temp_tr, cc_thresh, windowlength=window,
                            interp_len=window, mad=mad, mean=mean)
        except (CorrelationError, AssertionError):
            return new_tr
    return new_tr


def denoise(st):
    # Do SVD denoising to "remove" 50 Hz electrical and switching noise
    stream_list = [st.select(station=tr.stats.station).copy()
                   for tr in st if tr.stats.station != 'CTrg']
    for strm in stream_list:
        strm[0].stats.station = 'XXX'
    u, s, v, stachans = clustering.svd(stream_list=stream_list,
                                       full=False)
    # Reweight the first singular vector
    noise_vect = np.dot(u[0][:, 0] * s[0][0], v[0][0, 0])
    for tr in st:
        if tr.stats.station != 'CTrg':
            tr.data -= noise_vect
    return st


def launch_processing(project):
    # Helper function to compute intervals over the project.
    intervals = util.compute_intervals(
        project=project, interval_length_in_seconds=30, interval_overlap_in_seconds=0.1
    )
    total_event_count = 0

    templates = glob('{}/*.ms'.format(project.config['paths']['spike_mseed']))
    templates.sort()
    spike_streams = [obspy.read(s) for s in templates]

    for interval_start, interval_end in tqdm.tqdm(intervals):
        # Run the trigger only on a few waveforms.
        print('Interval: {} {}'.format(interval_start, interval_end))
        # Catch file that isn't fully written
        try:
            st_all = project.waveforms.get_waveforms(
                channel_ids=[
                    'FS.B31..XNZ', 'FS.B31..XNX', 'FS.B31..XNY', 'FS.B32..XNZ', 'FS.B32..XNX', 'FS.B32..XNY',
                    'FS.B42..XNZ', 'FS.B42..XNX', 'FS.B42..XNY', 'FS.B43..XNZ', 'FS.B43..XNX', 'FS.B43..XNY',
                    'FS.B551..XNZ', 'FS.B551..XNX', 'FS.B551..XNY', 'FS.B585..XNZ', 'FS.B585..XNX', 'FS.B585..XNY',
                    'FS.B647..XNZ', 'FS.B647..XNX', 'FS.B647..XNY', 'FS.B659..XNZ', 'FS.B659..XNX', 'FS.B659..XNY',
                    'FS.B748..XNZ', 'FS.B748..XNX', 'FS.B748..XNY', 'FS.B75..XNZ', 'FS.B75..XNX', 'FS.B75..XNY',
                    'FS.B301..XN1', 'FS.B303..XN1', 'FS.B305..XN1', 'FS.B307..XN1', 'FS.B309..XN1', 'FS.B310..XN1',
                    'FS.B311..XN1', 'FS.B312..XN1', 'FS.B314..XN1', 'FS.B316..XN1', 'FS.B318..XN1', 'FS.B320..XN1',
                    'FS.B322..XN1', 'FS.B401..XN1', 'FS.B403..XN1', 'FS.B405..XN1', 'FS.B407..XN1', 'FS.B409..XN1',
                    'FS.B410..XN1', 'FS.B411..XN1', 'FS.B412..XN1', 'FS.B414..XN1', 'FS.B416..XN1', 'FS.B418..XN1',
                    'FS.B420..XN1', 'FS.B422..XN1', 'FS.CTrg..', 'FS.CEnc..', 'FS.PPS..', 'FS.CMon..', 'FS.B81..XN1',
                    'FS.B82..XN1', 'FS.B83..XN1', 'FS.B91..XN1'],
                start_time=interval_start,
                end_time=interval_end,
            )
        except ValueError as e:
            print('No waveform data. Next time span')
            continue
        if len(st_all) == 0:
            print('No waveform data found')
            continue
        for tr in st_all:
            if isinstance(tr.data, np.ma.masked_array):
                tr.data = tr.data.filled(fill_value=tr.data.mean())
        # Separate triggering and magnitude traces
        st_mags = obspy.Stream(
            traces=[tr for tr in st_all if tr.id in mag_chans]).copy()
        st_triggering = obspy.Stream(
            traces=[tr for tr in st_all if tr.id in trigger_chans]).copy()

        st_triggering = denoise(st_triggering)
        # Preprocess
        print('Preprocessing')
        print('Demasking')
        for tr in st_triggering:
            if isinstance(tr.data, np.ma.masked_array):
                tr.data = tr.data.filled(fill_value=tr.data.mean())
        print('Demean')
        st_triggering.detrend('demean')
        print('Filtering')
        st_triggering.filter('highpass', freq=2000)
        # Standard DUGSeis trigger.
        detected_events = dug_trigger(
            st=st_triggering,
            # Helps with classification.
            active_triggering_channel="FS.CTrig..",
            minimum_time_between_events_in_seconds=0.02,
            max_spread_electronic_interference_in_seconds=2e-5,
            # Passed on the coincidence trigger.
            conincidence_trigger_opts={
                "trigger_type": "recstalta",
                "thr_on": 4.0,
                "thr_off": 1.5,
                "thr_coincidence_sum": 6,
                # The time windows are given in seconds.
                "sta": 0.002,
                "lta": 0.05,
                "trigger_off_extension": 0.01,
                "details": True,
            },
            plot=False,
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
            # Skip CASSM and electronic noise
            if event_candidate['classification'] in ['electronic', 'active']:
                continue
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
            if len(picks) < 3:
                # Optionally save the picks to the database as unassociated picks.
                # if picks:
                #    project.db.add_object(picks)
                continue

            event = locate_in_homogeneous_background_medium(
                picks=picks,
                coordinates=project.cartesian_coordinates,
                velocity=2800.0,
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
            if event_candidate['classification'] == 'passive':
                # Write waveform snippet to file
                o = event.preferred_origin()
                write_st = st_all.slice(starttime=o.time - 0.05,
                                        endtime=o.time + 0.1)
                write_st.write('{}/{}.ms'.format(
                    project.config['paths']['out_wav_folder'],
                    event.resource_id.id.split('/')[-1]), format='MSEED')
                # try:
                #     est_magnitude_energy(
                #         event=event, stream=st_mags,
                #         coordinates=project.cartesian_coordinates,
                #         global_to_local=project.global_to_local_coordinates,
                #         Vs=3700, p=3050, G=40, inventory=project.inventory,
                #         Q=210, Rc=0.63, plot=False)
                # except ValueError:
                #     pass

            # Write the classification as a comment.
            event.comments = [
                obspy.core.event.Comment(
                    text=f"Classification: {event_candidate['classification']}"
                )
            ]
            # Add the event to the project.
            added_event_count += 1
            try:
                project.db.add_object(event)
            except OperationalError:
                # database locked error, wait to see if it gets cleared
                time.sleep(1)
                project.db._backend.connection.commit()
            del st_event
        del st_triggering, st_mags
        gc.collect()
        logger.info(
            f"Successfully located {added_event_count} of "
            f"{len(detected_events)} event(s)."
        )
        total_event_count += added_event_count
        # Plot catalog data
        # catalog = project.db.get_objects(object_type="Event")
        # boreholes = project.config['graphical_interface']['3d_view']['line_segments']
        # try:
        #     plot_all(catalog, boreholes,
        #              global_to_local=project.global_to_local_coordinates,
        #              outfile=project.config['paths']['output_figure'])
        # except (IndexError, ValueError) as e:
        #     print(e)
        #     pass
        # Dump catalog to file
        project.db.dump_as_quakeml_files(
            folder=project.config['paths']['out_catalog_folder'])
        logger.info("DONE.")
        logger.info(f"Found {total_event_count} events.")

config_path = "/global/home/users/chopp/DUGseis/scripts/live_processing_FSB_bearbin_5-12-23.yaml"

# Change start time to current time
# edit_config(config_path)

with open(config_path, "r") as fh:
    yaml_template = yaml.load(fh, Loader=yaml.SafeLoader)

all_folders = [
    pathlib.Path(i).absolute() for i in yaml_template["paths"]["asdf_folders"]
]

ping_interval_in_seconds = 5

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

sd = project.config['temporal_range']['start_time'].datetime
# These files have already been processed.
already_processed_files = set([p.absolute() for p in project.waveforms._files.keys()])
# Monitor the folders and launch the processing again.
while True:
    all_available_files = set()
    for p in all_folders:
        new_files = set(
            [p.absolute() for p in p.glob("vbox_*.dat")
             if obspy.UTCDateTime(*[int(i) for i in re.match(FILENAME_REGEX, p.stem).groups()]).datetime >= sd])
        all_available_files = all_available_files.union(new_files)

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
