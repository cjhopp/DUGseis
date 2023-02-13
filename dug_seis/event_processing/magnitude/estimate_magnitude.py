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
Functionality to estimate moment magnitude from accelerometer recordings


"""

import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from copy import copy, deepcopy
from obspy import read
from obspy.core import Trace
from obspy.core.event import Magnitude
from scipy import optimize

def do_fft(signal, delta):
    """Compute the complex Fourier transform of a signal."""
    npts = len(signal)
    # if npts is even, we make it odd
    # so that we do not have a negative frequency in the last point
    # (see numpy.fft.rfft doc)
    if not npts % 2:
        npts -= 1

    fft = np.fft.rfft(signal, n=npts) * delta
    fftfreq = np.fft.fftfreq(len(signal), d=delta)
    fftfreq = fftfreq[0:fft.size]
    return fft, fftfreq


def do_spectrum(trace):
    """Compute the spectrum of an ObsPy Trace object."""
    signal = trace.data
    delta = trace.stats.delta
    amp, freq = do_fft(signal, delta)

    tr = Spectrum()
    # remove DC component (freq=0)
    tr.data = abs(amp)[1:]
    tr.stats.delta = 1. / (len(signal) * delta)
    tr.stats.begin = tr.stats.delta  # the first frequency is not 0!
    tr.stats.npts = len(tr.data)
    # copy some relevant header field
    tr.stats.station = trace.stats.station
    tr.stats.network = trace.stats.network
    tr.stats.location = trace.stats.location
    tr.stats.channel = trace.stats.channel
    return tr


class Spectrum(Trace):

    def get_freq(self):
        fdelta = self.stats.delta
        freq = np.arange(0, self.stats.npts*fdelta, fdelta)
        freq = freq[0:self.stats.npts]
        freq += self.stats.begin
        return freq

    def plot(self, **kwargs):
        freq = self.get_freq()
        plt.loglog(freq, self.data, **kwargs)
        plt.grid(True)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        return plt.gca()

    def slice(self, fmin, fmax, pad=False, nearest_sample=True,
              fill_value=None):
        t = self.stats.starttime
        freq = self.get_freq()
        begin = self.stats.begin
        spec_slice = copy(self)
        spec_slice.stats = deepcopy(self.stats)
        spec_slice.trim(t-begin+fmin, t-begin+fmax, pad, nearest_sample,
                        fill_value)
        delta_t = t - spec_slice.stats.starttime
        if delta_t > 0:
            spec_slice.stats.begin = begin - delta_t
        else:
            # find the closest frequency to fmin:
            idx = (np.abs(spec_slice.get_freq()-fmin)).argmin()
            spec_slice.stats.begin = freq[idx]
        return spec_slice


def fit_wrapper(tt, n, gamma):
    """
    Wrap displacement function for fitting

    :param tt: travel time
    :param n: Fall-off rate at high freqs
    :param gamma: Sharpness of the corner
    :param Q: Quality factor
    :return: 
    """
    def abercrombie_spec(freqs, fc, Q, Omega0):
        """
        Return displacement spectra as specified in Abercrombie 1995 Eqn 1

        :param freqs: frequencies at which to calculate displacement
        :param fc: corner frequency
        :param Omega0: Long-period spectral level
        """
        numerator = Omega0 * np.exp(-(np.pi * freqs * tt / Q))
        denominator = (1 + ((freqs / fc)**(gamma * n)))**(1 / gamma)
        spec = numerator / denominator
        return spec
    return abercrombie_spec


def fit_abercrombie(freqs, spec, tt, n, gamma):
    popt, pcov = optimize.curve_fit(fit_wrapper(tt, n, gamma), freqs, spec * 1e18)
    return popt, pcov


def boatwright_wrapper(Rc, p, Vc, R):
    def boatwright_spec(freqs, fc, Qc, M0):
        omega = (freqs * M0) / (1 + (freqs / fc)**4)**0.5
        spec = (Rc / (2 * p * Vc**3 * R)) * omega * np.exp(-(np.pi * R * freqs) / (Qc * Vc))
        return spec
    return boatwright_spec


def fit_boatwright(freqs, spec, Rc, p, Vc, R):
    popt, pcov = optimize.curve_fit(boatwright_wrapper(Rc, p, Vc, R), freqs, spec)
    return popt, pcov


def calculate_M0(p, V, dist, rad, Om1, Om2, Om3):
    """
    Abercrombie 1995 Eqn 2 for moment calculation from spectral levels

    :param p: Density of rock
    :param V: Velocity of phase in question
    :param dist: Source-receiver distance
    :param rad: Radiation constant (0.32 for P, 0.21 for S)
    :param Om1: Spectral level 1st component
    :param Om2: Spectral level 2nd component
    :param Om3: Spectral level 3rd component
    :return:
    """
    M0 = (4 * np.pi * p * V**3 * dist * np.sqrt(Om1**2 + Om2**2 + Om3**2)) / rad
    return M0


def plot_fit(trace, freqs, fft, popt, tt, noise_spec, spec_diff):
    """
    Debug plotting func for spectral fitting

    :param freqs: Array of frequencies
    :param fft: FFT of phase arrival
    :param popt: Best-fit parameters for spectra
    :param tt: Travel time for the phase
    :return:
    """
    fig, axes = plt.subplots(nrows=2, figsize=(8, 12))
    ab_spec = fit_wrapper(tt, n=2, gamma=2)
    best_fit = ab_spec(freqs, popt[0], popt[1], popt[2])
    axes[0].plot(np.arange(trace.data.size) * trace.stats.delta, trace.data)
    axes[1].plot(noise_spec.get_freq(), noise_spec.data * 1e18, alpha=0.5,
                 color='darkgray')
    axes[1].plot(freqs, spec_diff * 1e18, alpha=0.5, color='steelblue')
    axes[1].plot(freqs, fft * 1e18, label='Spectra', color='k')
    axes[1].plot(freqs, best_fit, label='Model fit', linestyle='--',
                 color='firebrick')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_ylabel('Displacement')
    axes[1].set_xlabel('Frequency')
    plt.title(trace.id)
    axes[0].set_title(popt)
    plt.show()
    return


def est_magnitude_spectra(event, stream, coordinates, global_to_local, Vp, Vs, p,
                          inventory, plot=False):
    """
    Apply a magnitude estimation following Abercrombie 1995  that attempts
    to fit a Brune model to the displacement spectra.

    :param event: Event object with origin and picks
    :param stream: Stream object containing XYZ for all accelerometers with picks
    :param coordinates: Station coordinates
    :param global_to_local: Func for converting global to cartesian coords
    :param Vp: P-wave velocity (m/s)
    :param Vs: S-wave velocity (m/s)
    :param p: Density (km/m**3)
    :param plot: Debug plotting flag
    :return:
    """
    o = event.preferred_origin()
    # Remove response for stream
    stream.trim(starttime=o.time - 0.05, endtime=o.time + 0.1)
    stream.detrend('linear')
    stream.detrend('simple')
    stream.resample(sampling_rate=40000., window='hann')
    rms = [tr for tr in stream if tr.stats.station[0] in ['T', 'C', 'P']]
    for r in rms:
        stream.traces.remove(r)
    stream.remove_response(inventory=inventory, output="ACC",
                           water_level=600, plot='test_resp.png',
                           pre_filt=[20, 30, 40000, 50000])
    x, y, z = global_to_local(latitude=o.latitude, longitude=o.longitude,
                              depth=o.depth)
    M0s = []
    for pk in event.picks:
        sx, sy, sz = coordinates[pk.waveform_id.id]
        distance = np.sqrt((sx - x)**2 + (sy - y)**2 + (sz - z)**2)
        tt_P = pk.time - o.time
        tt_S = distance / Vs
        st = stream.select(station=pk.waveform_id.station_code).copy()
        st.filter('highpass', freq=2000.)
        if len(st) == 0:
            continue  # Pick from hydrophone
        st_noise = st.slice(starttime=pk.time - .1, endtime=pk.time - 0.01).copy()
        st_noise.integrate().detrend('linear')  # VEL
        st_noise.integrate().detrend('linear')  # DISP
        st_P = st.slice(starttime=pk.time - 0.0005, endtime=pk.time + 0.002).copy()
        st_P.integrate().detrend('linear').integrate().detrend('linear')
        st_S = st.slice(starttime=o.time + tt_S, endtime=o.time + tt_S + 0.002).copy()
        st_S.integrate().detrend('linear').integrate().detrend('linear')
        st_P.plot()
        st_S.plot()
        Oms_P = []
        for tr in st_P:
            spec = do_spectrum(tr.copy())
            noise_spec = do_spectrum(st_noise.select(id=tr.id)[0])
            noise_interp = np.interp(spec.get_freq(), noise_spec.get_freq(),
                                     noise_spec.data)
            spec_diff = spec.data - noise_interp
            try:
                popt, pcov = fit_abercrombie(spec.get_freq(), spec.data, 0,
                                             n=2, gamma=2)
            except RuntimeError as e:
                continue
            Oms_P.append(popt[-1] / 1e18)
            if plot:
                plot_fit(tr, spec.get_freq(), spec.data, popt, 0, noise_spec,
                         spec_diff)
        M0s.append(calculate_M0(p, Vp, distance, 0.32,
                                Oms_P[0], Oms_P[1], Oms_P[2]))
        Oms_S = []
        for tr in st_S:
            spec = do_spectrum(tr.copy())
            noise_spec = do_spectrum(st_noise.select(id=tr.id)[0])
            noise_interp = np.interp(spec.get_freq(), noise_spec.get_freq(),
                                     noise_spec.data)
            spec_diff = spec.data - noise_interp
            try:
                popt, pcov = fit_abercrombie(spec.get_freq(), spec.data, 0,
                                             n=2, gamma=2)
            except RuntimeError as e:
                continue
            Oms_S.append(popt[-1] / 1e18)
            if plot:
                plot_fit(tr, spec.get_freq(), spec.data, popt, 0, noise_spec,
                         spec_diff)
        M0s.append(calculate_M0(p, Vp, distance, 0.21,
                                  Oms_S[0], Oms_S[1], Oms_S[2]))

    M0_avg = np.mean(np.array(M0s))
    Mw = (2 / 3) * np.log10(M0_avg) - 6.07
    event.magnitudes = [Magnitude(mag=Mw, type='Mw',
                                  origin_id=o.resource_id)]
    event.preferredMagnitudeID = event.magnitudes[-1].resource_id
    print(event.magnitudes[-1])
    return


def q(r):
    """Kanamori attenuation curve function. r in units of km"""
    q = 0.49710 * r**-1.0322 * np.exp(-.0035 * r)
    return q


def est_magnitude_energy(event, stream, coordinates, global_to_local, p, G,
                         Q, inventory, plot=False):
    """
    Apply a magnitude estimation estimates the energy in the acceleration
    arrival of the S wave. Follows Kwiatek et al work from Aspo

    :param event: Event object with origin and picks
    :param stream: Stream object containing XYZ for all accelerometers with picks
    :param coordinates: Station coordinates
    :param global_to_local: Func for converting global to cartesian coords
    :param Vs: S-wave velocity (m/s)
    :param p: Density (kg/m**3)
    :param plot: Debug plotting flag
    :return:
    """
    o = event.preferred_origin()
    st = stream.copy()
    # Remove response for stream
    st.trim(starttime=o.time - 0.05, endtime=o.time + 0.1)
    st.detrend('linear')
    st.detrend('simple')
    st.resample(sampling_rate=40000., window='hann')
    rms = [tr for tr in st if tr.stats.station[0] in ['T', 'C', 'P']]
    for r in rms:
        st.traces.remove(r)
    st.remove_response(inventory=inventory, output="ACC",
                       water_level=600, plot=False,
                       pre_filt=[20, 30, 40000, 50000])
    # x, y, z = global_to_local(point=(o.latitude, o.longitude, o.depth))
    x, y, z = global_to_local(latitude=o.latitude, longitude=o.longitude,
                              depth=-float(o.extra.hmc_elev.value))
    M0_Ps = []
    M0_Ss = []
    ev_snrs = []
    used_stations = []
    pk_dict = {}
    for pick in event.picks:
        if pick.waveform_id.station_code not in pk_dict:
            pk_dict[pick.waveform_id.station_code] = [pick]
        else:
            pk_dict[pick.waveform_id.station_code].append(pick)
    for tr in st:
        if tr.stats.station not in pk_dict.keys() or tr.stats.station in used_stations:
            continue
        for pk in pk_dict[tr.stats.station]:
            sx, sy, sz = coordinates[pk.waveform_id.id]
            if pk.phase_hint == 'P':
                Rc = 0.52
                V = 5880
            else:
                Rc = 0.63
                V = 3700
            print(x, y, z, sx, sy, sz)
            # Distance in meters
            distance = np.sqrt((sx - x)**2 + (sy - y)**2 + (sz - z)**2)
            print('Distance {}'.format(distance))
            s = st.select(station=pk.waveform_id.station_code).copy()
            s.filter(type='highpass', freq=1000.)
            s.integrate().detrend('linear')  # To velocity
            if len(s) != 3:
                print('{} not 3C'.format(pk.waveform_id.station_code))
                continue  # Pick from hydrophone
            st_noise = s.slice(starttime=pk.time - 0.037, endtime=pk.time - 0.007).copy()
            st_S = s.slice(starttime=pk.time, endtime=pk.time + 0.0015).copy()
            sig_pow = np.sqrt(np.mean(st_S.select(id=pk.waveform_id.get_seed_string())[0].copy().data ** 2))
            noise_pow = np.sqrt(np.mean(st_noise.select(id=pk.waveform_id.get_seed_string())[0].copy().data ** 2))
            pk_snr = 20 * np.log10(sig_pow / noise_pow)
            if pk_snr < 10.:  # 10 dB limit for picks
                continue
            ev_snrs.append(pk_snr)
            V_specs = []
            for t in st_S:
                V_specs.append(do_spectrum(t))
            # Kwiatec & BenZion 2013 (JAGUARS) eqn 3 and 4
            freqs = V_specs[0].get_freq()
            spec_data = [s.data for s in V_specs]
            spec_data = np.sum(spec_data, axis=0)
            Espec = (spec_data * np.exp((np.pi * freqs * distance) / (V * Q)))**2
            # Integrate over passband
            band_ints = np.where(freqs > 1000.)
            int_f = freqs[band_ints]
            Jc = 2 * np.trapz(Espec[band_ints], x=int_f)
            E_acc = 4 * np.pi * p * V * Rc**2 * (distance / Rc)**2 * Jc
            # Fit spectra
            try:
                popt, pcov = fit_boatwright(freqs, spec_data, Rc, p, V, distance)
            except RuntimeError as e:
                continue
            bw_spec = boatwright_wrapper(Rc, p, V, distance)
            best_fit = bw_spec(freqs, popt[0], popt[1], popt[2])
            if plot:
                plot_magnitude_calc(
                    s.select(id=pk.waveform_id.get_seed_string()), st_S.select(id=pk.waveform_id.get_seed_string()),
                    V_specs, spec_data, E_acc, pk_snr, st_noise.select(id=pk.waveform_id.get_seed_string()),
                    best_fit)
            M0 = 2 * G * E_acc / 1e-3
            if pk.phase_hint == 'P':
                M0_Ps.append(M0)  # Stress drop = 1e-3 GPa
            elif pk.phase_hint == 'S':
                M0_Ss.append(M0)
            used_stations.append(tr.stats.station)
    if len(M0_Ps) == 0 and len(M0_Ss) == 0:
        return np.nan, np.nan, np.nan, np.nan
    # Moment calculation from above gives N m (SI), must convert to dyn cm
    Mw_P = (0.667 * np.log10(np.mean(M0_Ps) * 1e7)) - 6.07
    Mw_S = (0.667 * np.log10(np.mean(M0_Ss) * 1e7)) - 6.07
    Mw = (0.6667 * np.log10(np.mean(M0_Ps + M0_Ss) * 1e7)) - 6.07
    ev_snr = np.mean(ev_snrs)
    print(Mw)
    magnitude = Magnitude(mag=Mw, type='Mw', origin_id=o.resource_id)
    event.magnitudes.append(magnitude)
    event.preferred_magnitude_id = event.magnitudes[-1].resource_id
    return Mw_P, Mw_S, ev_snr, Mw


def estimate_mags_catalog(catalog, project, stream_dict, plot_picks=False, plot_mag_stats=False):
    mwps = []
    mwss = []
    snrs = []
    Mws = []
    for ev in catalog:
        mwp, mws, snr, mw = est_magnitude_energy(ev, read(stream_dict[ev.resource_id.id.split('/')[-1]]),
                                                 project.cartesian_coordinates,
                                                 project.global_to_local_coordinates, p=3050, G=40, inventory=
                                                 project.inventory, Q=210, plot=plot_picks)
        mwps.append(mwp)
        mwss.append(mws)
        snrs.append(snr)
        Mws.append(mw)
    if plot_mag_stats:
        MwP = np.array(mwps)
        MwS = np.array(mwss)
        SNR = np.array(snrs)
        plot_mag_results(MwS[np.where((~np.isnan(MwS)) & (~np.isnan(MwP)))],
                         MwP[np.where((~np.isnan(MwS)) & (~np.isnan(MwP)))],
                         SNR[np.where((~np.isnan(MwS)) & (~np.isnan(MwP)))],
                         np.array(Mws))
    return


def freq_band_correction(fM, fo):
    """
    Correct radiated energy estimate for limited frequency band following Ide & Beroza 2001 (GRL)
    :return:
    """
    F = (-fM / fo) / (1 + (fM / fo))**2 + np.arctan(fM / fo)
    return (2 / np.pi) * F


def plot_magnitude_calc(st, st_S, V_specs, spec_sum, E_acc, snr, st_noise, best_fit):
    """QC plot for magnitude estimation"""
    fig, axes = plt.subplots(nrows=2, figsize=(12, 8))
    axes[0].plot(st[0].times(), st[0].data, color='k', linewidth=0.7)
    axes[0].plot(st_S[0].times(reftime=st[0].stats.starttime),
                 st_S[0].data, color='r', linewidth=0.8)
    axes[0].plot(st_noise[0].times(reftime=st[0].stats.starttime),
                 st_noise[0].data, color='b', linewidth=0.8)
    axes[0].set_title(st[0].stats.station)
    axes[1] = V_specs[0].plot()
    do_spectrum(st_noise[0]).plot(axes=axes[1], linestyle='--', color='gray')
    for vspec in V_specs[1:]:
        vspec.plot(axes=axes[1])
    axes[1].plot(vspec.get_freq(), spec_sum)
    axes[1].plot(vspec.get_freq(), best_fit, linestyle=':', color='magenta')
    axes[1].set_yscale('log')
    axes[1].annotate(xy=(0.03, 0.7), text='Energy: {}'.format(E_acc), fontsize=8,
                     xycoords='axes fraction')
    axes[1].annotate(xy=(0.03, 0.8), text='SNR: {} dB'.format(snr), fontsize=8,
                     xycoords='axes fraction')
    plt.show()
    return


def plot_mag_results(X, Y, SNRs, Mws):
    print(X, Y)
    fig, axes = plt.subplots(ncol=2)
    axes[0].scatter(X, Y, s=SNRs)
    res = stats.linregress(X, Y)
    regr_y = res.intercept + res.slope * X
    axes[0].plot(X, regr_y)
    axes[0].plot(X, X, linestyle='--', color='k')
    axes[0].annotate(xy=(0.2, 0.9), xytext=(0.2, 0.9), xycoords='axes fraction', text='{:0.2f}'.format(res.slope))
    axes[0].set_xlabel('M$_W$ from S')
    axes[0].set_ylabel('M$_W$ from P')
    axes[0].set_aspect('equal')
    axes[0].set_xlim([-7, -1])
    axes[0].set_ylim([-7, -1])
    axes[0].set_title('M$_{W}$ P vs S')
    axes[1].hist(Mws, cumulative=True, )
    plt.show()
    return


def est_magnitude_amplitude(event, stream, coordinates, global_to_local):
    """
    Will fill this is if there's time later (and if it's needed)
    """
    return