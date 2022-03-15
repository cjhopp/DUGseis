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

from copy import copy, deepcopy
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
        import matplotlib.pyplot as plt
        plt.loglog(freq, self.data, **kwargs)
        plt.grid(True)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.show()

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
    stream.resample(sampling_rate=40000.)
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


def est_magnitude_energy(event, stream, coordinates, global_to_local, Vs, p, G,
                         inventory, plot=False):
    """
    Apply a magnitude estimation estimates the energy in the acceleration
    arrival of the S wave. Follows Kwiatek et al work from Aspo

    :param event: Event object with origin and picks
    :param stream: Stream object containing XYZ for all accelerometers with picks
    :param coordinates: Station coordinates
    :param global_to_local: Func for converting global to cartesian coords
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
    stream.resample(sampling_rate=40000.)
    stream.remove_response(inventory=inventory, output="ACC",
                            water_level=600, plot=False,
                            pre_filt=[20, 30, 40000, 50000])
    x, y, z = global_to_local(latitude=o.latitude, longitude=o.longitude,
                              depth=o.depth)
    M0s = []
    for pk in event.picks:
        sx, sy, sz = coordinates[pk.waveform_id.id]
        print(pk.waveform_id.id)
        distance = np.sqrt((sx - x)**2 + (sy - y)**2 + (sz - z)**2)
        print('Distance {}'.format(distance))
        tt_S = distance / Vs
        s_time = o.time + tt_S
        st = stream.select(station=pk.waveform_id.station_code).copy()
        st.filter('highpass', 2000.)
        if len(st) == 0:
            continue  # Pick from hydrophone
        st_noise = st.slice(starttime=pk.time - .1,
                            endtime=pk.time - 0.01).copy()
        st_noise.integrate().detrend('linear')  # VEL
        st_S = st.slice(starttime=s_time, endtime=s_time + 0.016).copy()
        st_S.integrate().detrend('linear')  # VEL
        if plot:
            st_S.plot()
        E_Ss = []
        for tr in st_S:
            V_spec = do_spectrum(tr)
            int_V = np.trapz(V_spec.data**2, x=V_spec.get_freq())
            E_acc = 8 * np.pi * p * Vs * distance * int_V
            E_Ss.append(E_acc)
            if plot:
                print(int_V)
        M0s.append(2 * G * np.mean(E_Ss) * 1e3)  # Stress drop = 1e-3 GPa
    Mw = (0.6667 * np.log10(np.mean(M0s))) - 6.07
    print(Mw)
    magnitude = Magnitude(mag=Mw, type='Mw', origin_id=o.resource_id)
    event.magnitudes.append(magnitude)
    event.preferred_magnitude_id = event.magnitudes[-1].resource_id
    return


def est_magnitude_amplitude(event, stream, coordinates, global_to_local):
    """
    Will fill this is if there's time later (and if it's needed)
    """
    return