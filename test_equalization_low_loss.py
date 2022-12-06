#!/usr/bin/env python3

# Generate a PAM-N signal and see how it is degraded through the channel filter

import numpy as np
import matplotlib.pyplot as plt
import sk_dsp_comm.digitalcom as dc
from   scipy import signal
from   scipy import linalg

import os

def zero_force(tx_pulse, b_chan, a_chan, samples_per_symbol, n):
    npad = 4 * n * samples_per_symbol
    p = np.hstack((np.zeros(npad), tx_pulse, np.zeros(npad)))
    p = signal.lfilter(b_chan, a_chan, p)
    p = signal.lfilter(tx_pulse / samples_per_symbol, 1, p)

    p_max, I_max = max(p), np.argmax(p)

    p_vec = p[I_max - samples_per_symbol * 2 * n : I_max + samples_per_symbol * 2 * n + 1 : samples_per_symbol]
    p_c   = p[I_max - samples_per_symbol * 3 * n : I_max + samples_per_symbol * 4 * n + 1 : samples_per_symbol]
    n_tap = 2 * n + 1
    Pc    = np.zeros((n_tap, n_tap))
    for k in range(2*n + 1):
        Pc[:,k] = p_vec[2*n-k:2*n-k+n_tap]
    Peq = np.hstack((np.zeros(n), [1], np.zeros(n))).T
    Aopt = linalg.solve(Pc, Peq)
    return Aopt, p_c, p_vec

use_ctle           = int(os.getenv('USE_CTLE',         '1')) != 0
use_aggressor      = int(os.getenv('USE_AGGRESSOR',    '0')) != 0
offset_aggressor   = int(os.getenv('OFFSET_AGGRESSOR', '1')) != 0
aggressor_attenuation = float(os.getenv('AGGRESSOR_ATTENUATION', '0.1'))
n_levels           = int(os.getenv('N_LEVELS', '4'))
channel_div        = int(os.getenv('CHANNEL_DIV', '4'))
samples_per_symbol = int(os.getenv('SAMPLES_PER_SYMBOL', '4'))
symbols            = int(os.getenv('SYMBOLS', 116e3))
ffe_preset_index   = int(os.getenv('FFE_PRESET_INDEX', '1'))

use_dfe              = int(os.getenv('USE_DFE', '1')) != 0
use_zero_forcing_for_initial_guess = int(os.getenv('USE_ZERO_FORCING_FOR_INITIAL_GUESS', '0')) != 0
dfe_n                = int(os.getenv('DFE_N', 0))   # Pre Taps
dfe_m                = int(os.getenv('DFE_M', 5))   # Post Taps
dfe_mu               = float(os.getenv('DFE_MU', 1e-4))  # Training Rate
dfe_training_symbols = int(os.getenv('DFE_TRAINING_SYMBOLS', symbols - int(16e3)))

channel_fs         = 128e9
ui_rate            = channel_fs/samples_per_symbol

plot_freq = True

# Table 8-2, Tx FFE presens - FIRs at UI rate
ffe_presets = [
  # c-2     c-1     c0     c1       Preshoot2      Preshoot1       De-emphasesis
  [0.000,  0.000,  1.000,  0.000], #     0dB             0dB              -3.5dB
  [0.000, -0.083,  1.000,  0.000], #     0dB           1.6dB              -6.0dB
  [0.000, -0.167,  1.000,  0.000], #    0dB            3.5dB                 0dB
  [0.000,  0.000,  1.000, -0.083], #     0dB             0dB              -1.6dB
  [0.000,  0.000,  1.000, -0.167], #     0dB             0dB              -3.5dB
  [0.042, -0.208,  1.000,  0.000], #  -1.3dB           4.7dB                 0dB
  [0.042, -0.125,  1.000, -0.125], #  -1.6dB           3.5dB              -3.5dB
  [0.083, -0.208,  1.000,  0.000], #  -2.9dB           4.7dB                 0dB
  [0.083, -0.250,  1.000,  0.000], #  -3.5dB           6.0dB                 0dB
  [0.083,  0.250,  1.000, -0.042], #  -4.4dB           6.9dB              -1.6dB
]

print(f"use_ctle:               {use_ctle}\n"\
      f"use_aggressor:          {use_aggressor}\n"\
      f"n_levels:               {n_levels}\n"\
      f"samples_per_symbol:     {samples_per_symbol}\n"\
      f"fs:                     {channel_fs}\n"\
      f"ui_rate:                {ui_rate}\n"\
      f"ffe_preset_index:       {ffe_preset_index}\n"\
      f"use_dfe:                {use_dfe}\n"\
      f"use_zfeq_initial_guess: {use_zero_forcing_for_initial_guess}\n"\
      f"dfe_n:                  {dfe_n}\n"\
      f"dfe_m:                  {dfe_m}\n"\
      f"dfe_mu:                 {dfe_mu}")

ffe_filter = ffe_presets[ffe_preset_index]

channel            = np.zeros(128000)
with open("impulse_responses/impulse_victim_differential_low_loss.txt", "r") as f:
    for i,li in enumerate(f):
        channel[i] = float(li)

ctle_b = []
ctle_a = []
if use_ctle:
    with open("ctle.txt", "r") as f:
        for i,li in enumerate(f):
            if i == 0:
                for j,xj in enumerate(li.split()):
                    ctle_b.append(float(xj))
            elif i == 1:
                for j,xj in enumerate(li.split()):
                    ctle_a.append(float(xj))
            else:
                print("error")
                exit(1)
ctle_b = np.array(ctle_b)
ctle_a = np.array(ctle_a)

channel_trunc = channel[0:len(channel)//channel_div]

if use_ctle:
    channel_b = signal.convolve(channel_trunc, ctle_b)
    channel_a = ctle_a
else:
    channel_b = channel_trunc
    channel_a = [1]

eff_filt_len = len(channel_b) + len(channel_a) - 1 # approximately
nsyms = int(symbols + np.ceil((eff_filt_len*1.0)/samples_per_symbol))
group_delay_fudge_factor = 3 # so that the sampling point is at about 0, samples_per_symbol, ... of filtered_trunc. Could fractionally interpolate but no need for now

# generate n equalually spaced levels in [-1, 1] inclusive
rand_ints       = np.random.randint(0, n_levels, nsyms)
normalized_rand = 2 * (rand_ints / (n_levels - 1.0)) - 1
riffe = signal.convolve(normalized_rand, ffe_filter)

pre_channel_symbols = np.zeros(samples_per_symbol * len(normalized_rand))
for i in range(len(pre_channel_symbols)):
    pre_channel_symbols[i] = riffe[i//samples_per_symbol]
pre_channel_symbols = np.hstack(([-1] * (samples_per_symbol*((eff_filt_len + group_delay_fudge_factor)//samples_per_symbol)), pre_channel_symbols))  

filtered_signal = signal.lfilter(channel_b, channel_a, pre_channel_symbols)
filtered_trunc  = filtered_signal[eff_filt_len+group_delay_fudge_factor:]

# generate a seperate, independant lane and combine it with the data
if use_aggressor:
    channel_aggressor_to_victim = np.zeros(128000)
    with open("impulse_responses/impulse_aggressor_to_victim_low_loss.txt", "r") as f:
        for i,li in enumerate(f):
            channel_aggressor_to_victim[i] = float(li)
    channel_trunc_aggressor_to_victim = channel_aggressor_to_victim[0:len(channel)//channel_div]
    if use_ctle:
        channel_b_aggressor = signal.convolve(channel_trunc_aggressor_to_victim, ctle_b)
        channel_a_aggressor = ctle_a
    else:
        channel_b_aggressor = channel_trunc_aggressor_to_victim
        channel_a_aggressor = [1]
    # generate n equalually spaced levels in [-1, 1] inclusive
    rand_ints_aggressor       = np.random.randint(0, n_levels, nsyms)
    normalized_rand_aggressor = 2 * (rand_ints_aggressor / (n_levels - 1.0)) - 1
    riffe_aggressor = signal.convolve(normalized_rand_aggressor, ffe_filter)

    pre_channel_symbols_aggressor = np.zeros(samples_per_symbol * len(normalized_rand_aggressor))
    for i in range(len(pre_channel_symbols_aggressor)):
        pre_channel_symbols_aggressor[i] = riffe_aggressor[i//samples_per_symbol]
    pre_channel_symbols_aggressor = np.hstack(([-1] * (samples_per_symbol*((eff_filt_len + group_delay_fudge_factor)//samples_per_symbol)), pre_channel_symbols_aggressor))

    filtered_signal_aggressor = signal.lfilter(channel_b_aggressor, channel_a_aggressor, pre_channel_symbols_aggressor)
    filtered_trunc_aggressor  = filtered_signal_aggressor[eff_filt_len+group_delay_fudge_factor+(samples_per_symbol//2 if offset_aggressor else 0):]

    if len(filtered_trunc_aggressor) < len(filtered_trunc):
        filtered_trunc = filtered_trunc[:len(filtered_trunc_aggressor)]
    else:
        filtered_trunc_aggressor = filtered_trunc_aggressor[:len(filtered_trunc)]
    filtered_trunc = filtered_trunc + aggressor_attenuation * filtered_trunc_aggressor

if use_dfe:
    # The center of the filter is at about 202
    dfe_out         = np.zeros(len(filtered_trunc))
    mmse            = []

    # use zero forcing to get an initial guess
    if use_zero_forcing_for_initial_guess:
        chan_trim = channel_b[135:175]
        Aopt, p_c1, p_vec = zero_force(np.ones(samples_per_symbol), chan_trim, channel_a, samples_per_symbol, dfe_n)
    else:
        Aopt = np.hstack((np.zeros(dfe_n), [2], np.zeros(dfe_n)))

    Aopt = np.hstack((Aopt, np.zeros(dfe_m)))

    pre_pres_states = np.zeros((samples_per_symbol, 2 * dfe_n + 1))
    post_states     = np.zeros(dfe_m)
    levels          = ((2.0/(n_levels - 1)) * np.linspace(0, n_levels - 1, n_levels)) - 1
    mod             = 0
    for i,xi in enumerate(filtered_trunc):
        mod = mod + 1 if mod != samples_per_symbol - 1 else 0
        pre_pres_states[mod] = np.hstack((xi, pre_pres_states[mod][:-1]))
        states               = np.hstack((pre_pres_states[mod], post_states))
        dfe_out[i]           = np.dot(Aopt, states)
        if mod == 0: # update post states and train
            dfe_error       = dfe_out[i] - levels
            min_error,index = min(abs(dfe_error)), int(np.argmin(abs(dfe_error)))
            post_states     = np.hstack((levels[index], post_states[:-1]))
            if i < dfe_training_symbols * samples_per_symbol:
                e               = post_states[0] - dfe_out[i]
                mmse.append(np.dot(e, e))
                Aopt            = Aopt + dfe_mu * (np.conj(states)) * e
    print('    Pre taps = [')
    for tap in (Aopt[:2*dfe_n+1]):
        print(f'        {tap},')
    print('    ]')
    print('    Post taps = [')
    for tap in (Aopt[2*dfe_n+1:]):
        print(f'        {tap},')
    print('    ]')
    to_plot = dfe_out[samples_per_symbol*dfe_training_symbols:]
    plt.figure(figsize=(6,4))
    ax = plt.subplot(1, 1, 1)
    ax.plot(mmse)
    ax.set_xlabel('UI Index')
    ax.set_ylabel('Mean Squared Error')
    ax.grid()
else:
    to_plot = filtered_trunc

dc.eye_plot(to_plot.real, 4 * samples_per_symbol, samples_per_symbol)
plt.show()

if not plot_freq:
    exit(0)

plt.figure(figsize=(6,4))
ax = plt.subplot(1, 1, 1)

f = np.arange(0, channel_fs/2, 1e7)
a = [1]
b = np.zeros(samples_per_symbol * len(ffe_filter))
for i in range(len(ffe_filter)):
    b[samples_per_symbol * i] = ffe_filter[i]
w, H1 = signal.freqz(b, a, 2*np.pi*f/channel_fs)
ax.semilogx(f, 20*np.log10(np.abs(H1)), label='Response After Tx FFE')

b = signal.convolve(b, channel_trunc)
w, H1 = signal.freqz(b, a, 2*np.pi*f/channel_fs)
ax.semilogx(f, 20*np.log10(np.abs(H1)), label='Response After Channel')

b = signal.convolve(b, ctle_b)
a = signal.convolve(a, ctle_a)
w, H1 = signal.freqz(b, a, 2*np.pi*f/channel_fs)
ax.semilogx(f, 20*np.log10(np.abs(H1)), label='Response After CTLE')

if use_dfe:
    dd_interpolated_b = np.zeros(samples_per_symbol * (2*dfe_n + 1))
    dd_interpolated_a = np.zeros(samples_per_symbol * (dfe_m + 1))
    for i in range(2*dfe_n + 1):
        dd_interpolated_b[samples_per_symbol * i] = Aopt[i]
    dd_interpolated_a[0] = 1
    for i in range(dfe_m):
        dd_interpolated_a[samples_per_symbol * (i + 1)] = Aopt[i + (2*dfe_n+1)]
    b = signal.convolve(b, dd_interpolated_b)
    a = signal.convolve(a, dd_interpolated_a)
    w, H1 = signal.freqz(b, a, 2*np.pi*f/channel_fs)
    ax.semilogx(f, 20*np.log10(np.abs(H1) + 1e-12), label='Response After RX Eq')

ax.legend(loc='lower left')
ax.set_ylim([-40, 10])
ax.set_xlim([10e6, channel_fs/2])
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Gain (dB)')
ax.grid()
plt.show()

plt.figure(figsize=(6,4))
ax = plt.subplot(1, 1, 1)
b = signal.convolve(np.ones(samples_per_symbol), b)
s = signal.lfilter(b, a, np.hstack((np.ones(1), np.zeros(1000))))[140:210]
x = np.linspace(130, 130+len(s)-1, len(s))
main_x = np.argmax(s)
ax.stem(x[0:main_x],        s[0:main_x],        linefmt='green', label='Precursor')
ax.stem(x[main_x:main_x+1], s[main_x:main_x+1], linefmt='blue',  label='Main Cursor')
ax.stem(x[main_x+1:],       s[main_x+1:],       linefmt='red',   label='Postcursor')
ax.set_xlabel('Sample Index (@128GSa/s)')
ax.set_ylabel('Amplitude')
ax.grid()
ax.legend()
plt.show()
