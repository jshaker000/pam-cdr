#!/usr/bin/env python3

# Combine the equalization and timing recovery simulations to study the dynamics of both the equalizer and the
# phase updating simulataneously

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

# points is a 2d array of [[x0, y0], [x1, y1], [x2, y2], ..], xout is the point of the desired sample to produce
def lagrange_interp(points, x):
    y = 0
    for j,pj in enumerate(points):
        product = pj[1]
        for m,pm in enumerate(points):
            if m != j:
                product *= (x - pm[0]) / (pj[0] - pm[0])
        y += product
    return y

# from a numpy array (assuming the indicies are from equally spaced time series),
# produce the data that would be at a (possibly fractional) index, i by taking samples and sending to a lagrange interpolator
def interpolated_index(x, i, num_points):
    p = np.zeros((num_points,2))
    fi = int(np.floor(i))
    for k in range(num_points):
        if k % 2 == 0:
            index_to_use = fi + k //2
        else:
            index_to_use = fi - (k + 1)//2
        if index_to_use < 0 or index_to_use >= len(x):
            x_use = 0
        else:
            x_use = x[index_to_use]
        p[k] = np.array([index_to_use, x_use.real])
    return lagrange_interp(p, i)

def mm_ted(candidate, candidate_quantized, last, last_quantized):
    return candidate * last_quantized - last * candidate_quantized

use_ctle           = int(os.getenv('USE_CTLE',         '1')) != 0
use_aggressor      = int(os.getenv('USE_AGGRESSOR',    '0')) != 0
offset_aggressor   = int(os.getenv('OFFSET_AGGRESSOR', '1')) != 0
aggressor_attenuation = float(os.getenv('AGGRESSOR_ATTENUATION', '0.1'))
n_levels           = int(os.getenv('N_LEVELS', '2'))
channel_div        = int(os.getenv('CHANNEL_DIV', '4'))
samples_per_symbol = int(os.getenv('SAMPLES_PER_SYMBOL', '4'))
symbols            = int(os.getenv('SYMBOLS', 116e3))
ffe_preset_index   = int(os.getenv('FFE_PRESET_INDEX', '1'))

use_dfe              = int(os.getenv('USE_DFE', '1')) != 0
use_zero_forcing_for_initial_guess = int(os.getenv('USE_ZERO_FORCING_FOR_INITIAL_GUESS', '0')) != 0
dfe_n                = int(os.getenv('DFE_N', 0))   # Pre Taps
dfe_m                = int(os.getenv('DFE_M', 10))   # Post Taps
dfe_mu               = float(os.getenv('DFE_MU', 1e-4))  # Training Rate
lock_symbols         = int(os.getenv('LOCK_SYMBOLS', symbols - int(16e3)))

channel_fs         = 128e9
ui_rate            = channel_fs/samples_per_symbol

plot_freq = True

samples_per_symbol_uncertainty_factor = float(os.getenv('SAMPLES_PER_SYMBOL_UNCERTAINTY_FACTOR', '1e-4'))
extra_loop_latency                    = int(os.getenv('EXTRA_LOOP_LATENCY', '0'))
loop_bw                               = float(os.getenv('LOOP_BW', '5e6'))
damping_factor                        = float(os.getenv('DAMPING_FACTOR', '2.5'))
k0                                    = float(os.getenv('K0', 3))
delay                                 = float(os.getenv('DELAY', (2 * samples_per_symbol * (np.random.rand() - 0.5)))) # fractional delay to add
snr_dac                               = float(os.getenv('SNR_DAC', 30))
snr_adc                               = float(os.getenv('SNR_ADC', 30))
timing_jitter_sa_std                  = float(os.getenv('TIMING_JITTER_SA_STD', 0.00001))

# what does the receiver think samples_per_symbol is?
# This is essentially equivalent to resampling the Tx signal
thought_samples_per_symbol = float(os.getenv('THOUGHT_SAMPLES_PER_SYMBOL', (1 + samples_per_symbol_uncertainty_factor*(np.random.rand() - 0.5)) * samples_per_symbol))

# loop filter settings - kp is proportional gain, ki is integral gain
T_loop  = 1.0 / ui_rate
wc_loop = 2 * np.pi * loop_bw
wn_loop = wc_loop / np.sqrt(2*(damping_factor**2) + np.sqrt(((2*(damping_factor**2))**2)+1) - 1)
kp = 2 * damping_factor * wn_loop * T_loop
ki = ((wn_loop * T_loop)**2)

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
      f"use_zfeq_initial_guess: {use_zero_forcing_for_initial_guess}\n"\
      f"dfe_n:                  {dfe_n}\n"\
      f"dfe_m:                  {dfe_m}\n"\
      f"dfe_mu:                 {dfe_mu}\n"\
      f"samples_per_symbol_uncertainty_factor: {samples_per_symbol_uncertainty_factor}\n"\
      f"thought_samples_per_symbol:            {thought_samples_per_symbol}\n"\
      f"timing_jitter_sa_std:                  {timing_jitter_sa_std}\n"\
      f"loop_bw:                               {loop_bw}\n"\
      f"damping_factor:                        {damping_factor}\n"\
      f"k0:                                    {k0}\n"
      f"extra_loop_latency:                    {extra_loop_latency}\n"\
      f"delay:                                 {delay}\n"\
      f"snr_dac:                               {snr_dac}dB\n"\
      f"snr_adc:                               {snr_adc}dB")

ffe_filter = ffe_presets[ffe_preset_index]

channel            = np.zeros(128000)
with open("impulse_responses/impulse_victim_differential.txt", "r") as f:
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

# generate n equalually spaced levels in [-1, 1] inclusive
rand_ints       = np.random.randint(0, n_levels, nsyms)
normalized_rand = 2 * (rand_ints / (n_levels - 1.0)) - 1
riffe = signal.convolve(normalized_rand, ffe_filter)

pre_channel_symbols = np.zeros(samples_per_symbol * len(normalized_rand))
for i in range(len(pre_channel_symbols)):
    pre_channel_symbols[i] = riffe[i//samples_per_symbol]
pre_channel_symbols = np.hstack(([-1] * (samples_per_symbol*((eff_filt_len)//samples_per_symbol)), pre_channel_symbols))
pre_channel_symbols = dc.cpx_awgn(pre_channel_symbols, snr_dac, samples_per_symbol)

filtered_signal = signal.lfilter(channel_b, channel_a, pre_channel_symbols)
filtered_trunc  = filtered_signal[eff_filt_len:]

# generate a seperate, independant lane and combine it with the data
if use_aggressor:
    channel_aggressor_to_victim = np.zeros(128000)
    with open("impulse_responses/impulse_aggressor_to_victim.txt", "r") as f:
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
    pre_channel_symbols_aggressor = np.hstack(([-1] * (samples_per_symbol*((eff_filt_len)//samples_per_symbol)), pre_channel_symbols_aggressor))
    pre_channel_symbols_aggressor = dc.cpx_awgn(pre_channel_symbols_aggressor, snr_dac, samples_per_symbol)

    filtered_signal_aggressor = signal.lfilter(channel_b_aggressor, channel_a_aggressor, pre_channel_symbols_aggressor)
    filtered_trunc_aggressor  = filtered_signal_aggressor[eff_filt_len+(samples_per_symbol//2 if offset_aggressor else 0):]

    if len(filtered_trunc_aggressor) < len(filtered_trunc):
        filtered_trunc = filtered_trunc[:len(filtered_trunc_aggressor)]
    else:
        filtered_trunc_aggressor = filtered_trunc_aggressor[:len(filtered_trunc)]
    filtered_trunc = filtered_trunc + aggressor_attenuation * filtered_trunc_aggressor
ft = np.zeros(len(filtered_trunc))
for i in range(len(ft)):
    ft[i] = interpolated_index(filtered_trunc, i + delay + np.random.normal(0, timing_jitter_sa_std), 3)
ft = dc.cpx_awgn(ft, snr_adc, samples_per_symbol)

# use zero forcing to get an initial guess
if use_zero_forcing_for_initial_guess:
    chan_trim = channel_b[135:175]
    Aopt, p_c1, p_vec = zero_force(np.ones(samples_per_symbol), chan_trim, channel_a, samples_per_symbol, dfe_n)
else:
    Aopt = np.hstack((np.zeros(dfe_n), [2], np.zeros(dfe_n)))
Aopt = np.hstack((Aopt, np.zeros(dfe_m)))

pre_pres_states = np.zeros(2 * dfe_n + 1)
post_states     = np.zeros(dfe_m)
levels          = ((2.0/(n_levels - 1)) * np.linspace(0, n_levels - 1, n_levels)) - 1
candidate_index = 0.01
ri              = -1

received           = np.zeros(int(1.1*len(ft) // int(samples_per_symbol) + 1))
received_quantized = np.zeros(len(received))
timing_error       = np.zeros(len(received))
loop_integral      = np.zeros(len(received))
control_effort     = np.zeros(len(received) + extra_loop_latency)
mse                = np.zeros(len(received))

while candidate_index < len(filtered_trunc):
    candidate = interpolated_index(ft, candidate_index, 3)
    ri += 1
    pre_pres_states        = np.hstack((candidate, pre_pres_states[:-1]))
    states                 = np.hstack((pre_pres_states, post_states))
    received[ri]           = np.dot(Aopt, states)
    dfe_error              = received[ri] - levels
    min_error,index        = min(abs(dfe_error)), int(np.argmin(abs(dfe_error)))
    received_quantized[ri] = levels[index]
    post_states            = np.hstack((received_quantized[ri], post_states[:-1]))
    e    = post_states[0] - received[ri]
    Aopt = Aopt + dfe_mu * states * e
    mse[ri] = e**2
    timing_error[ri]   = k0 * mm_ted(received[ri], received_quantized[ri], received[ri-1], received_quantized[ri-1])
    loop_integral[ri]  = timing_error[ri-1] + loop_integral[ri-1]
    control_effort[ri] = (kp * timing_error[ri]) + (ki * loop_integral[ri]) + control_effort[ri - 1]
    candidate_index  = (ri + 1) * thought_samples_per_symbol + control_effort[ri]

# try to determine correlation to line up sample steams
c = np.correlate(received_quantized[lock_symbols:], normalized_rand[lock_symbols:], mode='full')
shift = len(normalized_rand) - lock_symbols - c.argmax() - 1

x_d = np.arange(0, len(normalized_rand))
x_r = np.arange(shift, shift + ri)
x_r = x_r[0:len(normalized_rand)]

bits        = 0
bit_errors  = 0
lock_errors = 0
print(f"Found a shift value of {shift}")
for i,xi in enumerate(x_r):
    if i > 0 and xi < len(normalized_rand):
        bits += 1
        if abs(normalized_rand[xi] - received_quantized[i]) > 1e-3:
            bit_errors = bit_errors + 1
            if xi > lock_symbols:
                lock_errors = lock_errors + 1
print(f"Errors after the first {lock_symbols} symbols {lock_errors}/{bits - lock_symbols}")

plt.figure(figsize=(6,12))
ax_0 = plt.subplot(3, 1, 1)
ax_0.plot(x_d, normalized_rand, 'b.')
ax_0.set_title('Sent symbols')
ax_0.grid()

ax_1 = plt.subplot(3, 1, 2, sharex=ax_0)
ax_1.plot(x_r, received_quantized[0:ri], 'b.')
ax_1.set_title('Received Quantized')
ax_1.grid()

ax_2 = plt.subplot(3, 1, 3, sharex=ax_0)
ax_2.plot(x_r, received[0:ri], 'b.')
ax_2.set_title('Received symbols')
ax_2.grid()

plt.show()


plt.figure(figsize=(6,12))

ax_0 = plt.subplot(3, 1, 1)
ax_0.plot(x_r, received[0:ri], 'b.')
ax_0.set_title('Received symbols')
ax_0.set_ylabel('Amplitude')
ax_0.grid()

ax_1 = plt.subplot(3, 1, 2, sharex=ax_0)
ax_1.plot(mse)
ax_1.set_xlabel('UI Index')
ax_1.set_title('LMS Mean Squared Error')
ax_1.set_ylabel('Amplitude')
ax_1.grid()

ax_2 = plt.subplot(3, 1, 3, sharex=ax_0)
ax_2.plot(control_effort[0:ri])
ax_2.set_xlabel('UI Index')
ax_2.set_ylabel('Amplitude (ADC Samples)')
ax_2.set_title(r'Loop Filter Output $l[n]$')
ax_2.grid()

plt.tight_layout()
plt.show()
