#!/usr/bin/env python3

import sk_dsp_comm.sigsys as ss
import scipy.signal as signal
import sk_dsp_comm.digitalcom as dc
import scipy.special as special
import scipy.stats as stats
import numpy as np
import math
import matplotlib.pyplot as plt
import re

import os

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

# early late says that, on average, with a symmetrical pulse and 0 ISI at the sampling instant,
# and perfect timing, and assuming data bits are uncorrelated from one another, then
# the early symbol (1/2 UI early) and the late symbol (1/2 UI late)
# subtracted from one another should be equal to one another
# If we are too early, then the early symbol will be further from the candidate,
# if we are too late, the late symbol will be further from the candidate
# We are essentially estimating the derivative at our candidate point using the early and late
# Note that this needs 2 samples / symbol
def early_late_ted(early, candidate, late):
    return candidate * (late - early)

# mm says that, on average, with a symmetrical pulse and 0 ISI at the sampling instant,
# and perfect timing, and assuming data bits are uncorrelated with one another, then the
# early symbol (1 UI early) and the late symbol (1 UI late) should on average be equal to one another, and that
# the early symbol will have the effect of candidate*pulse(-1UI-delta) on it and the late symbol will have effect candidate*pulse(UI-delta)
# if we sample early, then the early sample and late sample should have opposite signs to one another (with late having the same sign as the candidate)
# if we sample late, then the early sample and late sample should have opposite signs to one another (with early having the same sign as the candidate)
# We compute this more efficients by saying:
#    For symbol n-1, n sends value pulse(1UI - delta) into the past to n-1.
#      We can remove modulation by multiplying y[n] with z[n-1]; leaving pulse(1UI-delta)
#    For symbol n,   it sends pulse(-1UI - delta) into the past to symbol n-1.
#      We can remove modulation by multiplying z[n] with y[n-1]; leaving pulse(-1UI-delta)
# If delta is 0, then we are approximately correctly time aligned
# Note this only needs 1 sample / symbol
def mm_ted(candidate, candidate_quantized, last, last_quantized):
    return candidate * last_quantized - last * candidate_quantized

ONS      = 100                                   # Original num samples / symbol
n_bits   = int(os.getenv('N_BITS', '20000'))     # symbols to generate
stimulus = os.getenv('STIMULUS', 'MSEQ16')       # stim type
ns       = float(os.getenv('NS', '4'))           # final samples / symbol
alpha    = float(os.getenv('ALPHA', '0.3'))      # pulse shape (deprecated for PAM, useful for experiments)
snr      = float(os.getenv('SNR', '30'))         # SNR
delay    = float(os.getenv('DELAY', (2 * ns * (np.random.rand() - 0.5)))) # fractional delay to add
use_mm   = int(os.getenv('USE_MM', '1')) != 0
ns_uncertainty_factor = float(os.getenv('NS_UNCERTAINTY_FACTOR', '1e-4'))
extra_loop_latency    = int(os.getenv('EXTRA_LOOP_LATENCY', '0'))
ui_rate               = float(os.getenv('UI_RATE', '32e9'))
loop_bw               = float(os.getenv('LOOP_BW', '5e6'))
damping_factor      = float(os.getenv('DAMPING_FACTOR', '1'))
default_k0            = 6.4 if use_mm else 6
k0                    = float(os.getenv('K0', default_k0))
timing_jitter_sa_std  = float(os.getenv('TIMING_JITTER_SA_STD', 0.00001))

# loop filter settings - kp is proportional gain, ki is integral gain
T_loop  = 1.0 / ui_rate
wc_loop = 2 * np.pi * loop_bw
wn_loop = wc_loop / np.sqrt(2*(damping_factor**2) + np.sqrt(((2*(damping_factor**2))**2)+1) - 1)
kp = 2 * damping_factor * wn_loop * T_loop
ki = ((wn_loop * T_loop)**2)

lock_symbols = int(os.getenv('LOCK_SYMBOLS', '10000'))

plot = int(os.getenv('PLOT', '0')) != 0

# lets always generate with ONS 'original ns' and then use lagrange to decimate into our receiver
# The matched filter 'b' is also resampled and used on that domain
found_period = re.search(r'PERIOD(\d+)', stimulus)
if found_period:
    d = np.zeros(n_bits)
    period = int(found_period.group(1))
    for i in range(len(d)):
      d[i] = (0 if i % (2 * period) < period else 1)
    print(f"STIMULUS={stimulus}, generating a square wave with period {period}")
found_msequence = None if found_period else re.search(r'MSEQ(\d+)', stimulus)
if found_msequence:
    b = int(found_msequence.group(1))
    m = ss.m_seq(b)
    while (len(m) < n_bits):
        m = np.concatenate([m, m])
    d = m[0:n_bits]
    print(f"STIMULUS={stimulus}, generating an msequence of {b} bits")
if not found_msequence and not found_period:
    print(f"Unknown STIMULUS={stimulus}, Should be of form STIMULUS=PERIOD<Period> or STIMULUS=MSEQ<Bits>")
    exit(1)

# This is essentially equivalent to resampling the Tx signal
thought_ns = float(os.getenv('THOUGHT_NS', (1 + ns_uncertainty_factor*(np.random.rand() - 0.5)) * ns))

print(
      f"n_bits                = {n_bits}\n"\
      f"use_mm                = {use_mm}\n"\
      f"ns                    = {ns}\n"\
      f"ns_uncertainty_factor = {ns_uncertainty_factor}\n"\
      f"thought_ns            = {thought_ns}\n"\
      f"alpha                 = {alpha}\n"\
      f"snr                   = {snr}dB\n"\
      f"delay                 = {delay}\n"\
      f"extra_loop_latency    = {extra_loop_latency}\n"\
      f"ui_rate               = {ui_rate}\n"\
      f"loop_bw               = {loop_bw}\n"\
      f"k0                    = {k0}\n"\
      f"damping_factor        = {damping_factor}\n"\
      f"timing_jitter_sa_std  = {timing_jitter_sa_std}")

x,b = ss.nrz_bits2(d, ONS, pulse='src', alpha=alpha)

yn_len = int((n_bits * ONS) // ns)
yn = np.zeros(yn_len)
for i in range(yn_len):
    yn[i] = interpolated_index(x, ((i + delay + np.random.normal(0, timing_jitter_sa_std)) * ONS / ns), 3)
yn = dc.cpx_awgn(yn, snr, ns)

bn_len = int(len(b) * ONS // ns)
bn = np.zeros(bn_len)
for i in range(bn_len):
    bn[i] = interpolated_index(b, ((i) * ONS / ns), 3) * float(ONS) / ns

e = np.convolve(bn, yn)

received           = np.zeros(len(d))
received_quantized = np.zeros(len(d))
timing_error       = np.zeros(len(d))
loop_integral      = np.zeros(len(d))
control_effort     = np.zeros(len(d) + extra_loop_latency)

candidate_index = 0.01
ri              = -1
early = late = candidate = 0

while candidate_index + ns < len(e) and ri + 1 < len(received):
    candidate = interpolated_index(e, candidate_index, 3)
    if not use_mm:
        early     = late
        late      = interpolated_index(e, candidate_index + thought_ns/2, 3)

    ri += 1

    received[ri] = candidate
    received_quantized[ri] = 1 if candidate > 0 else -1

    if not use_mm:
        timing_error[ri] = k0 * early_late_ted(early, candidate, late)
    else:
        timing_error[ri] = k0 * mm_ted(received[ri], received_quantized[ri], received[ri-1], received_quantized[ri-1])
    loop_integral[ri]    = timing_error[ri-1] + loop_integral[ri-1]
    control_effort[ri]   = (kp * timing_error[ri]) + (ki * loop_integral[ri]) + control_effort[ri - 1]

    candidate_index  = (ri + 1) * thought_ns + control_effort[ri]

print(f"Debug: min_control_effort: {min(control_effort)}, max_control_effort: {max(control_effort)}, control_effort[{ri}]={control_effort[ri]}")

# Plotting - find the autocorrelation and then use this to try to line things up
d_q = 2 * (d - 0.5)
c = np.correlate(received_quantized[lock_symbols:], d_q[lock_symbols:], mode='full')
shift = len(d) - lock_symbols - c.argmax() - 1

x_d = np.arange(0, len(d))
x_r = np.arange(shift, shift + ri)
x_r = x_r[0:len(d)]

bits = 0
bit_errors = 0
lock_errors = 0
print(f"detected a shift of {shift}")
print(f"Debug: min(x_r): {min(x_r)}, max_xr: {max(x_r)}, len(x_r)={len(x_r)}, len(d_q)={len(d_q)}, len(received_quantized)={len(received_quantized)}, ri={ri}")
for i,xi in enumerate (x_r):
    if i > 0 and xi < len(d_q):
        bits = bits + 1
        if d_q[xi] * received_quantized[i] < 0:
            bit_errors = bit_errors + 1
            if xi >= lock_symbols:
                lock_errors = lock_errors + 1

print(f"Errors after the first {lock_symbols} symbols {lock_errors}/{bits - lock_symbols}")

if not plot:
    exit(0)

plt.figure(figsize=(6, 12))

ax_0 = plt.subplot(3, 1, 1)
ax_0.plot(x_r, received[0:ri], 'b.')
ax_0.set_title(r'Equalizer Output $y[n]$')
ax_0.set_ylabel('Amplitude')
ax_0.grid()

ax_1 = plt.subplot(3, 1, 2, sharex=ax_0)
ax_1.plot(x_r, timing_error[0:ri])
ax_1.set_title('TED Output $e[n]$')
ax_1.set_ylabel('Amplitude (ADC Samples)')
ax_1.grid()

ax_2 = plt.subplot(3, 1, 3, sharex=ax_0)
ax_2.plot(x_r, control_effort[0:ri])
ax_2.set_title(r'Loop Filter Output $l[n]$')
ax_2.set_ylabel('Amplitude (ADC Samples)')
ax_2.set_xlabel('UI index')
ax_2.grid()

plt.tight_layout()
plt.show()
