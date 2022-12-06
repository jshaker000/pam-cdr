#!/usr/bin/env python3
# Equation 8-15
import numpy as np
from   scipy import signal
import matplotlib.pyplot as plt

fs = 128e9

a_dc_db = -9
a_dc    = 10**(a_dc_db / 20.0)

wz1 = 2 * np.pi * 250e6
wz3 = 2 * np.pi * 7.7e9

wp1 = 1.30 * wz1
wp2 = 2 * np.pi * 7.7e9
wp3 = 2 * np.pi * 22e9
wp4 = 2 * np.pi * 28e9
wp5 = 2 * np.pi * 32e9
wp6 = 2 * np.pi * 32e9

wz2 = abs(a_dc) * wp2

z = [-wz1, -wp2*a_dc, -wz3]
p = [-wp1, -wp2, -wp3, -wp4, -wp5, -wp6]
g  = (wp1 * wp3 * wp4 * wp5 * wp6) / (wz1 * wz3)


z_z, p_z, k_z = signal.bilinear_zpk(z, p, g, fs)
b_z, a_z      = signal.zpk2tf(z_z, p_z, k_z)
print(f"discrete zeros: {z_z}\ndiscrete poles: {p_z}")

with open("ctle.txt", "w") as clte_f:
    for x in b_z:
        clte_f.write(f" {x}")
    clte_f.write("\n")
    for x in a_z:
        clte_f.write(f" {x}")
    clte_f.write("\n")

f = np.arange(0, fs/2, 1e7)

plt.figure(figsize=(6,4))
ax = plt.subplot(1, 1, 1)

w, H1 = signal.freqz(b_z, a_z, 2*np.pi*f/fs)
ax.semilogx(f, 20*np.log10(np.abs(H1)))
ax.set_ylim([-40, 10])
ax.set_xlim([10e6, fs/2])
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Gain (dB)')
ax.grid()
plt.show()

channel_low_loss = np.zeros(128000)
with open("impulse_responses/impulse_victim_differential_low_loss.txt", "r") as file:
    for i,li in enumerate(file):
        channel_low_loss[i] = float(li)

channel = np.zeros(128000)
with open("impulse_responses/impulse_victim_differential.txt", "r") as file:
    for i,li in enumerate(file):
        channel[i] = float(li)

plt.figure(figsize=(6,4))
ax = plt.subplot(1, 1, 1)

w, H1 = signal.freqz(channel_low_loss, [1], 2*np.pi*f/fs)
plt.semilogx(f, 20*np.log10(np.abs(H1)), label='Response pre-CTLE')

w, H1 = signal.freqz(signal.convolve(b_z, channel_low_loss), a_z, 2*np.pi*f/fs)
plt.semilogx(f, 20*np.log10(np.abs(H1)), label='Response post-CTLE')

ax.set_ylim([-40, 10])
ax.set_xlim([10e6, fs/2])
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Gain (dB)')
ax.legend()
ax.grid()

plt.show()

plt.figure(figsize=(6,4))
ax = plt.subplot(1, 1, 1)

w, H1 = signal.freqz(channel, [1], 2*np.pi*f/fs)
ax.semilogx(f, 20*np.log10(np.abs(H1)), label='Response pre-CTLE')

w, H1 = signal.freqz(signal.convolve(b_z, channel), a_z, 2*np.pi*f/fs)
ax.semilogx(f, 20*np.log10(np.abs(H1)), label='Response post-CTLE')

ax.set_ylim([-40, 10])
ax.set_xlim([10e6, fs/2])
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Gain (dB)')
ax.legend()
ax.grid()

plt.show()
