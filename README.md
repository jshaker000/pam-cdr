# pam-cdr

This code is designed offer simulations to help one understand and model high speed PAM-N systems (specifically PCIe systems) and is also supplemental material to my thesis titled [Study and simulation of PCIe 6, with an emphasis on the physical layer](https://www.proquest.com/dissertations-theses/study-simulation-pcie-6-with-emphasis-on-physical/docview/2755778475/se-2).

The `impulse_responses` directory contains impulses responses sampled at 128GSa/s of realistic channels from the PCIe 6.0 standard, and include a low loss channel
and a normal channel and example impulse responses of an aggressor (ie an adjacent cross talk lane). These impulse responses came from extracting s-parameters from
Keysight ADS on differential pairs and then combining them appropriately to represent a response that effectively involes going from a single ended signal through to a differential channel and then being converted
back to a single ended signal.

To run these examples its required to have Python3 or above, and the packages matplotlib, numpy, scipy, and scikit-dsp-comm are requred.

- `initial_cdr` investigates performing clock data recovery with either an early late or a MM TED and a second order loop filter on a relatively ideal, raised cosine pulse shape and supports a wide variety of environment variables to alter its behavior.
- `ctle.py` generates the bilinear transformed poles and zeros of the Continuous Time Linear Equalizer as described by the PCIe 6.0 standard
- `test_equalization.py` and `test_equalization_low_loss.py` run an LMS algorithm to determine DD FFE and DFE coefficients for the medium and low loss  channels respectively and support a wide variety of environment variables. Note that at 4 Samples/Symbol and PAM-4 the medium loss channel need `Aopt` to be initialized from coefficients determined from running a PAM-2 simulation.
Here hand-of-God timing synchronization is done through the `group_delay_fude_factor` variable (which may need to tuned for different filter choices). There are also options to add an aggressor channel into the system.
- `aggregate_system.py` and `aggregate_system_low_loss.py` run simultaneous timing recovery and LMS equalization on a receiver for the medium and low loss channels respectively and support a wide variety of environment variables. This shows the complex dynamics of performing simultaneous channel equalization and timing estimations. These simulations are oftentimes aided by
providing initial tap weights `Aopt` from a run of the `test_equalization` simulation to aid in simultaneous convergence.
