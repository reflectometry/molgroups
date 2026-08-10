"""Minimal test for BraggExperiment with synthetic data."""

import numpy as np
from refl1d.names import Parameter, SLD, Slab, FitProblem
from refl1d.probe import QProbe

from molgroups.refl1d_interface.bragg import (
    GaussianBraggPeak,
    BraggExperiment,
)

# === Synthetic probe ===
Q = np.linspace(0.01, 0.15, 200)
dQ = 0.001 * np.ones_like(Q)

# Flat slab stack: silicon substrate / thin film / D2O
silicon = SLD(name='silicon', rho=2.07)
film    = SLD(name='film',    rho=4.0)
d2o     = SLD(name='d2o',     rho=6.36)

sample = silicon() | Slab(material=film, thickness=100, interface=3) | d2o()

# Generate synthetic R(Q) with a Gaussian Bragg peak added
true_peak = GaussianBraggPeak(q0=0.08, sigma=0.005, scale=5e-5, background=0.0)

probe = QProbe(Q, dQ, name='synthetic')
bare_exp = BraggExperiment(sample=sample, probe=probe)
_, R_bare = bare_exp.reflectivity()
R_synth = R_bare + true_peak.calculate(Q)

# Add noise and assign to probe
rng = np.random.default_rng(42)
dR = 0.02 * R_synth
probe = QProbe(Q, dQ, data=(R_synth + rng.normal(0, dR), dR), name='synthetic')

# === Model with fittable Bragg peak ===
peak = GaussianBraggPeak(
    q0         = Parameter(value=0.079, name='bragg_q0'   ).range(0.06, 0.10),
    sigma      = Parameter(value=0.006, name='bragg_sigma' ).range(0.001, 0.02),
    scale      = Parameter(value=4e-5,  name='bragg_scale' ).range(0.0, 1e-3),
    background = 0.0,
)

model = BraggExperiment(sample=sample, probe=probe)
model.bragg = peak

problem = FitProblem(model)
problem.name = 'bragg_test'
