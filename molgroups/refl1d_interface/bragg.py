"""
Bragg peak contributions to reflectivity curves.

Provides standalone peak-shape dataclasses and experiment classes that add
a Bragg peak on top of the standard reflectivity calculation.

Classes:
    BraggPeak: Abstract base for peak-shape models.
    GaussianBraggPeak: Gaussian peak shape.
    LorentzianBraggPeak: Lorentzian (Cauchy) peak shape.
    VoigtBraggPeak: Voigt (Gaussian * Lorentzian convolution) peak shape.
    BraggExperiment: Standard Refl1D experiment with a Bragg peak.
    BraggMolgroupsExperiment: MolgroupsExperiment with a Bragg peak.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
from scipy.special import voigt_profile

from bumps.parameter import Parameter
from refl1d.experiment import Experiment as Refl1DExperiment
from refl1d.probe import ProbeSet

from .experiment import MolgroupsExperiment
from .layers import MolgroupsStack


# =============================================================================
# 1. PEAK SHAPE MODELS
# =============================================================================

@dataclass
class BraggPeak:
    """
    Abstract base class for Bragg peak models.

    Subclasses implement `_shape(Q)`, which returns a normalised lineshape
    (peak value of 1 at q0). The base class multiplies by `scale` and adds
    `background`.
    """
    q0: Union[Parameter, float] = 0.1
    scale: Union[Parameter, float] = 1e-4
    background: Union[Parameter, float] = 0.0

    def __post_init__(self) -> None:
        for name in ['q0', 'scale', 'background']:
            val = getattr(self, name)
            if not hasattr(val, 'name'):
                setattr(self, name, Parameter.default(val, name=f'bragg_{name}'))

    def _shape(self, Q: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement _shape")

    def calculate(self, Q: np.ndarray) -> np.ndarray:
        return self.scale.value * self._shape(Q) + self.background.value  # type: ignore

    @property
    def parameters(self) -> Dict[str, Parameter]:
        return {f'bragg_{n}': getattr(self, n) for n in ['q0', 'scale', 'background']}


@dataclass
class GaussianBraggPeak(BraggPeak):
    """Gaussian peak: exp(-0.5 * ((Q - q0) / sigma)^2)."""
    sigma: Union[Parameter, float] = 0.01

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.sigma, Parameter):
            self.sigma = Parameter.default(self.sigma, name='bragg_sigma')

    def _shape(self, Q: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * ((Q - self.q0.value) / self.sigma.value) ** 2)  # type: ignore

    @property
    def parameters(self) -> Dict[str, Parameter]:
        return super().parameters | {'bragg_sigma': self.sigma}  # type: ignore


@dataclass
class LorentzianBraggPeak(BraggPeak):
    """Lorentzian peak: gamma^2 / ((Q - q0)^2 + gamma^2)."""
    gamma: Union[Parameter, float] = 0.01

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.gamma, Parameter):
            self.gamma = Parameter.default(self.gamma, name='bragg_gamma')

    def _shape(self, Q: np.ndarray) -> np.ndarray:
        g = self.gamma.value  # type: ignore
        return g ** 2 / ((Q - self.q0.value) ** 2 + g ** 2)  # type: ignore

    @property
    def parameters(self) -> Dict[str, Parameter]:
        return super().parameters | {'bragg_gamma': self.gamma}  # type: ignore


@dataclass
class VoigtBraggPeak(BraggPeak):
    """
    Voigt peak via scipy.special.voigt_profile.

    The Voigt profile is the exact convolution of a Gaussian (width sigma)
    and a Lorentzian (half-width gamma). The result is normalised to a peak
    value of 1 at Q = q0.
    """
    sigma: Union[Parameter, float] = 0.01
    gamma: Union[Parameter, float] = 0.01

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ['sigma', 'gamma']:
            val = getattr(self, name)
            if not hasattr(val, 'name'):
                setattr(self, name, Parameter.default(val, name=f'bragg_{name}'))

    def _shape(self, Q: np.ndarray) -> np.ndarray:
        sig, gam = self.sigma.value, self.gamma.value  # type: ignore
        peak_value = voigt_profile(0.0, sig, gam)
        if peak_value == 0:
            return np.zeros_like(Q)
        return voigt_profile(Q - self.q0.value, sig, gam) / peak_value  # type: ignore

    @property
    def parameters(self) -> Dict[str, Parameter]:
        return super().parameters | {'bragg_sigma': self.sigma, 'bragg_gamma': self.gamma}  # type: ignore


# =============================================================================
# 2. EXPERIMENT CLASSES
# =============================================================================

def _get_Q(probe) -> np.ndarray:
    if isinstance(probe, ProbeSet):
        return np.hstack([p.Q for p in probe.probes])
    return probe.Q


@dataclass(init=False)
class BraggExperiment(Refl1DExperiment):
    """Standard Refl1D experiment with an additive Bragg peak."""
    bragg: Optional[BraggPeak] = None

    def __init__(self,
                 bragg: Optional[BraggPeak] = None,
                 sample=None,
                 probe=None,
                 name=None,
                 roughness_limit=0,
                 dz=None,
                 dA=None,
                 step_interfaces=None,
                 smoothness=None,
                 interpolation=0,
                 constraints=None,
                 version=None,
                 auto_tag=False):
        super().__init__(sample, probe, name, roughness_limit, dz, dA,
                         step_interfaces, smoothness, interpolation,
                         constraints, version, auto_tag)
        self.bragg = bragg

    def reflectivity(self, resolution=True, interpolation=0):
        Q, Rq = super().reflectivity(resolution, interpolation)
        if self.bragg is not None:
            Rq = Rq + self.bragg.calculate(_get_Q(self.probe))
        return Q, Rq

    def parameters(self):
        base = super().parameters()
        if self.bragg is not None:
            return base | {'bragg': self.bragg.parameters}
        return base


@dataclass(init=False)
class BraggMolgroupsExperiment(MolgroupsExperiment):
    """MolgroupsExperiment with an additive Bragg peak."""
    bragg: Optional[BraggPeak] = None

    def __init__(self,
                 bragg: Optional[BraggPeak] = None,
                 sample: Optional[MolgroupsStack] = None,
                 probe=None,
                 name=None,
                 roughness_limit=0,
                 dz=None,
                 dA=None,
                 step_interfaces=None,
                 smoothness=None,
                 interpolation=0,
                 constraints=None,
                 version=None,
                 auto_tag=False):
        super().__init__(sample, probe, name, roughness_limit, dz, dA,
                         step_interfaces, smoothness, interpolation,
                         constraints, version, auto_tag)
        self.bragg = bragg

    def reflectivity(self, resolution=True, interpolation=0):
        Q, Rq = super().reflectivity(resolution, interpolation)
        if self.bragg is not None:
            Rq = Rq + self.bragg.calculate(_get_Q(self.probe))
        return Q, Rq

    def parameters(self):
        base = super().parameters()
        if self.bragg is not None:
            return base | {'bragg': self.bragg.parameters}
        return base
