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

import concurrent.futures
import multiprocessing
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import dill
import numpy as np
import plotly.graph_objs as go
from scipy.special import voigt_profile

from bumps.dream.state import MCMCDraw
from bumps.parameter import Parameter
from bumps.plotutil import form_quantiles

try:
    from bumps.plots.custom_plot import CustomWebviewPlot
    from bumps.plots.colors import COLORS
except ImportError:  # CRUFT: bumps pre-1.1
    from bumps.webview.server.custom_plot import CustomWebviewPlot
    from bumps.webview.server.colors import COLORS

from refl1d.experiment import Experiment as Refl1DExperiment
from refl1d.probe import ProbeSet

from .experiment import MolgroupsExperiment
from .layers import MolgroupsStack
from .plots import hex_to_rgb


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
        if not hasattr(self.sigma, 'name'):
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
        if not hasattr(self.gamma, 'name'):
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


def _register_bragg_plots(experiment) -> None:
    experiment.register_webview_plot(
        plot_title='Bragg Decomposition',
        plot_function=bragg_decomposition_plot,
        change_with='parameter',
    )
    experiment.register_webview_plot(
        plot_title='Bragg Decomposition with Uncertainty',
        plot_function=bragg_uncertainty_plot,
        change_with='uncertainty',
    )


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
        _register_bragg_plots(self)

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
        _register_bragg_plots(self)

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


# =============================================================================
# 3. PLOTTING FUNCTIONS
# =============================================================================

def _decompose(model) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (Q, total, bragg, refl) arrays for the current parameter state."""
    Q_raw, total_raw = model.reflectivity()
    total = np.ravel(np.array(total_raw, dtype=float))
    Q = np.ravel(np.array(Q_raw, dtype=float))
    bragg = model.bragg.calculate(Q) if model.bragg is not None else np.zeros_like(Q)
    return Q, total, bragg, total - bragg


def bragg_decomposition_plot(model, problem=None) -> CustomWebviewPlot:
    """Plotly decomposition: data, total theory, pure reflectivity, Bragg peak."""
    Q, total, bragg, refl = _decompose(model)
    probes = model.probe.probes if isinstance(model.probe, ProbeSet) else [model.probe]

    fig = go.Figure()
    cursor = 0
    csv_rows = ['Q,R,dR,Total,Reflectivity,Bragg']

    for i, probe in enumerate(probes):
        n = len(probe.Q)
        sl = slice(cursor, cursor + n)
        Q_i, total_i, bragg_i, refl_i = Q[sl], total[sl], bragg[sl], refl[sl]
        R_i  = np.ravel(probe.R)  if probe.R  is not None else np.zeros(n)
        dR_i = np.ravel(probe.dR) if probe.dR is not None else np.zeros(n)
        color = COLORS[i % len(COLORS)]

        fig.add_trace(go.Scatter(
            x=Q_i, y=R_i,
            error_y=dict(type='data', array=dR_i, visible=True, color=color, thickness=1),
            mode='markers', name=f'Data {i+1}',
            marker=dict(color=color, size=6, opacity=0.4), legendgroup=f'g{i}'))
        fig.add_trace(go.Scatter(
            x=Q_i, y=total_i, mode='lines', name=f'Total {i+1}',
            line=dict(color=color, width=3), legendgroup=f'g{i}'))
        fig.add_trace(go.Scatter(
            x=Q_i, y=refl_i, mode='lines', name=f'Reflectivity {i+1}',
            line=dict(color=color, width=2, dash='dash'), legendgroup=f'g{i}'))
        fig.add_trace(go.Scatter(
            x=Q_i, y=bragg_i, mode='lines', name=f'Bragg {i+1}',
            line=dict(color=color, width=2, dash='dot'), legendgroup=f'g{i}'))

        for q, r, dr, t, rv, b in zip(Q_i, R_i, dR_i, total_i, refl_i, bragg_i):
            csv_rows.append(f'{q:.6e},{r:.6e},{dr:.6e},{t:.6e},{rv:.6e},{b:.6e}')
        cursor += n

    fig.update_layout(
        title=f'Bragg Decomposition: {model.name}',
        xaxis_title='Q (Å⁻¹)',
        yaxis=dict(title='Intensity', type='log', exponentformat='power', showexponent='all', range=[-10, None]),
        template='plotly_white',
        legend=dict(x=0.01, y=0.01, xanchor='left', yanchor='bottom', bgcolor='rgba(255,255,255,0.8)'),
    )
    return CustomWebviewPlot(fig_type='plotly', plotdata=fig, exportdata='\n'.join(csv_rows))


# --- Uncertainty plot worker infrastructure ---

_bragg_shared_problem = None
_bragg_model_index: int = 0


def _bragg_initialize_worker(serialized_problem, model_index: int) -> None:
    global _bragg_shared_problem, _bragg_model_index
    _bragg_shared_problem = dill.loads(serialized_problem[:])
    _bragg_model_index = model_index


def _bragg_worker_calc(point: np.ndarray):
    """Evaluate one MCMC draw; returns (total, bragg, refl) arrays."""
    _bragg_shared_problem.setp(point)
    model = list(_bragg_shared_problem.models)[_bragg_model_index]
    model.update()
    model.nllf()
    Q_raw, total_raw = model.reflectivity()
    total = np.ravel(np.array(total_raw, dtype=float))
    Q = _get_Q(model.probe)
    bragg = model.bragg.calculate(Q) if model.bragg is not None else np.zeros_like(Q)
    return total, bragg, total - bragg


def bragg_uncertainty_plot(model, problem=None, state: Optional[MCMCDraw] = None, n_samples: int = 50) -> CustomWebviewPlot:
    """Bragg decomposition with 68% credible interval bands from MCMC draws."""
    if state is None:
        return bragg_decomposition_plot(model, problem)

    print('Starting Bragg uncertainty calculation...')
    t0 = time.time()

    points = state.draw().points
    n_samples = min(n_samples, points.shape[0])
    points = points[np.random.permutation(len(points) - 1)][-n_samples:-1]

    model_index = list(problem.models).index(model)

    with multiprocessing.Manager() as manager:
        shared = manager.Array('B', dill.dumps(problem))
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=None,
            initializer=_bragg_initialize_worker,
            initargs=(shared, model_index),
        ) as executor:
            results = list(executor.map(_bragg_worker_calc, points))

    print(f'Bragg uncertainty done in {time.time() - t0:.1f}s')

    totals = [r[0] for r in results]
    braggs = [r[1] for r in results]
    refls  = [r[2] for r in results]

    Q = _get_Q(model.probe)
    probes = model.probe.probes if isinstance(model.probe, ProbeSet) else [model.probe]

    fig = go.Figure()
    csv_rows = ['Q,R,dR,Total_median,Total_lo68,Total_hi68,Refl_median,Refl_lo68,Refl_hi68,Bragg_median,Bragg_lo68,Bragg_hi68']

    cursor = 0
    for i, probe in enumerate(probes):
        n = len(probe.Q)
        sl = slice(cursor, cursor + n)
        Q_i  = Q[sl]
        R_i  = np.ravel(probe.R)  if probe.R  is not None else np.zeros(n)
        dR_i = np.ravel(probe.dR) if probe.dR is not None else np.zeros(n)
        color = COLORS[i % len(COLORS)]
        rgb = ','.join(map(str, hex_to_rgb(color)))

        def _bands(samples: List[np.ndarray], sl=sl):
            sub = [s[sl] for s in samples]
            med = np.median(sub, axis=0)
            _, (qs,) = form_quantiles(sub, (68,))
            lo, hi = qs
            return med, lo, hi

        total_med, total_lo, total_hi   = _bands(totals)
        bragg_med, bragg_lo, bragg_hi   = _bands(braggs)
        refl_med,  refl_lo,  refl_hi    = _bands(refls)

        # Data (fixed)
        fig.add_trace(go.Scatter(
            x=Q_i, y=R_i,
            error_y=dict(type='data', array=dR_i, visible=True, color=color, thickness=1),
            mode='markers', name=f'Data {i+1}',
            marker=dict(color=color, size=6, opacity=0.4), legendgroup=f'g{i}'))

        for label, med, lo, hi, dash in [
            (f'Total {i+1}',       total_med, total_lo, total_hi, 'solid'),
            (f'Reflectivity {i+1}', refl_med,  refl_lo,  refl_hi,  'dash'),
            (f'Bragg {i+1}',       bragg_med, bragg_lo, bragg_hi, 'dot'),
        ]:
            # CI band: add hi first, then lo fills tonexty (= fills up to hi)
            fig.add_trace(go.Scatter(
                x=Q_i, y=hi, mode='lines', line=dict(width=0),
                legendgroup=f'g{i}', showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(
                x=Q_i, y=lo, mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor=f'rgba({rgb},0.25)',
                legendgroup=f'g{i}', showlegend=False, hoverinfo='skip'))
            # Median line
            fig.add_trace(go.Scatter(
                x=Q_i, y=med, mode='lines', name=label,
                line=dict(color=color, width=2, dash=dash), legendgroup=f'g{i}'))

        for j in range(n):
            csv_rows.append(
                f'{Q_i[j]:.6e},{R_i[j]:.6e},{dR_i[j]:.6e},'
                f'{total_med[j]:.6e},{total_lo[j]:.6e},{total_hi[j]:.6e},'
                f'{refl_med[j]:.6e},{refl_lo[j]:.6e},{refl_hi[j]:.6e},'
                f'{bragg_med[j]:.6e},{bragg_lo[j]:.6e},{bragg_hi[j]:.6e}'
            )
        cursor += n

    fig.update_layout(
        title=f'Bragg Decomposition with Uncertainty: {model.name}',
        xaxis_title='Q (Å⁻¹)',
        yaxis=dict(title='Intensity', type='log', exponentformat='power', showexponent='all', range=[-10, None]),
        template='plotly_white',
        legend=dict(x=0.01, y=0.01, xanchor='left', yanchor='bottom', bgcolor='rgba(255,255,255,0.8)'),
    )
    return CustomWebviewPlot(fig_type='plotly', plotdata=fig, exportdata='\n'.join(csv_rows))
