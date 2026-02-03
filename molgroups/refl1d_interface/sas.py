""" Module for interfacing combined reflectivity and small angle scattering models
"""

from dataclasses import dataclass, field
import copy

import numpy as np
import plotly.graph_objs as go

from bumps.parameter import Parameter
from bumps.webview.server.custom_plot import CustomWebviewPlot
from refl1d.experiment import Experiment
from refl1d.probe import ProbeSet
from refl1d.probe.resolution import dTdL2dQ, sigma2FWHM
from refl1d.webview.server.colors import COLORS

from sasmodels.core import load_model
from sasmodels.direct_model import DirectModel
from sasmodels.data import Data1D

from .experiment import MolgroupsExperiment

# --- 1. THE INTERFACE (Base Class) ---

class SASModel:
    """
    Base class for SAS calculation engines.
    Subclasses must implement bind() and calculate().
    """
    def bind(self, probe):
        """
        Associate the model with a Probe (or ProbeSet).
        Use this to perform expensive setup (like compiling kernels).
        """
        raise NotImplementedError("Subclasses must implement bind()")

    def calculate(self):
        """
        Return the I(Q) array matching the bound probe.
        """
        raise NotImplementedError("Subclasses must implement calculate()")
    
    @property
    def parameters(self):
        """
        Return a dictionary of Bumps Parameter objects.
        """
        return {}


# --- 2. CONCRETE IMPLEMENTATION (Standard Sasmodels) ---

@dataclass
class StandardSASModel(SASModel):
    """
    A SAS model that uses the standard sasmodels library (DirectModel).
    Manages its own pickling state to handle C-pointers safely.
    Params should match the parameters expected by the model.
    """
    sas_model_name: str
    params: dict[str, float | Parameter] = field(default_factory=dict)
    dtheta_l: float | None = None
    
    # Internal state (excluded from __init__)
    _engines: list[DirectModel] | None = field(default=None, init=False, repr=False)
    _probe: object = field(default=None, init=False, repr=False)

    def __post_init__(self):
        # Ensure all inputs in params are converted to Bumps Parameters.
        # This runs immediately on instantiation, so 'params' always holds objects.
        for k, v in self.params.items():
            if not isinstance(v, Parameter):
                self.params[k] = Parameter.default(v, name=k)

    def bind(self, probe):
        """
        Store the probe and initialize the DirectModel engines.
        """
        self._probe = probe
        # Clear existing engines so they are rebuilt for the new probe
        self._engines = None 
        # Build immediately for speed
        self._build_engines()

    def calculate(self):
        """
        Lazy-load engines if missing (e.g. after unpickling), then calculate.
        """
        if self._engines is None:
            self._build_engines()
            
        # Run calculation across all engines
        if not self._engines:
            return np.array([])
            
        pars = {k: v.value for k, v in self.params.items()}
        parts = [model(**pars) for model in self._engines]
        return np.hstack(parts)

    def _build_engines(self):
        """
        The compilation logic. Rebuilds _engines using the bound probe and kernel.
        """
        if not self.sas_model_name or self._probe is None:
            self._engines = []
            return

        kernel = load_model(self.sas_model_name)
        
        # Handle ProbeSet vs Single Probe
        probes = [self._probe] if not isinstance(self._probe, ProbeSet) else self._probe.probes

        # Handle dtheta_l broadcasting
        if np.isscalar(self.dtheta_l) or self.dtheta_l is None:
            dtheta_list = [self.dtheta_l] * len(probes)
        else:
            dtheta_list = self.dtheta_l
        
        new_engines = []
        for probe, dt in zip(probes, dtheta_list):
            data = Data1D(x=probe.Q)
            # Resolution calculation
            data.dxl = dTdL2dQ(np.zeros_like(probe.T), dt, probe.L, probe.dL)
            data.dxw = 2 * sigma2FWHM(probe.dQ) if hasattr(probe, 'dQ') else np.zeros_like(probe.Q)
            
            new_engines.append(DirectModel(data=data, model=kernel))
            
        self._engines = new_engines

    def __getstate__(self):
        """
        Custom pickling: Drop the C-pointer objects (_engines).
        IMPORTANT: 'params' IS preserved here, allowing Bumps to handle 
        parameter state persistence during serialization.
        """
        state = self.__dict__.copy()
        state['_engines'] = None 
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # _engines is None; will be rebuilt by calculate()

    @property
    def parameters(self):
        return self.params


# --- 3. THE MIXIN ---

class SASReflectivityMixin:
    """
    Mixin class that adds SAS capabilities to ANY Refl1D Experiment.
    Delegates calculation to an instance of SASModel.
    """
    
    sas_model: SASModel | None
    _cache: dict
    probe: object
    name: str

    def _init_sas(self, sas_model: SASModel | None):
        self.sas_model = sas_model
        
        if self.sas_model is not None:
            # BINDING: Pass the experiment's probe to the model
            self.sas_model.bind(self.probe)
        
        # Register plots
        self.register_webview_plot(
            plot_title='SAS/Refl Decomposition',
            plot_function=sas_decomposition_plot,
            change_with='parameter'
        )

    def parameters(self):
        # Merge experiment params with the model's params
        base = super().parameters()
        if self.sas_model:
            return base | {'sas': self.sas_model.parameters}
        return base

    def sas(self):
        """ Calculate the small angle scattering I(q) """
        key = ("small_angle_scattering")
        if key not in self._cache:
             if self.sas_model:
                 self._cache[key] = self.sas_model.calculate()
             else:
                 # Fallback for plotting if no model exists
                 if isinstance(self.probe, ProbeSet):
                     n = sum(len(p.Q) for p in self.probe.probes)
                 else:
                     n = len(self.probe.Q)
                 self._cache[key] = np.zeros(n)
                 
        return self._cache[key]

    def reflectivity(self, resolution=True, interpolation=0):
        Q, Rq = super().reflectivity(resolution, interpolation)
        if self.sas_model is not None:
            # Add SAS signal (create new array, do not modify in-place)
            Rq = Rq + self.sas()
        return Q, Rq


# --- 4. CONCRETE EXPERIMENT CLASSES ---

@dataclass(init=False)
class SASReflectivityExperiment(SASReflectivityMixin, Experiment):
    """
    Standard SAS + Reflectivity Experiment.
    """
    sas_model: SASModel | None = None

    def __init__(self, sas_model: SASModel | None = None, sample=None, probe=None, name=None, **kwargs):
        super().__init__(sample, probe, name, **kwargs)
        self._init_sas(sas_model)

@dataclass(init=False)
class SASReflectivityMolgroupsExperiment(SASReflectivityMixin, MolgroupsExperiment):
    """
    Molgroups-Enabled SAS + Reflectivity Experiment.
    """
    sas_model: SASModel | None = None

    def __init__(self, sas_model: SASModel | None = None, sample=None, probe=None, name=None, **kwargs):
        super().__init__(sample, probe, name, **kwargs)
        self._init_sas(sas_model)


# --- 5. PLOTTING FUNCTION ---

def sas_decomposition_plot(model: SASReflectivityExperiment, problem=None) -> CustomWebviewPlot:
    """
    Webview plot that shows the decomposition of the signal into 
    Reflectivity (Rq) and SAS (Iq) components.
    """

    def to_flat(arr):
        if arr is None: return np.array([])
        return np.ravel(np.array(arr, dtype=float))

    Q_all_raw, total_theory_raw = model.reflectivity()
    Q_all = to_flat(Q_all_raw)
    total_theory = to_flat(total_theory_raw)
    
    if model.sas_model is not None:
        Iq_all = to_flat(model.sas())
    else:
        Iq_all = np.zeros_like(Q_all)
    Rq_all = total_theory - Iq_all

    if isinstance(model.probe, ProbeSet):
        probes = model.probe.probes
    else:
        probes = [model.probe]

    fig = go.Figure()
    cursor = 0
    for i, probe in enumerate(probes):
        n_points = len(probe.Q)
        start = cursor
        end = cursor + n_points
        Q = Q_all[start:end]
        Total = total_theory[start:end]
        Rq = Rq_all[start:end]
        Iq = Iq_all[start:end]
        
        data_y = to_flat(probe.R)
        data_dy = to_flat(probe.dR)
        base_color = COLORS[i % len(COLORS)]
        
        fig.add_trace(go.Scatter(x=Q, y=data_y, error_y=dict(type='data', array=data_dy, visible=True, color=base_color, thickness=1),
            mode='markers', name=f'Data (Probe {i+1})', marker=dict(color=base_color, symbol='circle', size=6, opacity=0.4), legendgroup=f'group{i}'))
        fig.add_trace(go.Scatter(x=Q, y=Total, mode='lines', name=f'Total (Probe {i+1})', line=dict(color=base_color, width=3), legendgroup=f'group{i}'))
        fig.add_trace(go.Scatter(x=Q, y=Rq, mode='lines', name=f'Refl (Probe {i+1})', line=dict(color=base_color, width=2, dash='dash'), legendgroup=f'group{i}', showlegend=True))
        fig.add_trace(go.Scatter(x=Q, y=Iq, mode='lines', name=f'SANS (Probe {i+1})', line=dict(color=base_color, width=2, dash='dot'), legendgroup=f'group{i}', showlegend=True))
        cursor += n_points

    fig.update_layout(title=f'Signal Decomposition: {model.name}', xaxis_title='Q (Å⁻¹)', xaxis_type='linear', template='plotly_white',
        yaxis=dict(title='Intensity (R + I)', type='log', exponentformat='power', showexponent='all'),
        legend=dict(x=0.01, y=0.01, xanchor='left', yanchor='bottom', bgcolor='rgba(255,255,255,0.8)'))

    csv_header = "Q,R,dR,Theory,Rq,Iq\n"
    csv_rows = []
    n_pts_total = min(len(Q_all), len(total_theory))
    
    if hasattr(model.probe, 'probes'):
        all_data_y = np.hstack([to_flat(p.R) for p in model.probe.probes])
        all_data_dy = np.hstack([to_flat(p.dR) for p in model.probe.probes])
    else:
        all_data_y = to_flat(model.probe.R)
        all_data_dy = to_flat(model.probe.dR)

    for i in range(n_pts_total):
        row = f"{Q_all[i]:.6e},{all_data_y[i]:.6e},{all_data_dy[i]:.6e},{total_theory[i]:.6e},{Rq_all[i]:.6e},{Iq_all[i]:.6e}"
        csv_rows.append(row)
    
    return CustomWebviewPlot(fig_type='plotly', plotdata=fig, exportdata=csv_header + "\n".join(csv_rows))