""" Module for interfacing combined reflectivity and small angle scattering models
"""

from dataclasses import dataclass

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

@dataclass
class SASReflectivityModel:
    """ Class to hold sasmodels model information
    """
    sas_model_name: str | None = None
    dtheta_l: float | None = None
    parameters: dict[str, float | Parameter] | None = None

class SASReflectivityMixin:
    """
    Mixin class that adds SAS capabilities to ANY Refl1D Experiment.
    It overrides reflectivity(), parameters(), and registers the SAS plot.

    Requires a probe object with Q, T, L, dL attributes, due to the 
    need to calculate resolution.
    """
    
    # Type hinting for the mixin (expects these to exist on the child)
    sas_model: 'SASReflectivityModel'
    _sasmodel: object
    _cache: dict
    probe: object
    name: str

    def _init_sas(self, sas_model):
        """ Helper to initialize SAS components. Call this from the child __init__ """
        self.sas_model = sas_model
        
        # Initialize Parameters
        if sas_model is not None:
            for k, p in self.sas_model.parameters.items():
                if not isinstance(p, Parameter):
                    self.sas_model.parameters[k] = Parameter.default(p, name=k)

            # Initialize Kernel
            if sas_model.sas_model_name is not None:
                self._sasmodel = load_model(sas_model.sas_model_name)
        
        # Register the Decomposition Plot
        # Note: 'self' here will be the full Experiment instance
        self.register_webview_plot(
            plot_title='SAS/Refl Decomposition',
            plot_function=sas_decomposition_plot,
            change_with='parameter'
        )

    def parameters(self):
        # Merge parent parameters (Experiment/Molgroups) with SAS parameters
        return super().parameters() | {'sas': self.sas_model.parameters if self.sas_model is not None else {}}

    def _calc_Iq(self, probe, dtheta_l):
        """ Calculate the small angle scattering I(q) """
        data = Data1D(x=probe.Q)
        
        # calculate Q-transformed slit widths. dQ is assumed to be sigma, while dtheta_l is 2 * FWHM
        data.dxl = dTdL2dQ(np.zeros_like(probe.T), dtheta_l, probe.L, probe.dL)
        data.dxw = 2 * sigma2FWHM(probe.dQ)
        
        pars = {k: float(p) for k, p in self.sas_model.parameters.items()}
        
        # Calculate
        sasmodel = DirectModel(data=data, model=self._sasmodel)
        Iq = sasmodel(**pars)
        return Iq

    def sas(self):
        """ Calculate the small angle scattering I(q) """
        key = ("small_angle_scattering")
        if key not in self._cache:
            probes = [self.probe] if not isinstance(self.probe, ProbeSet) else self.probe.probes
            
            # Broadcast dtheta_l if it is None or a scalar
            dtheta_val = self.sas_model.dtheta_l
            if np.isscalar(dtheta_val) or dtheta_val is None:
                dtheta_list = [dtheta_val] * len(probes)
            else:
                dtheta_list = dtheta_val
            
            # Calculate and Stack
            Iq_parts = [self._calc_Iq(probe, dt) for probe, dt in zip(probes, dtheta_list)]
            self._cache[key] = np.hstack(Iq_parts)

            Iq = np.hstack([self._calc_Iq(probe, dtheta_l) for probe, dtheta_l in zip(probes, dtheta_list)])
            self._cache[key] = Iq
        return self._cache[key]

    def reflectivity(self, resolution=True, interpolation=0):
        # 1. Get base reflectivity (Calculated by Experiment or MolgroupsExperiment)
        Q, Rq = super().reflectivity(resolution, interpolation)

        # 2. Add SAS signal
        if self.sas_model is not None:
            Rq = Rq + self.sas()
        return Q, Rq

# --- 2. CONCRETE CLASSES ---

@dataclass(init=False)
class SASReflectivityExperiment(SASReflectivityMixin, Experiment):
    """
    Standard SAS + Reflectivity Experiment.
    Inherits from Experiment.
    """
    sas_model: 'SASReflectivityModel' = None

    def __init__(self, sas_model=None, sample=None, probe=None, name=None, **kwargs):
        # 1. Initialize Parent (Experiment)
        super().__init__(sample, probe, name, **kwargs)
        
        # 2. Initialize Mixin
        self._init_sas(sas_model)


@dataclass(init=False)
class SASReflectivityMolgroupsExperiment(SASReflectivityMixin, MolgroupsExperiment):
    """
    Molgroups-Enabled SAS + Reflectivity Experiment.
    Inherits from MolgroupsExperiment.
    """
    sas_model: 'SASReflectivityModel' = None

    def __init__(self, sas_model=None, sample=None, probe=None, name=None, **kwargs):
        # 1. Initialize Parent (MolgroupsExperiment)
        # This automatically registers the CVO plots, Table plots, etc.
        super().__init__(sample, probe, name, **kwargs)
        
        # 2. Initialize Mixin
        self._init_sas(sas_model)
    
def sas_decomposition_plot(model: SASReflectivityExperiment, problem=None) -> CustomWebviewPlot:
    """
    Webview plot that shows the decomposition of the signal into 
    Reflectivity (Rq) and SAS (Iq) components.
    
    Supports both single Probe and ProbeSet.
    """

    # 1. Helpers for Flattening
    def to_flat(arr):
        if arr is None: return np.array([])
        return np.ravel(np.array(arr, dtype=float))

    # 2. Get Concatenated Theory Components
    # model.reflectivity() and sas() return 1D arrays matching the full concatenated Q
    Q_all_raw, total_theory_raw = model.reflectivity()
    Q_all = to_flat(Q_all_raw)
    total_theory = to_flat(total_theory_raw)
    
    if model.sas_model is not None:
        Iq_all = to_flat(model.sas())
    else:
        Iq_all = np.zeros_like(Q_all)
        
    Rq_all = total_theory - Iq_all

    # 3. Identify Probes (Single vs ProbeSet)
    if hasattr(model.probe, 'probes'):
        probes = model.probe.probes
    else:
        probes = [model.probe]

    # 4. Construct Plotly Figure
    fig = go.Figure()
    
    # Cursor to track where we are in the concatenated arrays
    cursor = 0
    
    # Loop over probes to slice data and plot traces
    for i, probe in enumerate(probes):
        # Determine slice range for this probe
        n_points = len(probe.Q)
        start = cursor
        end = cursor + n_points
        
        # Slice the arrays
        Q = Q_all[start:end]
        Total = total_theory[start:end]
        Rq = Rq_all[start:end]
        Iq = Iq_all[start:end]
        
        # Get Data for this probe
        data_y = to_flat(probe.R)
        data_dy = to_flat(probe.dR)
        
        # Define Color for this Probe (Cycle through COLORS)
        base_color = COLORS[i % len(COLORS)]
        
        # -- Trace: Data --
        fig.add_trace(go.Scatter(
            x=Q, y=data_y,
            error_y=dict(
                type='data', 
                array=data_dy, 
                visible=True,
                color=base_color,
                thickness=1
            ),
            mode='markers',
            name=f'Data (Probe {i+1})',
            marker=dict(
                color=base_color,
                symbol='circle',
                size=6,
                opacity=0.4
            ),
            legendgroup=f'group{i}'
        ))

        # -- Trace: Total Theory (Solid) --
        fig.add_trace(go.Scatter(
            x=Q, y=Total,
            mode='lines',
            name=f'Total (Probe {i+1})',
            line=dict(color=base_color, width=3),
            legendgroup=f'group{i}'
        ))

        # -- Trace: Reflectivity (Dash) --
        # showlegend=True explicitly added
        fig.add_trace(go.Scatter(
            x=Q, y=Rq,
            mode='lines',
            name=f'Refl (Probe {i+1})',
            line=dict(color=base_color, width=2, dash='dash'),
            legendgroup=f'group{i}',
            showlegend=True 
        ))

        # -- Trace: SANS (Dot) --
        # showlegend=True explicitly added
        fig.add_trace(go.Scatter(
            x=Q, y=Iq,
            mode='lines',
            name=f'SANS (Probe {i+1})',
            line=dict(color=base_color, width=2, dash='dot'),
            legendgroup=f'group{i}',
            showlegend=True 
        ))

        # Advance cursor
        cursor += n_points

    # 5. Styling
    fig.update_layout(
        title=f'Signal Decomposition: {model.name}',
        xaxis_title='Q (Å⁻¹)',
        xaxis_type='linear',
        template='plotly_white',
        yaxis=dict(
            title='Intensity (R + I)',
            type='log',
            exponentformat='power', 
            showexponent='all'
        ),
        legend=dict(x=0.01, y=0.01, xanchor='left', yanchor='bottom', bgcolor='rgba(255,255,255,0.8)')
    )

    # 6. Prepare CSV Export Data (Concatenated)
    csv_header = "Q,R,dR,Theory,Rq,Iq\n"
    csv_rows = []
    
    n_pts_total = min(len(Q_all), len(total_theory))
    
    # Re-flatten probe data for CSV export
    if hasattr(model.probe, 'probes'):
        all_data_y = np.hstack([to_flat(p.R) for p in model.probe.probes])
        all_data_dy = np.hstack([to_flat(p.dR) for p in model.probe.probes])
    else:
        all_data_y = to_flat(model.probe.R)
        all_data_dy = to_flat(model.probe.dR)

    for i in range(n_pts_total):
        row = f"{Q_all[i]:.6e},{all_data_y[i]:.6e},{all_data_dy[i]:.6e},{total_theory[i]:.6e},{Rq_all[i]:.6e},{Iq_all[i]:.6e}"
        csv_rows.append(row)
    
    export_data = csv_header + "\n".join(csv_rows)

    return CustomWebviewPlot(fig_type='plotly', 
                             plotdata=fig, 
                             exportdata=export_data)