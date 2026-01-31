""" Module for interfacing combined reflectivity and small angle scattering models
"""

from dataclasses import dataclass

import numpy as np
import plotly.graph_objs as go

from bumps.parameter import Parameter
from bumps.webview.server.custom_plot import CustomWebviewPlot
from refl1d.experiment import Experiment
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

    def sas(self):
        """ Calculate the small angle scattering I(q) """
        key = ("small_angle_scattering")
        if key not in self._cache:
            data = Data1D(x=self.probe.Q)
            
            # calculate Q-transformed slit widths. dQ is assumed to be sigma, while dtheta_l is 2 * FWHM
            data.dxl = dTdL2dQ(np.zeros_like(self.probe.T), self.sas_model.dtheta_l, self.probe.L, self.probe.dL)
            data.dxw = 2 * sigma2FWHM(self.probe.dQ)
            
            pars = {k: float(p) for k, p in self.sas_model.parameters.items()}
            
            # Calculate
            sasmodel = DirectModel(data=data, model=self._sasmodel)
            Iq = sasmodel(**pars)
            self._cache[key] = Iq
        return self._cache[key]

    def reflectivity(self, resolution=True, interpolation=0):
        # 1. Get base reflectivity (Calculated by Experiment or MolgroupsExperiment)
        Q, Rq = super().reflectivity(resolution, interpolation)
        
        # 2. Add SAS signal
        Iq = self.sas() if self.sas_model is not None else 0.0
        return Q, Rq + Iq


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
    
    Args:
        model: The SASReflectivityExperiment instance (passed by webview)
        problem: The Bumps FitProblem (passed by webview)
    """

    # 2. Calculate Components
    # Total Theory = R(q) + I(q)
    # We call reflectivity() which returns the sum
    Q, total_theory = model.reflectivity()
    
    # SANS Component = I(q)
    # We call sas() directly. Handle case where sas_model might be None
    if model.sas_model is not None:
        Iq = model.sas()
    else:
        Iq = np.zeros_like(Q)
        
    # Reflectivity Component = R(q)
    # Derived by subtraction to ensure consistency
    Rq = total_theory - Iq

    # 3. Get Data for comparison
    data_y = model.probe.R
    data_dy = model.probe.dR

    # 4. Construct Plotly Figure
    fig = go.Figure()

    # -- Trace: Data --
    fig.add_trace(go.Scatter(
        x=Q, y=data_y,
        error_y=dict(
            type='data', 
            array=data_dy, 
            visible=True,
            color='rgba(0, 0, 0, 0.25)'  # <--- Explicit opacity for error bars
        ),
        mode='markers',
        name='Data',
        marker=dict(
            color='rgba(0, 0, 0, 0.25)', # <--- Explicit opacity for markers (0.25 = 25% visible)
            size=6
        )
    ))

    # -- Trace: Total Theory --
    fig.add_trace(go.Scatter(
        x=Q, y=total_theory,
        mode='lines',
        name='Total Model (R+I)',
        line=dict(color=COLORS[0], width=3)
    ))

    # -- Trace: Reflectivity Component --
    fig.add_trace(go.Scatter(
        x=Q, y=Rq,
        mode='lines',
        name='Reflectivity R(q)',
        line=dict(color=COLORS[1], width=2, dash='dash')
    ))

    # -- Trace: SANS Component --
    fig.add_trace(go.Scatter(
        x=Q, y=Iq,
        mode='lines',
        name='SANS I(q)',
        line=dict(color=COLORS[2], width=2, dash='dot')
    ))

    # 5. Styling
    fig.update_layout(
        title=f'Signal Decomposition: {model.name}',
        xaxis_title='Q (Å⁻¹)',
        xaxis_type='linear',
        template='plotly_white',
        yaxis=dict(
                title='Intensity (R + I)',
                type='log',
                exponentformat='power',  # <--- This forces 10^x notation
                showexponent='all'       # Ensures exponents are shown for all ticks
            ),
        legend=dict(x=0.01, y=0.01, xanchor='left', yanchor='bottom', bgcolor='rgba(255,255,255,0.8)')
    )

    # 6. Prepare CSV Export Data
    # Simple CSV format: Q, Data, Error, Total, Rq, Iq
    csv_header = "Q,R,dR,Theory,Rq,Iq\n"
    csv_rows = []
    for i in range(len(Q)):
        row = f"{float(Q[i]):.6e},{float(data_y[i]):.6e},{float(data_dy[i]):.6e},{float(total_theory[i]):.6e},{float(Rq[i]):.6e},{float(Iq[i]):.6e}"
        csv_rows.append(row)
    
    export_data = csv_header + "\n".join(csv_rows)

    return CustomWebviewPlot(fig_type='plotly', 
                             plotdata=fig, 
                             exportdata=export_data)
