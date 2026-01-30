""" Module for interfacing combined reflectivity and small angle scattering models
"""

from dataclasses import dataclass

import numpy as np
import plotly.graph_objs as go

from bumps.parameter import Parameter
from bumps.webview.server.custom_plot import CustomWebviewPlot
from refl1d.experiment import Experiment
from refl1d.probe.resolution import dTdL2dQ
from refl1d.webview.server.colors import COLORS

from sasmodels.core import load_model
from sasmodels.direct_model import DirectModel
from sasmodels.data import Data1D

@dataclass
class SASReflectivityModel:
    """ Class to hold sasmodels model information
    """
    sas_model_name: str | None = None
    dtheta_l: float | None = None
    parameters: dict[str, float | Parameter] | None = None

@dataclass(init=False)
class SASReflectivityExperiment(Experiment):
    """ Class to interface sasmodels with refl1d Experiment class
    """
    
    sas_model: SASReflectivityModel = None

    def __init__(self,
                 sas_model: SASReflectivityModel=None,
                 sample = None,
                 probe=None,
                 name=None,
                 roughness_limit=0,
                 dz=None,
                 dA=None,
                 step_interfaces=None,
                 smoothness=None,
                 interpolation=0,
                 constraints=None,
                 version = None,
                 auto_tag=False):
        
        super().__init__(sample, probe, name, roughness_limit, dz, dA, step_interfaces, smoothness, interpolation, constraints, version, auto_tag)
        self.sas_model = sas_model

        # convert all non-Parameter sasmodel parameters to bumps.Parameter
        if sas_model is not None:
            for k, p in self.sas_model.parameters.items():
                if not isinstance(p, Parameter):
                    self.sas_model.parameters[k] = Parameter.default(p, name=k)

        # create sasmodels kernel
        if sas_model is not None and sas_model.sas_model_name is not None:
            self._sasmodel = load_model(sas_model.sas_model_name)

        # Register the Decomposition Plot in the Webview
        # We use functools.partial to pass 'self' (this experiment instance) 
        # as the first argument to the plot function.
        self.register_webview_plot(
            plot_title='SAS/Refl Decomposition',
            plot_function=sas_decomposition_plot,
            change_with='parameter'
        )

    def parameters(self):
        return super().parameters() | {'sas': self.sas_model.parameters if self.sas_model is not None else {}}

    def sas(self):
        """ Calculate the small angle scattering from the reflectivity model
        """
        key = ("small_angle_scattering")
        if key not in self._cache:
            # Initialize data object for sasmodels
            # TODO: check whether the resolution functions are FWHM, sigma, etc.
            data = Data1D(x=self.probe.Q)
            data.dxl=dTdL2dQ(self.probe.T, self.sas_model.dtheta_l, self.probe.L, self.probe.dL)
            data.dxw=self.probe.dQ

            # calling float converts all bumps Parameters into their current values
            pars = {k: float(p) for k, p in self.sas_model.parameters.items()}

            # set data in sasmodels object
            sasmodel = DirectModel(data=data, model=self._sasmodel)

            # execute calculation
            Iq = sasmodel(**pars)
            self._cache[key] = Iq
        return self._cache[key]

    def reflectivity(self, resolution=True, interpolation=0):
        Q, Rq = super().reflectivity(resolution, interpolation)
        Iq = self.sas() if self.sas_model is not None else 0.0
        return Q, Rq + Iq
    
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
