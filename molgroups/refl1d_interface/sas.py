""" Module for interfacing combined reflectivity and small angle scattering models
"""

from dataclasses import dataclass, field
import copy
import functools

import numpy as np
import plotly.graph_objs as go

from bumps.parameter import Parameter
from bumps.webview.server.custom_plot import CustomWebviewPlot
from refl1d.experiment import Experiment
from refl1d.probe import ProbeSet
from refl1d.probe.resolution import dTdL2dQ, sigma2FWHM
from refl1d.webview.server.colors import COLORS

from sasmodels.core import load_model, load_model_info, build_model
from sasmodels.direct_model import DirectModel
from sasmodels.data import Data1D
from sasmodels.modelinfo import parse_parameter, ParameterTable

from .experiment import MolgroupsExperiment
from .layers import MolgroupsLayer
from .plots import cvo_plot, cvo_uncertainty_plot

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
    
    def get_profile(self):
        """
        Return the radial SLD profile as (radius_array, sld_array, labels).
        labels is a tuple (xlabel, ylabel).
        Returns (None, None, None) if not available.
        """
        return None, None, None

    def get_plots(self):
        """
        Return a dictionary of plots to register with the webview.
        Structure:
        {
            'parameter': [(title, plot_function), ...],
            'uncertainty': [(title, plot_function), ...]
        }
        """
        return {'parameter': [], 'uncertainty': []}
    
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
    """
    sas_model_name: str
    params: dict[str, float | Parameter] = field(default_factory=dict)
    dtheta_l: float | None = None
    
    # Internal state (excluded from __init__)
    _engines: list[DirectModel] | None = field(default=None, init=False, repr=False)
    _probe: object = field(default=None, init=False, repr=False)

    def __post_init__(self):
        for k, v in self.params.items():
            if not isinstance(v, Parameter):
                self.params[k] = Parameter.default(v, name=k)

    def bind(self, probe):
        self._probe = probe
        self._engines = None 
        self._build_engines()

    def _generate_params(self):
        return {k: v.value for k, v in self.params.items()}

    def calculate(self):
        if self._engines is None:
            self._build_engines()
            
        if not self._engines:
            return np.array([])
            
        pars = self._generate_params()
        parts = [model(**pars) for model in self._engines]
        return np.hstack(parts)

    def get_profile(self):
        """ Retrieve profile from the engine if supported """
        if self._engines is None:
            self._build_engines()
        
        # Guard against empty engine list or missing profile method
        if not self._engines or not hasattr(self._engines[0], 'profile'):
            return None, None, None

        pars = self._generate_params()
        try:
            # sasmodels profile returns x, y, (xlabel, ylabel)
            return self._engines[0].profile(**pars)
        except (AttributeError, TypeError, NotImplementedError):
            return None, None, None

    def get_plots(self):
        """ 
        Return list of Standard SAS plots categorized by update trigger.
        """
        if self._engines is None:
            self._build_engines()
            
        plots = {'parameter': [], 'uncertainty': []}
        
        if self._engines and hasattr(self._engines[0], 'profile'):
             plots['parameter'].append(('SANS Profile', sans_profile_plot))
             
        return plots

    def _build_engines(self):
        if not self.sas_model_name or self._probe is None:
            self._engines = []
            return

        kernel = load_model(self.sas_model_name)
        
        probes = [self._probe] if not isinstance(self._probe, ProbeSet) else self._probe.probes

        if np.isscalar(self.dtheta_l) or self.dtheta_l is None:
            dtheta_list = [self.dtheta_l] * len(probes)
        else:
            dtheta_list = self.dtheta_l
        
        new_engines = []
        for probe, dt in zip(probes, dtheta_list):
            data = Data1D(x=probe.Q)
            data.dxl = dTdL2dQ(np.zeros_like(probe.T), dt, probe.L, probe.dL)
            data.dxw = 2 * sigma2FWHM(probe.dQ) if hasattr(probe, 'dQ') else np.zeros_like(probe.Q)
            new_engines.append(DirectModel(data=data, model=kernel))
            
        self._engines = new_engines

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_engines'] = None 
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    @property
    def parameters(self):
        return self.params


# --- 3. MOLGROUPS IMPLEMENTATION ---

@dataclass
class MolgroupsSphereSASModel(SASModel):
    """
    Maps a MolgroupsLayer profile to the sasmodels 'core_multi_shell' kernel.
    Assumes spherical symmetry for volume scaling.
    """
    molgroups_layer: MolgroupsLayer
    dz: float = 5.0
    
    # Common parameters
    r_core: Parameter | float = 0.0
    scale: Parameter | float = 1.0
    background: Parameter | float = 0.0

    # Fixed configuration
    sas_model_name: str = 'core_multi_shell'
    geometry_exponent: int = 2 # Sphere (p=2)

    # Internal state
    _engines: list[DirectModel] | None = field(default=None, init=False, repr=False)
    _probe: object = field(default=None, init=False, repr=False)
    _last_n_shells: int = field(default=0, init=False, repr=False)
    _kernel: object = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        for name in ['r_core', 'scale', 'background']:
            val = getattr(self, name)
            if not isinstance(val, Parameter):
                setattr(self, name, Parameter.default(val, name=name))

    @property
    def parameters(self):
        mg_params = self.molgroups_layer.parameters()
        own_params = {
            'r_core': self.r_core,
            'scale': self.scale, 
            'background': self.background
        }
        return mg_params | own_params

    def bind(self, probe):
        self._probe = probe
        self._engines = None
        self._kernel = None
        self._last_n_shells = 0

    def get_profile(self):
        """ 
        Reconstruct the radial SLD profile using the engine's profile method. 
        """
        # 1. Discretize Layer
        thickness = self.molgroups_layer.thickness.value
        if thickness <= 0: return None, None, None

        z = np.arange(0, thickness, self.dz)
        sld_layer = self.molgroups_layer._filled_profile(z)
        n_shells = len(z)
        if n_shells == 0: return None, None, None

        # 2. Ensure Kernel is built
        self._ensure_kernel(n_shells)
        
        # 3. Generate parameters
        pars = self._generate_params(z, sld_layer, n_shells)
        
        # 4. Retrieve Profile from Engine
        if not self._engines:
             self._build_engines_from_kernel()
             
        if not self._engines or not hasattr(self._engines[0], 'profile'):
            return None, None, None
            
        try:
            return self._engines[0].profile(**pars)
        except (AttributeError, TypeError, NotImplementedError):
            return None, None, None

    def get_plots(self):
        """ Return dictionary of categorized plots """
        plots = {
            'parameter': [
                ('SANS Layer Profile', functools.partial(cvo_plot, self.molgroups_layer))
            ],
            'uncertainty': [
                ('SANS Layer CVO', functools.partial(cvo_uncertainty_plot, self.molgroups_layer))
            ]
        }
        
        # Initialize kernel with safe limit (10) to check for 'profile' capability
        if self._engines is None and self._probe is not None:
             self._ensure_kernel(10)
             
        if self._engines and hasattr(self._engines[0], 'profile'):
            # Insert at the beginning of the parameter list
            plots['parameter'].insert(0, ('SANS Radial Profile', sans_profile_plot))

        return plots

    def _ensure_kernel(self, n_shells):
        """
        Dynamically patches the core_multi_shell definition to allow 'n' 
        to reach the current shell count.
        """
        if self._kernel is not None and self._last_n_shells >= n_shells:
            return

        base_info = load_model_info(self.sas_model_name)
        my_info = copy.deepcopy(base_info)
        
        desired_limit = max(20, n_shells + 10)

        # DEFINE RAW PARAMETERS
        raw_params = [
            ["sld_core", "1e-6/Ang^2", 1.0, [-np.inf, np.inf], "sld", "Core scattering length density"],
            ["radius", "Ang", 200., [0, np.inf], "volume", "Radius of the core"],
            ["sld_solvent", "1e-6/Ang^2", 6.4, [-np.inf, np.inf], "sld", "Solvent scattering length density"],
            ["n", "", float(1), [0, n_shells], "volume", "number of shells"],
            ["sld[n]", "1e-6/Ang^2", 1.7, [-np.inf, np.inf], "sld", "scattering length density of shell k"],
            ["thickness[n]", "Ang", 40., [0, np.inf], "volume", "Thickness of shell k"],
        ]

        # PARSE PARAMETERS
        processed_list = []
        for entry in raw_params:
            p = parse_parameter(*entry)
            p.length_control = None  # Disable sasmodels' internal length checks
            
            if '[n]' in p.name:
                p.length = n_shells
            else:
                p.length = 1
            
            processed_list.append(p)

        # CREATE TABLE
        partable = ParameterTable(processed_list)
        
        # BUILD MODEL
        my_info.parameters = partable
        self._kernel = build_model(my_info)
        self._last_n_shells = n_shells 
        
        # REBUILD ENGINES
        self._build_engines_from_kernel()

    def _build_engines_from_kernel(self):
        if self._probe is None: return
        
        probes = [self._probe] if not isinstance(self._probe, ProbeSet) else self._probe.probes
        dtheta_list = [0.0] * len(probes)

        new_engines = []
        for probe, dt in zip(probes, dtheta_list):
            data = Data1D(x=probe.Q)
            data.dxl = dTdL2dQ(np.zeros_like(probe.T), dt, probe.L, probe.dL)
            data.dxw = 2 * sigma2FWHM(probe.dQ) if hasattr(probe, 'dQ') else np.zeros_like(probe.Q)
            new_engines.append(DirectModel(data=data, model=self._kernel))
        
        self._engines = new_engines

    def calculate(self):
        thickness = self.molgroups_layer.thickness.value
        if thickness <= 0: return np.array([])
        
        z = np.arange(0, thickness, self.dz)
        sld = self.molgroups_layer._filled_profile(z)
        n_shells = len(z)
        
        if n_shells == 0: return np.array([])

        self._ensure_kernel(n_shells)
        
        pars = self._generate_params(z, sld, n_shells)
        
        if not self._engines: return np.array([])
        parts = [model(**pars) for model in self._engines]
        return np.hstack(parts)

    def _generate_params(self, z, sld, n_shells):
        pars = {
            'scale': self.scale.value,
            'background': self.background.value,
            'n': float(n_shells),
        }
        
        r_core_val = self.r_core.value
        overlap_obj = self.molgroups_layer.base_group.overlap
        overlap_val = overlap_obj.value if isinstance(overlap_obj, Parameter) else float(overlap_obj)

        if r_core_val > overlap_val:
            pars['radius'] = r_core_val - overlap_val
        else:
            pars['radius'] = 0.0

        pars['sld_core'] = sld[0]
        pars['sld_solvent'] = self.molgroups_layer.contrast.rho.value

        p = self.geometry_exponent 

        r_start = pars['radius']
        effective_r_core = max(r_start, overlap_val)
        
        for i in range(n_shells):
            r_current = r_start + z[i]
            
            if effective_r_core > 1e-9 and r_current > 1e-9:
                thick_i = self.dz * (effective_r_core / r_current)**p
            else:
                thick_i = self.dz
            
            pars[f'thickness{i+1}'] = thick_i
            pars[f'sld{i+1}'] = sld[i]

        return pars

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_engines'] = None 
        state['_kernel'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


# --- 4. THE MIXIN ---

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
            self.sas_model.bind(self.probe)
        
        # Register main SAS/Refl plot
        self.register_webview_plot(
            plot_title='SAS/Refl Decomposition',
            plot_function=sas_decomposition_plot,
            change_with='parameter'
        )
        
        # Register model-specific plots via Polymorphism
        if self.sas_model is not None:
            plot_groups = self.sas_model.get_plots()
            
            # Register parameter-driven plots
            for title, func in plot_groups.get('parameter', []):
                self.register_webview_plot(
                    plot_title=title, 
                    plot_function=func, 
                    change_with='parameter'
                )
                
            # Register uncertainty-driven plots
            for title, func in plot_groups.get('uncertainty', []):
                self.register_webview_plot(
                    plot_title=title, 
                    plot_function=func, 
                    change_with='uncertainty'
                )

    def parameters(self):
        base = super().parameters()
        if self.sas_model:
            return base | {'sas': self.sas_model.parameters}
        return base

    def sas(self):
        key = ("small_angle_scattering")
        if key not in self._cache:
             if self.sas_model:
                 self._cache[key] = self.sas_model.calculate()
             else:
                 if isinstance(self.probe, ProbeSet):
                     n = sum(len(p.Q) for p in self.probe.probes)
                 else:
                     n = len(self.probe.Q)
                 self._cache[key] = np.zeros(n)
        return self._cache[key]

    def reflectivity(self, resolution=True, interpolation=0):
        Q, Rq = super().reflectivity(resolution, interpolation)
        if self.sas_model is not None:
            Rq = Rq + self.sas()
        return Q, Rq


# --- 5. CONCRETE EXPERIMENT CLASSES ---

@dataclass(init=False)
class SASReflectivityExperiment(SASReflectivityMixin, Experiment):
    sas_model: SASModel | None = None
    def __init__(self, sas_model: SASModel | None = None, sample=None, probe=None, name=None, **kwargs):
        super().__init__(sample, probe, name, **kwargs)
        self._init_sas(sas_model)

@dataclass(init=False)
class SASReflectivityMolgroupsExperiment(SASReflectivityMixin, MolgroupsExperiment):
    sas_model: SASModel | None = None
    def __init__(self, sas_model: SASModel | None = None, sample=None, probe=None, name=None, **kwargs):
        super().__init__(sample, probe, name, **kwargs)
        self._init_sas(sas_model)


# --- 6. PLOTTING FUNCTIONS ---

def sas_decomposition_plot(model: SASReflectivityExperiment, problem=None) -> CustomWebviewPlot:
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

def sans_profile_plot(experiment: SASReflectivityExperiment, problem=None) -> CustomWebviewPlot:
    """
    Unified plot for SANS SLD Profiles (Radius vs SLD).
    Works for both StandardSASModel (via engine.profile) and MolgroupsSphereSASModel (via calculation).
    """
    model = experiment.sas_model
    if model is None:
        return CustomWebviewPlot(fig_type='plotly', plotdata=go.Figure(), exportdata="")

    # Retrieve profile via the polymorphic method
    # Expects (x, y, labels) or (None, None, None)
    r, sld, labels = model.get_profile()

    if r is None or sld is None:
        return CustomWebviewPlot(fig_type='plotly', plotdata=go.Figure(layout=dict(title="Profile not available")), exportdata="")

    xlabel, ylabel = 'Radius (Å)', 'SLD (10⁻⁶ Å⁻²)'

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r, y=sld, mode='lines', name='SLD Profile', line=dict(color=COLORS[0], width=3)))

    title_text = getattr(model, 'sas_model_name', 'SAS Model')
    fig.update_layout(
        title=f'SANS Radial Profile: {title_text}',
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template='plotly_white'
    )

    csv_header = f"{xlabel},{ylabel}\n"
    csv_rows = [f"{ri:.6e},{sldi:.6e}" for ri, sldi in zip(r, sld)]
    
    return CustomWebviewPlot(fig_type='plotly', plotdata=fig, exportdata=csv_header + "\n".join(csv_rows))