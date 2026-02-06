"""
Module for interfacing combined reflectivity and small angle scattering (SAS) models.

This module provides the architecture for simultaneous fitting of Reflectivity and SANS data
within the Refl1D framework. It defines a base `SASModel` class and concrete implementations
for standard sasmodels usage (`StandardSASModel`) and complex molecular layers (`MolgroupsSphereSASModel`).

Key Classes:
    - SASModel: Abstract base class defining the interface for SAS engines.
    - StandardSASModel: Wrapper for standard sasmodels library models (e.g., cylinder, sphere).
    - MolgroupsSphereSASModel: specialized model mapping a MolgroupsLayer profile to a
      'core_multi_shell' sasmodel, handling dynamic shell count and parameter mapping.
    - SASReflectivityMixin: Mixin for Experiment classes to add SAS calculation capabilities.

Dependencies:
    - sasmodels: Used for the underlying scattering kernel calculations.
    - refl1d: Provides the experiment and probe framework.
    - bumps: Handles parameter management.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import copy
import functools
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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

# Type alias for the profile return signature
ProfileType = Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[str, str]]]
# Type alias for plot registry
PlotList = List[Tuple[str, Callable[..., CustomWebviewPlot]]]
PlotDict = Dict[str, PlotList]

# --- 1. THE INTERFACE (Base Class) ---

class SASModel:
    """
    Base class for SAS calculation engines.
    
    Subclasses must implement `bind()` to link to an experiment probe and `calculate()`
    to return the theoretical intensity I(Q).
    """
    def bind(self, probe: Any) -> None:
        """
        Associate the model with a Probe (or ProbeSet).

        This method is used to perform expensive setup steps, such as compiling
        SAS kernels or initializing data structures that depend on the Q-vector.

        Args:
            probe (Any): The experimental probe containing Q values (typically Probe or ProbeSet).
        """
        raise NotImplementedError("Subclasses must implement bind()")

    def calculate(self) -> np.ndarray:
        """
        Calculate the scattering intensity I(Q).

        Returns:
            np.ndarray: The calculated I(Q) matching the Q-points of the bound probe.
                        If multiple probes are bound, the arrays should be concatenated.
        """
        raise NotImplementedError("Subclasses must implement calculate()")
    
    def get_profile(self) -> ProfileType:
        """
        Retrieve the radial SLD profile of the model, if applicable.

        Returns:
            tuple: A tuple containing (radius_array, sld_array, labels).
                   - radius_array (np.ndarray): The radial distance axis.
                   - sld_array (np.ndarray): The SLD values at each radius.
                   - labels (tuple): A tuple of strings (xlabel, ylabel).
                   Returns (None, None, None) if the profile cannot be generated.
        """
        return None, None, None

    def get_plots(self) -> PlotDict:
        """
        Return a dictionary of plots to register with the webview.

        Returns:
            dict: A dictionary with keys 'parameter' and 'uncertainty'.
                  Each value is a list of tuples: [(title, plot_function), ...].
                  - 'parameter': Plots that update when model parameters change.
                  - 'uncertainty': Plots used for uncertainty analysis (e.g., CVO).
        """
        return {'parameter': [], 'uncertainty': []}
    
    @property
    def parameters(self) -> Dict[str, Parameter]:
        """
        Return a dictionary of Bumps Parameter objects managed by this model.

        Returns:
            dict: Dictionary mapping parameter names to Parameter objects.
        """
        return {}


# --- 2. CONCRETE IMPLEMENTATION (Standard Sasmodels) ---

@dataclass
class StandardSASModel(SASModel):
    """
    A SAS model that uses the standard sasmodels library (DirectModel).

    This class wraps a standard sasmodels kernel (e.g., 'cylinder', 'sphere')
    and manages the mapping of Bumps parameters to the kernel inputs.
    """
    sas_model_name: str
    params: Dict[str, Union[float, Parameter]] = field(default_factory=dict)
    dtheta_l: Optional[Union[float, List[float]]] = None
    
    # Internal state (excluded from __init__)
    _engines: Optional[List[DirectModel]] = field(default=None, init=False, repr=False)
    _probe: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # Ensure all inputs in params are converted to Bumps Parameters.
        for k, v in self.params.items():
            if not isinstance(v, Parameter):
                self.params[k] = Parameter.default(v, name=k)

    def bind(self, probe: Any) -> None:
        """
        Bind the model to a probe and build the calculation engines.

        Args:
            probe (Any): The experimental data probe.
        """
        self._probe = probe
        self._engines = None 
        self._build_engines()

    def _generate_params(self) -> Dict[str, float]:
        """
        Extract current values from Bumps parameters for the SAS kernel.

        Returns:
            dict: Dictionary of parameter values (floats) expected by sasmodels.
        """
        return {k: v.value for k, v in self.params.items()}  # type: ignore

    def calculate(self) -> np.ndarray:
        """
        Calculate I(Q) using the sasmodels DirectModel engine.

        Returns:
            np.ndarray: Calculated intensity.
        """
        if self._engines is None:
            self._build_engines()
            
        if not self._engines:
            return np.array([])
            
        pars = self._generate_params()
        # Calculate for each probe/engine and concatenate results
        parts = [model(**pars) for model in self._engines]
        return np.hstack(parts)

    def get_profile(self) -> ProfileType:
        """
        Retrieve the SLD profile from the underlying sasmodels engine.

        Returns:
            tuple: (r, sld, (xlabel, ylabel)) or (None, None, None).
        """
        if self._engines is None:
            self._build_engines()
        
        # Guard against empty engine list or missing profile method on the kernel
        if not self._engines or not hasattr(self._engines[0], 'profile'):
            return None, None, None

        pars = self._generate_params()
        try:
            # sasmodels profile returns x, y, (xlabel, ylabel)
            return self._engines[0].profile(**pars)  # type: ignore
        except (AttributeError, TypeError, NotImplementedError):
            return None, None, None

    def get_plots(self) -> PlotDict:
        """ 
        Return list of Standard SAS plots categorized by update trigger.
        Checks existence of profile method WITHOUT performing a calculation.
        """
        if self._engines is None:
            self._build_engines()
            
        plots: PlotDict = {'parameter': [], 'uncertainty': []}
        
        # Register profile plot only if supported by the kernel
        if self._engines and self._engines[0].model.info.profile is not None:
             plots['parameter'].append(('SANS Profile', sans_profile_plot))
             
        return plots

    def _build_engines(self) -> None:
        """
        Compile the sasmodels kernel and create DirectModel instances for each probe.
        """
        if not self.sas_model_name or self._probe is None:
            self._engines = []
            return

        kernel = load_model(self.sas_model_name)
        
        probes = [self._probe] if not isinstance(self._probe, ProbeSet) else self._probe.probes

        # Handle angular divergence (dtheta) logic
        if np.isscalar(self.dtheta_l) or self.dtheta_l is None:
            dtheta_list = [self.dtheta_l] * len(probes)
        else:
            dtheta_list = self.dtheta_l  # type: ignore
        
        new_engines = []
        for probe, dt in zip(probes, dtheta_list):
            # Create Data1D objects required by sasmodels
            data = Data1D(x=probe.Q)
            
            # Map resolution parameters
            data.dxl = dTdL2dQ(np.zeros_like(probe.T), dt, probe.L, probe.dL)
            data.dxw = 2 * sigma2FWHM(probe.dQ) if hasattr(probe, 'dQ') else np.zeros_like(probe.Q)
            
            new_engines.append(DirectModel(data=data, model=kernel))
            
        self._engines = new_engines

    def __getstate__(self) -> Dict[str, Any]:
        # Exclude unpickleable C-objects
        state = self.__dict__.copy()
        state['_engines'] = None 
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)

    @property
    def parameters(self) -> Dict[str, Parameter]:
        return self.params  # type: ignore


# --- 3. MOLGROUPS IMPLEMENTATION ---

@dataclass
class MolgroupsSphereSASModel(SASModel):
    """
    Maps a MolgroupsLayer profile to the sasmodels 'core_multi_shell' kernel.
    
    This model assumes spherical symmetry to convert the linear volume profile 
    of a MolgroupsLayer into a core-multi-shell spherical model. It handles 
    dynamic resizing of the kernel based on the layer discretization.

    Attributes:
        molgroups_layer (MolgroupsLayer): The layer source for SLD profile.
        dz (float): Step size for discretizing the layer (Angstroms).
        r_core (Parameter): Radius of the inner core.
        scale (Parameter): Overall intensity scaling factor.
        background (Parameter): Background intensity.
        dtheta_l (float or list, optional): Angular divergence for resolution smearing.
    """
    molgroups_layer: MolgroupsLayer
    dz: float = 5.0
    
    # Common parameters
    r_core: Union[Parameter, float] = 0.0
    scale: Union[Parameter, float] = 1.0
    background: Union[Parameter, float] = 0.0
    dtheta_l: Optional[Union[float, List[float]]] = None

    # Fixed configuration
    sas_model_name: str = 'core_multi_shell'
    geometry_exponent: int = 2 # Sphere (p=2)

    # Internal state
    _engines: Optional[List[DirectModel]] = field(default=None, init=False, repr=False)
    _probe: Any = field(default=None, init=False, repr=False)
    _last_n_shells: int = field(default=0, init=False, repr=False)
    _kernel: Any = field(default=None, init=False, repr=False)
    
    def __post_init__(self) -> None:
        for name in ['r_core', 'scale', 'background']:
            val = getattr(self, name)
            if not isinstance(val, Parameter):
                setattr(self, name, Parameter.default(val, name=name))

    @property
    def parameters(self) -> Dict[str, Parameter]:
        """ Merge molgroups parameters with specific SAS parameters. """
        mg_params = self.molgroups_layer.parameters()
        own_params = {
            'r_core': self.r_core,
            'scale': self.scale, 
            'background': self.background
        }
        return mg_params | own_params  # type: ignore

    def bind(self, probe: Any) -> None:
        self._probe = probe
        self._engines = None
        self._kernel = None
        self._last_n_shells = 0

    def get_profile(self) -> ProfileType:
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

        # 2. Ensure Kernel is built for correct N
        self._ensure_kernel(n_shells)
        
        # 3. Generate parameters
        pars = self._generate_params(z, sld_layer, n_shells)
        
        # 4. Retrieve Profile from Engine
        if not self._engines:
             self._build_engines_from_kernel()
             
        if not self._engines or not hasattr(self._engines[0], 'profile'):
            return None, None, None
            
        try:
            return self._engines[0].profile(**pars)  # type: ignore
        except (AttributeError, TypeError, NotImplementedError):
            return None, None, None

    def get_plots(self) -> PlotDict:
        """ Return dictionary of categorized plots """
        plots: PlotDict = {
            'parameter': [
                (f'{self.molgroups_layer.name}', functools.partial(cvo_plot, self.molgroups_layer))
            ],
            'uncertainty': [
                (f'{self.molgroups_layer.name} CVO plot', functools.partial(cvo_uncertainty_plot, self.molgroups_layer))
            ]
        }
        
        # Initialize kernel with safe limit (10) to check for 'profile' capability
        if self._engines is None and self._probe is not None:
             self._ensure_kernel(10)
        
        if self._engines and self._engines[0].model.info.profile is not None:
            # Insert at the beginning of the parameter list
            plots['parameter'].insert(0, ('SANS Radial Profile', sans_profile_plot))

        return plots

    def _ensure_kernel(self, n_shells: int) -> None:
        """
        Dynamically patches the core_multi_shell definition to allow 'n' 
        to reach the current shell count.

        This uses 'parse_parameter' to reconstruct the parameter table with 
        a new limit for 'n' and expanded vector definitions.
        
        Args:
            n_shells (int): The required number of shells.
        """
        if self._kernel is not None and self._last_n_shells >= n_shells:
            return

        base_info = load_model_info(self.sas_model_name)
        my_info = copy.deepcopy(base_info)
        
        # DEFINE RAW PARAMETERS
        # Note: We must explicitly define the vectors sld[n] and thickness[n]
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
            
            # Explicitly set the length of vector parameters
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

    def _build_engines_from_kernel(self) -> None:
        """ Create DirectModel instances linking data to the compiled kernel. """
        if self._probe is None: return
        
        probes = [self._probe] if not isinstance(self._probe, ProbeSet) else self._probe.probes
        
        # Handle angular divergence (dtheta) logic
        if np.isscalar(self.dtheta_l) or self.dtheta_l is None:
            dtheta_list = [self.dtheta_l] * len(probes)
        else:
            dtheta_list = self.dtheta_l # type: ignore

        new_engines = []
        for probe, dt in zip(probes, dtheta_list):
            data = Data1D(x=probe.Q)
            data.dxl = dTdL2dQ(np.zeros_like(probe.T), dt, probe.L, probe.dL)
            data.dxw = 2 * sigma2FWHM(probe.dQ) if hasattr(probe, 'dQ') else np.zeros_like(probe.Q)
            new_engines.append(DirectModel(data=data, model=self._kernel))
        
        self._engines = new_engines

    def calculate(self) -> np.ndarray:
        """
        Discretize the layer, generate parameters, and calculate I(Q).
        
        Returns:
            np.ndarray: Calculated intensity.
        """
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

    def _generate_params(self, z: np.ndarray, sld: np.ndarray, n_shells: int) -> Dict[str, float]:
        """
        Map the linear SLD profile to spherical shell parameters.

        This iterates through shells and generates scalar keys (thickness1, sld1, ...)
        expected by the dynamically built kernel.
        """
        pars = {
            'scale': self.scale.value,  # type: ignore
            'background': self.background.value,  # type: ignore
            'n': float(n_shells),
        }
        
        # Determine effective core radius (handling overlap)
        r_core_val = self.r_core.value  # type: ignore
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
        
        # Iterate to generate SCALAR parameters (thickness{i}, sld{i})
        for i in range(n_shells):
            r_current = r_start + z[i]
            
            # Map linear step to spherical thickness
            if effective_r_core > 1e-9 and r_current > 1e-9:
                thick_i = self.dz * (effective_r_core / r_current)**p
            else:
                thick_i = self.dz
            
            pars[f'thickness{i+1}'] = thick_i
            pars[f'sld{i+1}'] = sld[i]

        return pars

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state['_engines'] = None 
        state['_kernel'] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)


# --- 4. THE MIXIN ---

class SASReflectivityMixin:
    """
    Mixin class that adds SAS capabilities to ANY Refl1D Experiment.
    
    This mixin intercepts the `reflectivity` calculation to add the SAS contribution
    and registers relevant SAS plots to the webview.
    """
    
    sas_model: Optional[SASModel]
    _cache: Dict[str, Any]
    probe: Any
    name: str

    def _init_sas(self, sas_model: Optional[SASModel]) -> None:
        """
        Initialize the SAS model and register plots.
        """
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

    def parameters(self) -> Dict[str, Any]:
        base = super().parameters()  # type: ignore
        if self.sas_model:
            return base | {'sas': self.sas_model.parameters}
        return base

    def sas(self) -> np.ndarray:
        """ 
        Calculate the small angle scattering I(q).
        Uses caching to avoid re-calculation within the same fit step.
        """
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

    def reflectivity(self, resolution: bool = True, interpolation: int = 0) -> Tuple[Any, np.ndarray]:
        """
        Override standard reflectivity to add SAS contribution.
        Returns total intensity R(Q) + I(Q).
        """
        Q, Rq = super().reflectivity(resolution, interpolation)  # type: ignore
        if self.sas_model is not None:
            Rq = Rq + self.sas()
        return Q, Rq


# --- 5. CONCRETE EXPERIMENT CLASSES ---

@dataclass(init=False)
class SASReflectivityExperiment(SASReflectivityMixin, Experiment):
    """
    Standard SAS + Reflectivity Experiment.
    Combines a standard Experiment with a SASModel.
    """
    sas_model: Optional[SASModel] = None
    def __init__(self, sas_model: Optional[SASModel] = None, sample: Any = None, probe: Any = None, name: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(sample, probe, name, **kwargs)
        self._init_sas(sas_model)

@dataclass(init=False)
class SASReflectivityMolgroupsExperiment(SASReflectivityMixin, MolgroupsExperiment):
    """
    Molgroups-Enabled SAS + Reflectivity Experiment.
    Combines a MolgroupsExperiment with a SASModel.
    """
    sas_model: Optional[SASModel] = None
    def __init__(self, sas_model: Optional[SASModel] = None, sample: Any = None, probe: Any = None, name: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(sample, probe, name, **kwargs)
        self._init_sas(sas_model)

        if isinstance(self.sas_model, MolgroupsSphereSASModel):
            self._molgroups_layers.update({self.sas_model.molgroups_layer.name: self.sas_model.molgroups_layer})


# --- 6. PLOTTING FUNCTIONS ---

def sas_decomposition_plot(model: SASReflectivityExperiment, problem: Any = None) -> CustomWebviewPlot:
    """
    Generate a Plotly graph showing Data, Total Theory, Reflectivity, and SAS components.
    """
    def to_flat(arr: Any) -> np.ndarray:
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

def sans_profile_plot(experiment: SASReflectivityExperiment, problem: Any = None) -> CustomWebviewPlot:
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