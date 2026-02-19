"""
Module for interfacing combined reflectivity and small angle scattering (SAS) models.

This module provides the architecture for simultaneous fitting of Reflectivity and SANS data
within the Refl1D framework, as well as standalone SANS fitting using Bumps.

It defines a base `SASModel` class and concrete implementations for standard sasmodels usage 
(`StandardSASModel`) and complex molecular layers (`MolgroupsSphereSASModel`).

Key Classes:
    - SASModel: Abstract base class defining the interface for SAS engines. Expects standard 
      sasmodels Data1D objects.
    - StandardSASModel: Wrapper for standard sasmodels library models (e.g., cylinder, sphere).
    - MolgroupsSphereSASModel: Specialized model mapping a MolgroupsLayer profile to a
      'core_multi_shell' sasmodel, handling dynamic shell count and parameter mapping.
    - MolgroupsSASExperiment: Bumps Experiment for standalone SANS fitting.
    - SASReflectivityMixin: Mixin for Refl1D Experiment classes. It handles the conversion 
      of Refl1D Probes to SAS Data1D objects (including resolution smearing) and binds them 
      to the SASModel.

Dependencies:
    - sasmodels: Used for the underlying scattering kernel calculations.
    - refl1d: Provides the experiment and probe framework (required only for Refl1D experiments).
    - bumps: Handles parameter management.
"""

from dataclasses import dataclass, field
import copy
import functools
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import plotly.graph_objs as go

# Bumps
from bumps.parameter import Parameter
from bumps.webview.server.custom_plot import CustomWebviewPlot

# Sasmodels
from sasmodels.core import load_model, load_model_info, build_model
from sasmodels.direct_model import DirectModel
from sasmodels.data import Data1D
from sasmodels.modelinfo import parse_parameter, ParameterTable

# Refl1D
from refl1d.experiment import Experiment as Refl1DExperiment
from refl1d.probe import Probe, ProbeSet
from refl1d.probe.resolution import dTdL2dQ, sigma2FWHM
from refl1d.webview.server.colors import COLORS

# Molgroups
from .experiment import MolgroupsExperiment
from .layers import MolgroupsLayer
from .plots import cvo_plot, cvo_uncertainty_plot

# Type alias for the profile return signature
ProfileType = Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[str, str]]]
# Type alias for plot registry
PlotList = List[Tuple[str, Callable[..., CustomWebviewPlot]]]
PlotDict = Dict[str, PlotList]


# =============================================================================
# 1. PURE SAS MODELS
# =============================================================================

class SASModel:
    """
    Base class for SAS calculation engines.
    
    This class is responsible for the physics of the scattering calculation.
    It accepts a list of `sasmodels.data.Data1D` objects, which must already
    contain the Q values and resolution information (dx or dxl).
    
    Subclasses must implement `_build_engines()` to compile the kernel and `calculate()`
    to return the theoretical intensity I(Q).
    """
    _data_list: List[Data1D] = []
    _engines: Optional[List[DirectModel]] = None

    def bind(self, data: Union[None, Data1D, List[Data1D]]) -> None:
        """
        Associate the model with data.

        Args:
            data: A single `sasmodels.data.Data1D` object or a list of them.
                  The data objects must contain Q, I, dI, and resolution (dx or dxl).
        
        Raises:
            TypeError: If the input data is not of type Data1D.
        """
        if isinstance(data, (list, tuple)):
            self._data_list = data
        else:
            self._data_list = [data]
        
        # Strict validation: The model only understands Data1D
        if not all(isinstance(d, Data1D) for d in self._data_list):
            raise TypeError("SASModel.bind strictly expects sasmodels.data.Data1D objects. "
                            "Refl1D Probes must be processed by the Experiment/Mixin first.")

        self._build_engines()

    def _build_engines(self) -> None:
        """
        Construct the sasmodels DirectModel engines based on the bound data.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _build_engines")

    def calculate(self) -> np.ndarray:
        """
        Calculate the scattering intensity I(Q).

        Returns:
            np.ndarray: The calculated I(Q) matching the Q-points of the bound data.
                        If multiple data objects are bound, the arrays are concatenated.
        """
        raise NotImplementedError("Subclasses must implement calculate")
    
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


@dataclass
class StandardSASModel(SASModel):
    """
    A SAS model that uses the standard sasmodels library (DirectModel).

    This class wraps a standard sasmodels kernel (e.g., 'cylinder', 'sphere')
    and manages the mapping of Bumps parameters to the kernel inputs.
    The resolution smearing logic is handled automatically by the DirectModel
    based on the attributes (`dx` or `dxl`) of the bound Data1D objects.
    """
    sas_model_name: str
    params: Dict[str, Union[float, Parameter]] = field(default_factory=dict)
    
    _engines: Optional[List[DirectModel]] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # Ensure all inputs in params are converted to Bumps Parameters.
        for k, v in self.params.items():
            if not isinstance(v, Parameter):
                self.params[k] = Parameter.default(v, name=k)

    def _generate_params(self) -> Dict[str, float]:
        """
        Extract current values from Bumps parameters for the SAS kernel.

        Returns:
            dict: Dictionary of parameter values (floats) expected by sasmodels.
        """
        return {k: v.value for k, v in self.params.items()} # type: ignore

    def _build_engines(self) -> None:
        """
        Compile the sasmodels kernel and create DirectModel instances for each data object.
        """
        if not self.sas_model_name or not self._data_list:
            self._engines = []
            return

        kernel = load_model(self.sas_model_name)
        # DirectModel uses data.dx (pinhole) or data.dxl (slit) automatically from the Data1D object
        self._engines = [DirectModel(data=d, model=kernel) for d in self._data_list]

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
        parts = [model(**pars) for model in self._engines]
        return np.hstack(parts)

    def get_profile(self) -> ProfileType:
        """
        Retrieve the SLD profile from the underlying sasmodels engine.

        Returns:
            tuple: (r, sld, (xlabel, ylabel)) or (None, None, None).
        """
        if self._engines is None: self._build_engines()
        if not self._engines or not hasattr(self._engines[0], 'profile'):
            return None, None, None
        
        try:
            return self._engines[0].profile(**self._generate_params()) # type: ignore
        except (AttributeError, TypeError, NotImplementedError):
            return None, None, None
    
    def get_plots(self) -> PlotDict:
        """ 
        Return list of Standard SAS plots categorized by update trigger.
        Checks existence of profile method WITHOUT performing a calculation.
        """
        if self._engines is None: self._build_engines()
        plots: PlotDict = {'parameter': [], 'uncertainty': []}
        
        if self._engines and self._engines[0].model.info.profile is not None:
             plots['parameter'].append(('SANS Profile', sans_profile_plot))
        return plots

    @property
    def parameters(self) -> Dict[str, Parameter]:
        return self.params # type: ignore

    def __getstate__(self) -> Dict[str, Any]:
        """Custom pickling: Drop the C-pointer objects (_engines)."""
        state = self.__dict__.copy()
        state['_engines'] = None 
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)


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
    """
    molgroups_layer: MolgroupsLayer
    dz: float = 5.0
    r_core: Union[Parameter, float] = 0.0
    scale: Union[Parameter, float] = 1.0
    background: Union[Parameter, float] = 0.0
    
    # Fixed configuration
    sas_model_name: str = 'core_multi_shell'
    geometry_exponent: int = 0 # do not scale layer thicknesses by radius

    # Internal state
    _engines: Optional[List[DirectModel]] = field(default=None, init=False, repr=False)
    _last_n_shells: int = field(default=0, init=False, repr=False)
    _kernel: Any = field(default=None, init=False, repr=False)
    
    def __post_init__(self) -> None:
        for name in ['r_core', 'scale', 'background']:
            val = getattr(self, name)
            if not isinstance(val, Parameter):
                setattr(self, name, Parameter.default(val, name=name))

    @property
    def parameters(self) -> Dict[str, Parameter]:
        """Merge molgroups parameters with specific SAS parameters."""
        mg_params = self.molgroups_layer.parameters()
        own_params = {
            'r_core': self.r_core,
            'scale': self.scale, 
            'background': self.background
        }
        return mg_params | own_params # type: ignore

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
        
        # Build engines if missing or if the kernel object changed (e.g. N resize)
        if not self._engines or (self._engines and self._engines[0].model != self._kernel):
             self._engines = [DirectModel(data=d, model=self._kernel) for d in self._data_list]

        parts = [model(**pars) for model in self._engines]
        return np.hstack(parts)

    def _generate_params(self, z: np.ndarray, sld: np.ndarray, n_shells: int) -> Dict[str, float]:
        """
        Map the linear SLD profile to spherical shell parameters.

        This iterates through shells and generates scalar keys (thickness1, sld1, ...)
        expected by the dynamically built kernel.
        """
        pars = {
            'scale': self.scale.value, # type: ignore
            'background': self.background.value, # type: ignore
            'n': float(n_shells),
        }
        
        r_core_val = self.r_core.value # type: ignore
        overlap_obj = self.molgroups_layer.base_group.overlap
        overlap_val = overlap_obj.value if isinstance(overlap_obj, Parameter) else float(overlap_obj)

        # Handle core radius vs overlap
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
        
        # Explicitly define vector parameters for the core_multi_shell model
        raw_params = [
            ["sld_core", "1e-6/Ang^2", 1.0, [-np.inf, np.inf], "sld", "Core scattering length density"],
            ["radius", "Ang", 200., [0, np.inf], "volume", "Radius of the core"],
            ["sld_solvent", "1e-6/Ang^2", 6.4, [-np.inf, np.inf], "sld", "Solvent scattering length density"],
            ["n", "", float(1), [0, n_shells], "volume", "number of shells"],
            ["sld[n]", "1e-6/Ang^2", 1.7, [-np.inf, np.inf], "sld", "scattering length density of shell k"],
            ["thickness[n]", "Ang", 40., [0, np.inf], "volume", "Thickness of shell k"],
        ]

        processed_list = []
        for entry in raw_params:
            p = parse_parameter(*entry)
            p.length_control = None
            if '[n]' in p.name:
                p.length = n_shells
            else:
                p.length = 1
            processed_list.append(p)

        partable = ParameterTable(processed_list)
        my_info.parameters = partable
        self._kernel = build_model(my_info)
        self._last_n_shells = n_shells 
        self._engines = None # Force engine rebuild on next calc

    def get_profile(self) -> ProfileType:
        """Reconstruct the radial SLD profile using the engine's profile method."""
        # 1. Discretize Layer
        thickness = self.molgroups_layer.thickness.value
        if thickness <= 0: return None, None, None

        z = np.arange(0, thickness, self.dz)
        sld_layer = self.molgroups_layer._filled_profile(z)
        n_shells = len(z)
        if n_shells == 0: return None, None, None

        # 2. Re-Use Calculation Logic
        self._ensure_kernel(n_shells)
        pars = self._generate_params(z, sld_layer, n_shells)
        
        # 3. Build engines if needed
        if not self._engines:
             # Create temp engines just for profile
             self._engines = [DirectModel(data=d, model=self._kernel) for d in self._data_list]
             
        if not self._engines or not hasattr(self._engines[0], 'profile'):
            return None, None, None
            
        try:
            return self._engines[0].profile(**pars) # type: ignore
        except (AttributeError, TypeError, NotImplementedError):
            return None, None, None

    def get_plots(self) -> PlotDict:
        """Return dictionary of categorized plots (SANS Profile, CVO, etc)."""
        plots: PlotDict = {
            'parameter': [
                ('SANS Layer Profile', functools.partial(cvo_plot, self.molgroups_layer))
            ],
            'uncertainty': [
                ('SANS Layer CVO', functools.partial(cvo_uncertainty_plot, self.molgroups_layer))
            ]
        }
        
        # Check if we can generate a profile plot (needs kernel)
        if self._kernel is None:
             self._ensure_kernel(10) # Dummy init to check profile capability
        
        # If we have engines (or created a kernel that supports profile)
        if self._kernel and self._kernel.info.profile is not None:
            plots['parameter'].insert(0, ('SANS Radial Profile', sans_profile_plot))

        return plots
    
    def __getstate__(self) -> Dict[str, Any]:
        """Custom pickling: Drop the C-pointer objects."""
        state = self.__dict__.copy()
        state['_engines'] = None 
        state['_kernel'] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)


# =============================================================================
# 3. STANDALONE BUMPS EXPERIMENT
# =============================================================================
@dataclass(init=False)
class MolgroupsSASExperiment:
    """
    Standalone Bumps Experiment for Molgroups SAS fitting.
    
    This class allows for SANS fitting using Molgroups components without needing
    the full Refl1D experiment structure. It accepts standard `sasmodels.data.Data1D`
    objects directly.
    
    Args:
        data: Single or list of sasmodels.data.Data1D objects.
        model: A SASModel instance (e.g. MolgroupsSphereSASModel).
        name: Name of the experiment.
    """
    data: Union[Data1D, List[Data1D]]
    model: SASModel
    name: str = "SANS"

    def __init__(self, data: Union[Data1D, List[Data1D]], model: SASModel, name: str = "SANS"):
        self.data = data if isinstance(data, list) else [data]
        self.model = model
        self.name = name
        
        # Direct binding: Model gets pure Data1D
        self.model.bind(self.data)
        
        self._webview_plots = {}
        self._init_plots()

    def update(self) -> None:
        """Update the underlying molgroups layer before calculation."""
        if hasattr(self.model, 'molgroups_layer'):
            self.model.molgroups_layer.update()

    def numpoints(self) -> int:
        """Total number of data points across all datasets."""
        return sum(len(d.x) for d in self.data)

    def nllf(self) -> float:
        """Calculate negative log likelihood (chi-squared)."""
        self.update()
        
        Iq_calc = self.model.calculate()
        
        # Flatten data for comparison
        data_mask = np.hstack([d.mask if hasattr(d, 'mask') and d.mask is not None else np.ones_like(d.x, dtype=bool) for d in self.data])
        Iq_obs = np.hstack([d.y for d in self.data])
        dIq_obs = np.hstack([d.dy for d in self.data])
        
        # Standard Chi2
        residuals = (Iq_obs - Iq_calc) / dIq_obs
        return 0.5 * np.sum(residuals[data_mask]**2)

    def _init_plots(self) -> None:
        """Register plots for Bumps webview."""
        plot_groups = self.model.get_plots()
        plot_groups.setdefault('parameter', [])
        plot_groups['parameter'].insert(0, ('SAS Fit', self.plot_fit))
        for key in plot_groups.keys():
            for title, func in plot_groups.get(key, []):
                self._webview_plots[title] = dict(change_with=key, func=functools.partial(self._wrap_plot, func))

    @property
    def webview_plots(self):
        return self._webview_plots

    def plot(self, view: str = 'log') -> None:
        """Matplotlib interface (standard Bumps behavior)."""
        import matplotlib.pyplot as plt
        self.update()
        Iq_calc_all = self.model.calculate()
        
        cursor = 0
        for i, data in enumerate(self.data):
            n = len(data.x)
            Iq_calc = Iq_calc_all[cursor:cursor+n]
            cursor += n
            
            if view == 'log':
                plt.loglog(data.x, data.y, '.', label=f'{self.name} data {i}')
                plt.loglog(data.x, Iq_calc, '-', label=f'{self.name} theory {i}')
            else:
                plt.plot(data.x, data.y, '.', label=f'{self.name} data {i}')
                plt.plot(data.x, Iq_calc, '-', label=f'{self.name} theory {i}')
        plt.legend()

    def plot_fit(self, model=None, problem=None) -> CustomWebviewPlot:
        """Generate Plotly fit plot for webview."""
        self.update()
        Iq_calc_all = self.model.calculate()
        
        fig = go.Figure()
        cursor = 0
        for i, data in enumerate(self.data):
            n = len(data.x)
            Iq_calc = Iq_calc_all[cursor:cursor+n]
            cursor += n
            
            color = COLORS[i % len(COLORS)]
            fig.add_trace(go.Scatter(x=data.x, y=data.y, error_y=dict(type='data', array=data.dy, visible=True),
                                     mode='markers', name=f'Data {i}', marker=dict(color=color)))
            fig.add_trace(go.Scatter(x=data.x, y=Iq_calc, mode='lines', name=f'Theory {i}', line=dict(color=color)))

        fig.update_layout(title=f'{self.name} Fit', xaxis_title='Q (Å⁻¹)', yaxis_title='I(Q) (cm⁻¹)', 
                          yaxis_type='log', xaxis_type='log', template='plotly_white')
        return CustomWebviewPlot(fig_type='plotly', plotdata=fig)

    def _wrap_plot(self, func, model=None, problem=None):
        return func(model, problem)

    def parameters(self) -> Dict[str, Any]:
        """Bumps parameter discovery."""
        return self.model.parameters


# =============================================================================
# 4. REFL1D EXPERIMENTS & MIXIN
# =============================================================================

class SASReflectivityMixin:
    """
    Mixin class that adds SAS capabilities to ANY Refl1D Experiment.
    
    This mixin is responsible for bridging the gap between Refl1D objects and 
    the SASModel. Specifically, it converts Refl1D `Probe` objects into 
    `sasmodels.data.Data1D` objects, applying the appropriate resolution 
    smearing (slit vs pinhole) based on the `dtheta_l` parameter.
    """
    
    sas_model: Optional[SASModel]
    _cache: Dict[str, Any]
    probe: Any
    name: str
    dtheta_l: Optional[Union[float, List[float]]]

    def _init_sas(self, sas_model: Optional[SASModel], dtheta_l: Optional[Union[float, List[float]]] = None) -> None:
        """
        Initialize the SAS model and register plots.
        
        Args:
            sas_model: The SASModel calculation engine.
            dtheta_l: Optional angular divergence parameter for slit-smearing resolution.
                      If provided, slit smearing resolution (`dxl`) is calculated. 
                      If None, intrinsic probe resolution (`dQ`) is used for pinhole smearing.
        """
        self.sas_model = sas_model
        self.dtheta_l = dtheta_l
        
        if self.sas_model is not None:
            # 1. MIXIN RESPONSIBILITY: Convert Probe + dtheta_l -> Data1D
            sas_data = self._prepare_data(self.probe)
            # 2. BIND PURE DATA TO MODEL
            self.sas_model.bind(sas_data)
        
        # Register main SAS/Refl plot
        self.register_webview_plot(
            plot_title='SAS/Refl Decomposition',
            plot_function=sas_decomposition_plot,
            change_with='parameter'
        )
        
        # Register model-specific plots
        if self.sas_model is not None:
            plot_groups = self.sas_model.get_plots()
            
            for title, func in plot_groups.get('parameter', []):
                self.register_webview_plot(plot_title=title, plot_function=func, change_with='parameter')
                
            for title, func in plot_groups.get('uncertainty', []):
                self.register_webview_plot(plot_title=title, plot_function=func, change_with='uncertainty')

    def _prepare_data(self, probe_input: Any) -> List[Data1D]:
        """
        Convert Refl1D Probe/ProbeSet into a list of sasmodels Data1D objects.
        
        This method applies the resolution geometry logic.
        """
        if isinstance(probe_input, ProbeSet):
            raw_probes = probe_input.probes
        else:
            raw_probes = [probe_input]

        # Handle dtheta_l expansion
        if np.isscalar(self.dtheta_l) or self.dtheta_l is None:
            dtheta_list = [self.dtheta_l] * len(raw_probes)
        else:
            dtheta_list = self.dtheta_l # type: ignore

        data_list = []
        for probe, dtheta in zip(raw_probes, dtheta_list):
            # Create basic Data1D container
            data = Data1D(x=probe.Q)
            
            if dtheta is not None:
                # Slit Smearing: Calculate dxl based on dtheta/probe geometry
                data.dxl = dTdL2dQ(np.zeros_like(probe.T), dtheta, probe.L, probe.dL)
                # Ensure gaussian width is zero if using slit smearing (dxl) to avoid double smearing
                data.dxw = 2 * sigma2FWHM(probe.dQ)
            elif hasattr(probe, 'dQ'):
                # Pinhole/Gaussian Smearing: Map Refl1D 1-sigma dQ to sasmodels 1-sigma dx
                data.dx = probe.dQ
                data.dxl = None
            
            data_list.append(data)
        return data_list

    def parameters(self) -> Dict[str, Any]:
        base = super().parameters() # type: ignore
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
        Q, Rq = super().reflectivity(resolution, interpolation) # type: ignore
        if self.sas_model is not None:
            Rq = Rq + self.sas()
        return Q, Rq

@dataclass(init=False)
class SASReflectivityExperiment(SASReflectivityMixin, Refl1DExperiment):
    """
    Standard SAS + Reflectivity Experiment.
    Combines a standard Experiment with a SASModel.
    """
    sas_model: Optional[SASModel] = None
    dtheta_l: Optional[Union[float, List[float]]] = None
    
    def __init__(self, sas_model: Optional[SASModel] = None, dtheta_l=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._init_sas(sas_model, dtheta_l)

@dataclass(init=False)
class SASReflectivityMolgroupsExperiment(SASReflectivityMixin, MolgroupsExperiment):
    """
    Molgroups-Enabled SAS + Reflectivity Experiment.
    Combines a MolgroupsExperiment with a SASModel.
    """
    sas_model: Optional[SASModel] = None
    dtheta_l: Optional[Union[float, List[float]]] = None

    def __init__(self, sas_model: Optional[SASModel] = None, dtheta_l=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._init_sas(sas_model, dtheta_l)

        if isinstance(self.sas_model, MolgroupsSphereSASModel):
            self._molgroups_layers.update({self.sas_model.molgroups_layer.name: self.sas_model.molgroups_layer})


# =============================================================================
# 5. PLOTTING FUNCTIONS
# =============================================================================

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

def sans_profile_plot(experiment: Any, problem: Any = None) -> CustomWebviewPlot:
    """
    Unified plot for SANS SLD Profiles (Radius vs SLD).
    Works for any Experiment or Model that has a sas_model attribute or is a SASModel.
    """
    # Handle different calling conventions (Experiment vs SASModel)
    if hasattr(experiment, 'sas_model'):
        model = experiment.sas_model
    elif isinstance(experiment, SASModel):
        model = experiment
    elif hasattr(experiment, 'model') and isinstance(experiment.model, SASModel):
        model = experiment.model
    else:
        model = None

    if model is None:
        return CustomWebviewPlot(fig_type='plotly', plotdata=go.Figure(), exportdata="")

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