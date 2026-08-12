"""Refl1D interface classes for molgroups molecular groups.

Each class in this module wraps a lower-level ``mol.*`` object so that its
parameters are exposed as Refl1D ``Parameter`` objects, enabling bumps-based
fitting and model serialization.  The top-level workflow is:

1. Instantiate one or more interface classes, passing shared ``Parameter``
   objects to link parameters across groups.
2. Compose them into a :class:`~molgroups.refl1d_interface.MolgroupsLayer`.
3. Wrap that in a :class:`~molgroups.refl1d_interface.MolgroupsStack`.
4. Pass the stack to :class:`~molgroups.refl1d_interface.MolgroupsExperiment`.

---

Quick-reference: physical scenario → class
------------------------------------------

+---------------------------------------------------+---------------------------------------+
| Physical scenario                                 | Class                                 |
+===================================================+=======================================+
| Featureless substrate (Si, TiO2, …)              | :class:`Substrate`                    |
+---------------------------------------------------+---------------------------------------+
| Bilayer on Si/SiO2 (solid-supported)              | :class:`SolidSupportedBilayer`        |
+---------------------------------------------------+---------------------------------------+
| Bilayer on molecular tether + filler              | :class:`TetheredBilayer`              |
+---------------------------------------------------+---------------------------------------+
| Free-standing (floating) bilayer                  | :class:`Bilayer`                      |
+---------------------------------------------------+---------------------------------------+
| Single leaflet (Langmuir / asymmetric)            | :class:`Monolayer`                    |
+---------------------------------------------------+---------------------------------------+
| Protein/component with fixed volumes              | :class:`ComponentBox`                 |
+---------------------------------------------------+---------------------------------------+
| Protein/peptide (volume-fraction, H/D aware)      | :class:`VolumeFractionBox`            |
+---------------------------------------------------+---------------------------------------+
| Generic box with explicit SLD                     | :class:`VolumeBox`                    |
+---------------------------------------------------+---------------------------------------+
| Hermite-spline density profile                    | :class:`Freeform`                     |
+---------------------------------------------------+---------------------------------------+
| Rigid-body protein in Euler orientation           | :class:`ContinuousEuler`              |
+---------------------------------------------------+---------------------------------------+
| Parabolic polymer brush                           | :class:`PolymerBrush`                 |
+---------------------------------------------------+---------------------------------------+
| Low-density polymer mushroom                      | :class:`PolymerMushroom`              |
+---------------------------------------------------+---------------------------------------+
| Bilayer + protein complex (coupled area)          | :class:`BilayerProteinComplex`        |
+---------------------------------------------------+---------------------------------------+

---

ReferencePoint chaining
-----------------------
Every interface class exposes one or more **ReferencePoint** attributes.  A
``ReferencePoint`` is a read-only, auto-updating ``Parameter`` whose value is
recomputed from the underlying molgroups object after every ``update()`` call.

Reference points carry geometric meaning (e.g. "top of the outer headgroup")
and are the primary mechanism for positioning one group relative to another
without hardcoding numeric z-values::

    blm = SolidSupportedBilayer(name='bilayer', ...)

    # Place a protein box so its center is 5 Å above the outer headgroups.
    dz = Parameter(name='protein offset', value=5).range(0, 20)
    box = VolumeFractionBox(name='protein',
                            z=blm.outer_headgroup_top + dz, ...)

    # Place a spline starting at the top of the box.
    spline = Freeform(name='spline',
                      startz=box.top_surface, ...)

Arithmetic with a ``ReferencePoint`` produces a new ``bumps.Calculation``
object, which is accepted wherever a ``Parameter`` is accepted.
"""

from typing import List, Tuple, Callable, Dict, TypedDict, Type, TypeVar
from dataclasses import dataclass, field, fields
import copy
import uuid
import functools

import numpy as np

from scipy.integrate import trapezoid
from refl1d.names import Parameter
from bumps.parameter import Calculation

import molgroups.mol as mol

from molgroups.components import Component, Lipid, Tether, bme

from periodictable.fasta import H2O_SLD, D2O_SLD

def sld_from_bulk(rhoH: float, rhoD: float, bulknsld: float, protexchratio: float = 1.0) -> float:
    """Calculates scattering length density of material with
        labile hydrogens from bulk nSLD and SLDs in pure water and D2O

    Args:
        rhoH (float): nSLD in pure H2O
        rhoD (float): nSLD in pure D2O
        bulknsld (float): nSLD of bulk water

    Returns:
        float: nSLD of material
    """

    frac_d2o = (bulknsld - H2O_SLD) / (D2O_SLD - H2O_SLD)

    return rhoH + protexchratio * (rhoD - rhoH) * frac_d2o

class ReferencePoint(Parameter):

    def __init__(self, function: Callable | None = None, description: str = '', name: str | None = None, id: str | None = None, discrete: bool = False, tags: List[str] | None = None, **kw):
        calculation = Calculation(description=description)
        if function is not None:
            calculation.set_function(function)
        tags = [] if tags is None else tags
        kw.pop('fixed', None)
        kw.pop('slot', None)
        super().__init__(slot=calculation, fixed=True, name=name, id=id, discrete=discrete, tags=tags + ['Reference Point'], **kw)

    def set_function(self, function: Callable) -> None:

        self.slot.set_function(function)

@dataclass
class MolgroupsInterface:
    """Base class for interacting with molgroups objects
    """

    id: str | None = None
    name: str | None = None
    nf: Parameter = field(default_factory=lambda: Parameter(name='number fraction', value=1))
    bulknsld: Parameter = field(default_factory=lambda: Parameter(name='solvent rho', value=0.0))
    _molgroup: mol.nSLDObj | None = None
    _stored_profile: dict | None = None
    _group_names: dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:

        if self.id is None:
            self._generate_id()

        if not self._group_names:
            self._group_names = {f'{self.name}': [f'{self.name}']}

        for f in fields(self):
            if f.type == Parameter:
                default_name = f.default_factory().name
                p = getattr(self, f.name)
                if hasattr(p, 'name'):
                    if p.name == default_name:
                        p.name = f'{self.name} {p.name}'
                        setattr(self, f.name, p)
                else:
                    setattr(self, f.name, Parameter.default(p, name=f'{self.name} {default_name}'))
            elif f.type == List[Parameter]:
                plist = getattr(self, f.name)
                for i, p in enumerate(plist):
                    p = Parameter.default(p, name=f'{self.name} {f.name}{i}')
                    plist[i] = p
                setattr(self, f.name, plist)
            elif f.type == ReferencePoint:
                p: ReferencePoint = getattr(self, f.name)
                default_name = f.default_factory().name
                if p.name == default_name:
                    p.name = f'{self.name} {p.name}'
                setattr(self, f.name, p)

    def _generate_id(self):
        self.id = str(uuid.uuid4())

    def _get_parameters(self) -> dict[str, Parameter]:
        """Gets a list of the parameters associated with the interactor

        Returns:
            List[Parameter]: Parameter list
        """

        pars = {}
        for f in fields(self):
            if f.type in (Parameter, ReferencePoint):
                p = getattr(self, f.name)
                pars.update({f'{self.name} {p.name}': p})
            elif f.type == List[Parameter]:
                plist = getattr(self, f.name)
                for i, p in enumerate(plist):
                    pars.update({f'{self.name} {f.name}{i}': p})

        return pars

    def _set_bulknsld(self, bulknsld: Parameter):
        """Sets the bulknsld parameter. Allows subclassing for nested groups"""
        self.bulknsld = bulknsld

    def update(self) -> None:
        """Updates the molecular group with current values of the parameters,
            usually by calling fnSet
        """

        pass

    def old_render(self, z: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        """Renders the molecular group to an area and nSL

        Args:
            z (np.ndarray): spatial domain on which to render molecular group

        Returns:
            Tuple (float,np.ndarray, np.ndarray): normarea, area, nSL
        """

        normarea, area, nsl = self._molgroup.fnWriteProfile(z)

        return normarea, area, nsl

    def render(self, z: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        """Renders the molecular group to an area and nSL and stores result

        Args:
            z (np.ndarray): spatial domain on which to render molecular group

        Returns:
            Tuple (float,np.ndarray, np.ndarray): normarea, area, nSL
        """

        self.store_profile(z)
        area = self._stored_profile['area']
        nsl = self._stored_profile['sl']
        normarea = self._stored_profile['normarea']

        return normarea, area, nsl

    def store_profile(self, z: np.ndarray) -> dict:
        """Renders the molecular group and writes to a dict

        Args:
            z (np.ndarray): spatial domain on which to render molecular group

        Returns:
            dict: stored profile dictionary
        """

        self._stored_profile = self._molgroup.fnWriteGroup2Dict(dict(frac_replacement=1), self.name, z)
        self._stored_profile = self._molgroup.fnWriteProfile2Dict(self._stored_profile, z)
        self._stored_profile['normarea'] = self._stored_profile['area'].max()
        self._stored_profile['referencepoints'] = {p.name: p.value for p in self._get_parameters().values() if isinstance(p, ReferencePoint)}

    @property
    def group_names(self) -> dict[str, List[str]]:
        return self._group_names

    def _center_of_volume(self):

        if self._stored_profile is None:
            return 0.0

        z, area = self._stored_profile['zaxis'], self._stored_profile['area']

        return trapezoid(area * z, z) / trapezoid(area, z) if np.sum(area) else 0.0

@dataclass
class Bilayer(MolgroupsInterface):
    """Free-standing (floating) phospholipid bilayer.

    Models a symmetric bilayer suspended in solution, not attached to a
    substrate.  The bilayer position is set by ``startz``, which locates the
    inner hydrophobic interface.  Composition is defined per-leaflet via
    ``lipids``, ``inner_lipid_nf``, and ``outer_lipid_nf``.

    Attributes:
        lipids: List of :class:`~molgroups.components.Lipid` objects defining
            the bilayer composition (same list used for both leaflets).
        inner_lipid_nf: Number fractions of each lipid in the inner leaflet;
            must have the same length as ``lipids``.  Values are normalized
            internally.
        outer_lipid_nf: Number fractions of each lipid in the outer leaflet.
        xray_wavelength: If set (Å), nSL values are calculated for X-ray
            scattering at this wavelength.  ``None`` selects neutron mode.
        startz: z-position of the inner hydrophobic interface, Å (default 0.9).
            Use a ``ReferencePoint`` from another group to chain positions.
        vf_bilayer: Completeness of the bilayer (volume fraction of lipid
            relative to a perfect bilayer), unitless (default 0.9).
        l_hg1: Inner leaflet headgroup layer thickness, Å (default 10).
        l_lipid1: Inner leaflet acyl-chain region thickness, Å (default 10).
        l_lipid2: Outer leaflet acyl-chain region thickness, Å (default 10).
        l_hg2: Outer leaflet headgroup layer thickness, Å (default 10).
        sigma: Roughness applied to all bilayer interfaces, Å (default 5).
        normarea: Normalization area (read back from molgroups after update).
        nf: Overall number fraction scaling factor (default 1).
        bulknsld: Solvent neutron SLD in 10⁻⁶ Å⁻² units (set by
            :class:`~molgroups.refl1d_interface.MolgroupsLayer`).

    Reference Points:
        bilayer_center: Geometric center (midplane) of the bilayer, Å.
        inner_headgroup_bottom: Bottom of the inner leaflet headgroup layer, Å.
        inner_headgroup_center: Center of the inner leaflet headgroup layer, Å.
        inner_hydrophobic_interface: Interface between inner headgroups and acyl
            chains (equals ``startz``), Å.
        outer_hydrophobic_interface: Interface between outer acyl chains and
            outer headgroups, Å.
        outer_headgroup_center: Center of the outer leaflet headgroup layer, Å.
        outer_headgroup_top: Top of the outer leaflet headgroup layer, Å.

    Example::

        from molgroups.refl1d_interface import Bilayer
        from molgroups import components as cmp
        from refl1d.names import Parameter

        DOPC = cmp.Lipid(name='DOPC', headgroup=cmp.pc,
                         tails=2 * [cmp.oleoyl], methyls=[cmp.methyl])

        # Free-standing bilayer at fixed position 100 Å from the substrate
        ol = Bilayer(name='overlayer',
                     lipids=[DOPC],
                     inner_lipid_nf=[1.0],
                     outer_lipid_nf=[1.0],
                     startz=blm.outer_headgroup_top + dz_overlayer,
                     vf_bilayer=vf_overlayer,
                     l_lipid1=l_lipid1,
                     l_lipid2=l_lipid2,
                     sigma=sigma)
    """

    _molgroup: mol.BLM | None = None
    xray_wavelength: float | None = None

    lipids: List[Lipid] = field(default_factory=list)
    inner_lipid_nf: List[Parameter] = field(default_factory=list)
    outer_lipid_nf: List[Parameter] = field(default_factory=list)
    startz: Parameter = field(default_factory=lambda: Parameter(name='position of inner hydrophobic interface', value=0.9))
    vf_bilayer: Parameter = field(default_factory=lambda: Parameter(name='volume fraction', value=0.9))
    l_hg1: Parameter = field(default_factory=lambda: Parameter(name='inner headgroup thickness', value=10.0))
    l_lipid1: Parameter = field(default_factory=lambda: Parameter(name='inner acyl chain thickness', value=10.0))
    l_lipid2: Parameter = field(default_factory=lambda: Parameter(name='outer acyl chain thickness', value=10.0))
    l_hg2: Parameter = field(default_factory=lambda: Parameter(name='outer headgroup thickness', value=10.0))
    sigma: Parameter = field(default_factory=lambda: Parameter(name='roughness', value=5))
    normarea: Parameter = field(default_factory=lambda: Parameter(name='normarea', value=1))

    bilayer_center: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='bilayer_center', description='center of bilayer'))
    inner_headgroup_bottom: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='inner_headgroup_bottom', description='bottom of inner headgroups'))
    inner_headgroup_center: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='inner_headgroup_center', description='center of inner headgroups'))
    inner_hydrophobic_interface: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='inner_hydrophobic_interface', description='interface between inner headgroups and acyl chains'))
    outer_hydrophobic_interface: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='outer_hydrophobic_interface', description='interface between outer headgroups and acyl chains'))
    outer_headgroup_center: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='outer_headgroup_center', description='center of outer headgroups'))
    outer_headgroup_top: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='outer_headgroup_top', description='top of outer headgroups'))

    def __post_init__(self):
        self._molgroup = mol.BLM(inner_lipids=self.lipids,
                                 outer_lipids=self.lipids,
                                 inner_lipid_nf=[p.value if hasattr(p, 'value') else p for p in self.inner_lipid_nf],
                                 outer_lipid_nf=[p.value if hasattr(p, 'value') else p for p in self.outer_lipid_nf],
                                 xray_wavelength=self.xray_wavelength,
                                 name=self.name)

        n_lipids = len(self.lipids)
        self._group_names = {f'{self.name} inner headgroups': [f'{self.name}.headgroup1_{i}' for i in range(1, n_lipids + 1)],
                f'{self.name} inner acyl chains': [f'{self.name}.methylene1_{i}' for i in range(1, n_lipids + 1)] + [f'{self.name}.methyl1_{i}' for i in range(1, n_lipids + 1)],
                f'{self.name} outer acyl chains': [f'{self.name}.methylene2_{i}' for i in range(1, n_lipids + 1)] + [f'{self.name}.methyl2_{i}' for i in range(1, n_lipids + 1)],
                f'{self.name} outer headgroups': [f'{self.name}.headgroup2_{i}' for i in range(1, n_lipids + 1)],
                }

        # connect reference points
        self.bilayer_center.set_function(self._molgroup.fnGetCenter)
        self.inner_headgroup_bottom.set_function(self._inner_headgroup_bottom)
        self.inner_headgroup_center.set_function(self._inner_headgroup_center)
        self.inner_hydrophobic_interface.set_function(self._inner_hydrophobic_interface)
        self.outer_hydrophobic_interface.set_function(self._outer_hydrophobic_interface)
        self.outer_headgroup_center.set_function(self._outer_headgroup_center)
        self.outer_headgroup_top.set_function(self._outer_headgroup_top)

        super().__post_init__()

    def _inner_headgroup_bottom(self) -> float:
        """Returns the z position of the bottom of the inner headgroup

        Returns:
            float: z position of bottom of inner headgroup
        """

        return self._molgroup.z_ihc - 0.5 * self._molgroup.l_ihc - self._molgroup.av_hg1_l

    def _inner_headgroup_center(self) -> float:
        """Returns the z position of the center of the inner headgroup

        Returns:
            float: z position of center of inner headgroup
        """

        return self._molgroup.z_ihc - 0.5 * self._molgroup.l_ihc - 0.5 * self._molgroup.av_hg1_l

    def _inner_hydrophobic_interface(self) -> float:
        """Returns the z position of the inner hydrophobic interface
        
        Returns:

            float: z position of inner hydrophobic interface
        """
        return self._molgroup.z_ihc - 0.5 * self._molgroup.l_ihc

    def _outer_hydrophobic_interface(self) -> float:
        """Returns the z position of the outer hydrophobic interface

        Returns:
            float: z position of outer hydrophobic interface
        """

        return self._molgroup.z_ohc + 0.5 * self._molgroup.l_ohc

    def _outer_headgroup_center(self) -> float:
        """Returns the z position of the center of the outer headgroup

        Returns:
            float: z position of center of outer headgroup
        """

        return self._molgroup.z_ohc + 0.5 * self._molgroup.l_ohc + 0.5 * self._molgroup.av_hg2_l

    def _outer_headgroup_top(self) -> float:
        """Returns the z position of the top of the outer headgroup

        Returns:
            float: z position of top of outer headgroup
        """

        return self._molgroup.z_ohc + 0.5 * self._molgroup.l_ohc + self._molgroup.av_hg2_l

    def update(self):

        for hg in self._molgroup.headgroups1:
            hg.length = self.l_hg1.value

        for hg in self._molgroup.headgroups2:
            hg.length = self.l_hg2.value

        self._molgroup.fnSet(sigma=self.sigma.value,
            bulknsld=self.bulknsld.value * 1e-6,
            startz=self.startz.value,
            l_lipid1=self.l_lipid1.value,
            l_lipid2=self.l_lipid2.value,
            vf_bilayer=self.vf_bilayer.value,
            nf_inner_lipids=[p.value for p in self.inner_lipid_nf],
            nf_outer_lipids=[p.value for p in self.outer_lipid_nf],
            radius_defect=1e8)

        self.normarea.value = self._molgroup.normarea

@dataclass
class Monolayer(MolgroupsInterface):
    """Single phospholipid leaflet (monolayer / Langmuir film).

    Models a single leaflet of lipids with headgroups pointing away from the
    substrate (outward-facing geometry).  The position is set by ``startz``,
    which locates the hydrophobic (acyl-chain / headgroup) interface.

    Attributes:
        lipids: List of :class:`~molgroups.components.Lipid` objects defining
            the leaflet composition.
        lipid_nf: Number fractions of each lipid; must have the same length as
            ``lipids``.  Values are normalized internally.
        xray_wavelength: If set (Å), nSL values are calculated for X-ray
            scattering at this wavelength.  ``None`` selects neutron mode.
        startz: z-position of the hydrophobic interface (acyl-chain side),
            Å (default 20).  Use a ``ReferencePoint`` to chain positions.
        vf_lipids: Volume fraction of lipid in the monolayer, unitless
            (default 0.9).
        l_lipid: Acyl-chain region thickness, Å (default 10).
        l_hg: Headgroup layer thickness, Å (default 10).
        sigma: Roughness applied to all monolayer interfaces, Å (default 5).
        normarea: Normalization area (read back from molgroups after update).
        nf: Overall number fraction scaling factor (default 1).
        bulknsld: Solvent neutron SLD in 10⁻⁶ Å⁻² units (set by
            :class:`~molgroups.refl1d_interface.MolgroupsLayer`).

    Reference Points:
        acyl_chain_end: Position of the terminal acyl-chain end (innermost
            edge of the monolayer), Å.
        hydrophobic_interface: Interface between acyl chains and headgroups,
            Å.
        headgroup_center: Center of the headgroup layer, Å.
        headgroup_top: Outermost edge of the headgroup layer (solution side),
            Å.

    Example::

        # Deuterated DMPC Langmuir monolayer at the air/water interface.
        # Geometry (low z → high z): air | acyl chains | headgroups | water.
        # The air phase acts as the "substrate"; acyl chains point into the
        # air, headgroups project into the water subphase.

        from refl1d.names import Parameter, SLD, Slab, FitProblem, load4
        from molgroups import components as cmp
        from molgroups.refl1d_interface import (Substrate, Monolayer,
                                                MolgroupsLayer, MolgroupsStack,
                                                MolgroupsExperiment)

        # Shared structural parameters (fitted jointly across contrasts)
        sigma   = Parameter(name='roughness',           value=5).range(0.5, 9)
        l_lipid = Parameter(name='acyl chain thickness', value=12).range(8, 16)
        dz      = Parameter(name='interface shift',     value=0).range(-3, 3)
        overlap = 40.0

        air = SLD(name='air', rho=0.0)
        d2o = SLD(name='d2o', rho=6.36)
        h2o = SLD(name='h2o', rho=-0.56)

        dDMPC = cmp.Lipid(name='dDMPC', headgroup=cmp.pc,
                          tails=[cmp.d_myristoyl, cmp.d_myristoyl],
                          methyls=[cmp.dmethyl])

        def make_sample(substrate, contrast):
            # Air acts as the base "substrate"; its surface defines z=0 for
            # the molecular groups.
            air_box = Substrate(name='air', overlap=overlap,
                                rho=air.rho, sigma=sigma)

            # startz places the hydrophobic interface (acyl chain / headgroup
            # boundary) at the air/water interface, shifted by dz.
            mono = Monolayer(name='monolayer',
                             lipids=[dDMPC],
                             lipid_nf=[1.0],
                             startz=air_box.substrate_surface + dz + l_lipid,
                             l_lipid=l_lipid,
                             sigma=sigma)
            mono.vf_lipids.range(0, 1)

            mollayer = MolgroupsLayer(
                base_group=air_box,
                normarea_group=mono,   # normarea set by lipid packing density
                overlay_groups=[mono], # lipid displaces air (not just added)
                thickness=100.0,
                contrast=contrast,
                name='monolayer ' + contrast.name)
            return MolgroupsStack(substrate=substrate,
                                  molgroups_layer=mollayer)

        layer_air  = Slab(material=air, thickness=0.0, interface=0.0)
        sample_d2o = make_sample(layer_air, d2o)
        sample_h2o = make_sample(layer_air, h2o)

        model_d2o = MolgroupsExperiment(sample=sample_d2o, probe=probe_d2o, dz=0.5)
        model_h2o = MolgroupsExperiment(sample=sample_h2o, probe=probe_h2o, dz=0.5)
        problem = FitProblem([model_d2o, model_h2o])
    """

    _molgroup: mol.Monolayer | None = None
    xray_wavelength: float | None = None

    lipids: List[Lipid] = field(default_factory=list)
    lipid_nf: List[Parameter] = field(default_factory=list)
    startz: Parameter = field(default_factory=lambda: Parameter(name='position of hydrophobic interface', value=20))
    vf_lipids: Parameter = field(default_factory=lambda: Parameter(name='volume fraction', value=0.9))
    l_lipid: Parameter = field(default_factory=lambda: Parameter(name='acyl chain thickness', value=10.0))
    l_hg: Parameter = field(default_factory=lambda: Parameter(name='headgroup thickness', value=10.0))
    sigma: Parameter = field(default_factory=lambda: Parameter(name='roughness', value=5))
    normarea: Parameter = field(default_factory=lambda: Parameter(name='normarea', value=1))

    acyl_chain_end: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='acyl_chain_end', description='end of acyl chains'))
    hydrophobic_interface: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='hydrophobic_interface', description='interface between headgroups and acyl chains'))
    headgroup_center: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='outer_headgroup_center', description='center of outer headgroups'))
    headgroup_top: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='outer_headgroup_top', description='top of outer headgroups'))

    def __post_init__(self):
        self._molgroup = mol.Monolayer(lipids=self.lipids,
                                   lipid_nf=[p.value if hasattr(p, 'value') else p for p in self.lipid_nf],
                                   xray_wavelength=self.xray_wavelength,
                                   name=self.name)

        n_lipids = len(self.lipids)
        self._group_names = {
                f'{self.name} acyl chains': [f'{self.name}.methylene2_{i}' for i in range(1, n_lipids + 1)] + [f'{self.name}.methyl2_{i}' for i in range(1, n_lipids + 1)],
                f'{self.name} headgroups': [f'{self.name}.headgroup2_{i}' for i in range(1, n_lipids + 1)],
                }

        # connect reference points
        self.acyl_chain_end.set_function(self._molgroup.fnGetCenter)
        self.hydrophobic_interface.set_function(functools.partial(lambda blm: blm.z_ohc + 0.5 * blm.l_ohc, self._molgroup))
        self.headgroup_center.set_function(functools.partial(lambda blm: blm.z_ohc + 0.5 * blm.l_ohc + 0.5 * blm.av_hg2_l, self._molgroup))
        self.headgroup_top.set_function(functools.partial(lambda blm: blm.z_ohc + 0.5 * blm.l_ohc + blm.av_hg2_l, self._molgroup))

        super().__post_init__()

    def update(self):

        for hg in self._molgroup.headgroups2:
            hg.length = self.l_hg.value

        self._molgroup.fnSet(sigma=self.sigma.value,
            bulknsld=self.bulknsld.value * 1e-6,
            startz=self.startz.value,
            l_lipid2=self.l_lipid.value,
            vf_bilayer=self.vf_lipids.value,
            nf_lipids=[p.value for p in self.lipid_nf],
            radius_defect=1e8)

        self.normarea.value = self._molgroup.normarea

# ============= BaseGroup objects ===============

@dataclass
class BaseGroupInterface(MolgroupsInterface):
    """Interface specifically for base groups, i.e. those that occupy the edges of the molgroups canvas
    """

    normarea: Parameter | float = 1.0
    overlap: Parameter | float = 20.0

    def __post_init__(self) -> None:

        self.normarea = Parameter.default(self.normarea, name=f'{self.name} normarea', fixed=True)
        self.overlap = Parameter.default(self.overlap, name=f'{self.name} overlap', fixed=True)

        super().__post_init__()

@dataclass
class Substrate(BaseGroupInterface):
    """Featureless substrate slab used as the base group of the molgroups canvas.

    Represents a homogeneous substrate material (e.g. Si, TiO2, Au) as an
    error-function-broadened box.  It is always placed at z = 0 and extends
    into the negative-z half-space, with its surface at z = 0.  The substrate
    is only needed inside :class:`~molgroups.refl1d_interface.MolgroupsLayer`;
    the actual substrate material in the Refl1D layer stack is defined as a
    separate ``Slab``.

    Attributes:
        rho: Substrate neutron SLD in 10⁻⁶ Å⁻² (default 2.07, silicon).
        sigma: Roughness of the substrate surface, Å (default 2.07).
        normarea: In-plane normalization area in Å² (typically shared with the
            overlying bilayer; fixed, not fitted).
        overlap: Extent of the substrate box in the positive-z direction, Å
            (must match the overlap used in the Refl1D layer stack; fixed).

    Reference Points:
        substrate_surface: Top surface of the substrate box, Å.  Use this to
            position groups that sit directly on the substrate.

    Example::

        from molgroups.refl1d_interface import Substrate

        substrate_group = Substrate(name='substrate',
                                    overlap=overlap,
                                    rho=silicon.rho,
                                    sigma=global_rough)
    """

    _molgroup: mol.Box2Err | None = None

    rho: Parameter = field(default_factory=lambda: Parameter(name='rho substrate', value=2.07))
    sigma: Parameter = field(default_factory=lambda: Parameter(name='substrate roughness', value=2.07))

    substrate_surface: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='substrate_surface', description='surface of substrate'))

    def __post_init__(self) -> None:
        self._molgroup = mol.Box2Err(name=self.name)

        self.substrate_surface.set_function(functools.partial(lambda box: box.z + 0.5 * box.length, self._molgroup))

        super().__post_init__()

    def update(self):

        self._molgroup.fnSet(volume=self.normarea.value * self.overlap.value * 2.0,
                             length=2.0 * self.overlap.value,
                             position=0.0,
                             sigma=self.sigma.value,
                             nf=1.0,
                             nSL=self.normarea.value * self.overlap.value * 2.0 * self.rho.value * 1e-6)

@dataclass
class SolidSupportedBilayer(BaseGroupInterface):
    """Phospholipid bilayer on a planar solid support (ssBLM).

    Models a bilayer deposited on a solid substrate (typically Si/SiO2 or
    TiO2) with an optional silicon-oxide interlayer and a submembrane water
    gap between the substrate and the inner headgroups.

    This is the most common base group for membrane model fitting.  It serves
    as the ``base_group`` of a :class:`~molgroups.refl1d_interface.MolgroupsLayer`
    and exposes ReferencePoints that can be used to position additional groups
    (proteins, peptides, splines) relative to the bilayer geometry.

    Attributes:
        lipids: List of :class:`~molgroups.components.Lipid` objects defining
            the bilayer composition (same list used for both leaflets).
        inner_lipid_nf: Number fractions of each lipid in the inner leaflet;
            must have the same length as ``lipids``.
        outer_lipid_nf: Number fractions of each lipid in the outer leaflet.
        xray_wavelength: If set (Å), nSL values are calculated for X-ray
            scattering at this wavelength.  ``None`` selects neutron mode.
        rho_substrate: Substrate neutron SLD in 10⁻⁶ Å⁻² (default 2.07).
            Can be linked to the ``rho`` parameter of the matching Refl1D SLD
            material for joint refinement.
        rho_siox: Silicon oxide interlayer SLD in 10⁻⁶ Å⁻² (default 3.3).
        l_siox: Thickness of the silicon oxide interlayer, Å (default 0).
            Set to 0 to omit the SiO2 layer.
        vf_bilayer: Bilayer completeness (volume fraction of lipid relative to
            a perfect bilayer), unitless (default 0.9).
        l_hg1: Inner leaflet headgroup layer thickness, Å (default 10).
        l_lipid1: Inner leaflet acyl-chain region thickness, Å (default 10).
        l_lipid2: Outer leaflet acyl-chain region thickness, Å (default 10).
        l_hg2: Outer leaflet headgroup layer thickness, Å (default 10).
        sigma: Roughness applied to all bilayer interfaces, Å (default 5).
        substrate_rough: Roughness of the substrate surface, Å (default 5).
            Applied as a global roughness to the substrate and SiO2 layers.
        l_submembrane: Thickness of the water gap between the substrate
            surface and the bottom of the inner headgroups, Å (default 10).
            Negative values are clamped to zero internally.
        normarea: In-plane normalization area in Å² (read back from molgroups
            after each update; typically shared via ``overlap``).
        overlap: Extent of the substrate box in the positive-z direction, Å
            (must match the Refl1D layer stack overlap; fixed).
        nf: Overall number fraction scaling factor (default 1).
        bulknsld: Solvent neutron SLD in 10⁻⁶ Å⁻² units (set automatically
            by :class:`~molgroups.refl1d_interface.MolgroupsLayer`).

    Reference Points:
        substrate_surface: Top of the substrate box, Å.
        siox_surface: Top of the silicon oxide interlayer, Å.
        inner_headgroup_bottom: Bottom of the inner leaflet headgroup layer, Å.
        inner_headgroup_center: Center of the inner leaflet headgroup layer, Å.
        inner_hydrophobic_interface: Interface between inner headgroups and
            acyl chains, Å.
        bilayer_center: Geometric midplane of the bilayer, Å.
        outer_hydrophobic_interface: Interface between outer acyl chains and
            outer headgroups, Å.
        outer_headgroup_center: Center of the outer leaflet headgroup layer, Å.
        outer_headgroup_top: Top of the outer leaflet headgroup layer
            (solution side), Å.  Commonly used as the start position for
            groups added above the bilayer.

    Example::

        from molgroups.refl1d_interface import (SolidSupportedBilayer,
                                                VolumeFractionBox,
                                                MolgroupsLayer,
                                                MolgroupsStack,
                                                MolgroupsExperiment)
        from molgroups import components as cmp
        from refl1d.names import Parameter, SLD, Slab

        DOPC = cmp.Lipid(name='DOPC', headgroup=cmp.pc,
                         tails=2 * [cmp.oleoyl], methyls=[cmp.methyl])

        overlap = 30.0
        sigma   = Parameter(name='bilayer roughness', value=5).range(0.5, 9)
        tiox    = SLD(name='tiox', rho=2.163)

        blm = SolidSupportedBilayer(
            name='bilayer',
            overlap=overlap,
            lipids=[DOPC],
            inner_lipid_nf=[1.0],
            outer_lipid_nf=[1.0],
            rho_substrate=tiox.rho,
            l_siox=0.0,
            vf_bilayer=Parameter(value=0.9).range(0, 1),
            l_lipid1=Parameter(value=10.0).range(8, 30),
            l_lipid2=Parameter(value=10.0).range(8, 18),
            sigma=sigma)

        # Position a protein box above the outer headgroups
        dz  = Parameter(name='protein offset', value=5).range(0, 20)
        box = VolumeFractionBox(name='protein',
                                z=blm.outer_headgroup_top + dz, ...)

        mollayer = MolgroupsLayer(base_group=blm,
                                  add_groups=[box],
                                  thickness=200.0,
                                  contrast=contrast)
    """

    _molgroup: mol.ssBLM | None = None
    xray_wavelength: float | None = None

    lipids: List[Lipid] = field(default_factory=list)
    inner_lipid_nf: List[Parameter] = field(default_factory=list)
    outer_lipid_nf: List[Parameter] = field(default_factory=list)
    rho_substrate: Parameter = field(default_factory=lambda: Parameter(name='rho substrate', value=2.07))
    rho_siox: Parameter = field(default_factory=lambda: Parameter(name='rho siox', value=3.3))
    l_siox: Parameter = field(default_factory=lambda: Parameter(name='siox thickness', value=0.0))
    vf_bilayer: Parameter = field(default_factory=lambda: Parameter(name='volume fraction bilayer', value=0.9))
    l_hg1: Parameter = field(default_factory=lambda: Parameter(name='inner headgroup thickness', value=10.0))
    l_lipid1: Parameter = field(default_factory=lambda: Parameter(name='inner acyl chain thickness', value=10.0))
    l_lipid2: Parameter = field(default_factory=lambda: Parameter(name='outer acyl chain thickness', value=10.0))
    l_hg2: Parameter = field(default_factory=lambda: Parameter(name='outer headgroup thickness', value=10.0))
    sigma: Parameter = field(default_factory=lambda: Parameter(name='bilayer roughness', value=5))
    substrate_rough: Parameter = field(default_factory=lambda: Parameter(name ='substrate roughness', value=5))
    l_submembrane: Parameter = field(default_factory=lambda: Parameter(name='submembrane thickness', value=10))

    substrate_surface: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='substrate_surface', description='surface of substrate'))
    siox_surface: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='siox_surface', description='surface of siox layer'))
    bilayer_center: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='bilayer_center', description='center of bilayer'))
    inner_headgroup_bottom: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='inner_headgroup_bottom', description='bottom of inner headgroups'))
    inner_headgroup_center: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='inner_headgroup_center', description='center of inner headgroups'))
    inner_hydrophobic_interface: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='inner_hydrophobic_interface', description='interface between inner headgroups and acyl chains'))
    outer_hydrophobic_interface: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='outer_hydrophobic_interface', description='interface between outer headgroups and acyl chains'))
    outer_headgroup_center: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='outer_headgroup_center', description='center of outer headgroups'))
    outer_headgroup_top: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='outer_headgroup_top', description='top of outer headgroups'))

    def __post_init__(self):
        self._molgroup = mol.ssBLM(inner_lipids=self.lipids,
                               outer_lipids=self.lipids,
                             inner_lipid_nf=[p.value if hasattr(p, 'value') else p for p in self.inner_lipid_nf],
                             outer_lipid_nf=[p.value if hasattr(p, 'value') else p for p in self.outer_lipid_nf],
                             xray_wavelength=self.xray_wavelength,
                             name=self.name)

        n_lipids = len(self.lipids)
        self._group_names = {'substrate': [f'{self.name}.substrate'],
                'silicon dioxide': [f'{self.name}.siox'],
                f'{self.name} inner headgroups': [f'{self.name}.headgroup1_{i}' for i in range(1, n_lipids + 1)],
                f'{self.name} inner acyl chains': [f'{self.name}.methylene1_{i}' for i in range(1, n_lipids + 1)] + [f'{self.name}.methyl1_{i}' for i in range(1, n_lipids + 1)],
                f'{self.name} outer acyl chains': [f'{self.name}.methylene2_{i}' for i in range(1, n_lipids + 1)] + [f'{self.name}.methyl2_{i}' for i in range(1, n_lipids + 1)],
                f'{self.name} outer headgroups': [f'{self.name}.headgroup2_{i}' for i in range(1, n_lipids + 1)],
                }

        # connect reference points
        self.substrate_surface.set_function(functools.partial(lambda blm: blm.substrate.z + 0.5 * blm.substrate.length, self._molgroup))
        self.siox_surface.set_function(functools.partial(lambda blm: blm.siox.z + 0.5 * blm.siox.length, self._molgroup))
        self.bilayer_center.set_function(self._molgroup.fnGetCenter)
        self.inner_headgroup_bottom.set_function(functools.partial(lambda blm: blm.z_ihc - 0.5 * blm.l_ihc - blm.av_hg1_l, self._molgroup))
        self.inner_headgroup_center.set_function(functools.partial(lambda blm: blm.z_ihc - 0.5 * blm.l_ihc - 0.5 * blm.av_hg1_l, self._molgroup))
        self.inner_hydrophobic_interface.set_function(functools.partial(lambda blm: blm.z_ihc - 0.5 * blm.l_ihc, self._molgroup))
        self.outer_hydrophobic_interface.set_function(functools.partial(lambda blm: blm.z_ohc + 0.5 * blm.l_ohc, self._molgroup))
        self.outer_headgroup_center.set_function(functools.partial(lambda blm: blm.z_ohc + 0.5 * blm.l_ohc + 0.5 * blm.av_hg2_l, self._molgroup))
        self.outer_headgroup_top.set_function(functools.partial(lambda blm: blm.z_ohc + 0.5 * blm.l_ohc + blm.av_hg2_l, self._molgroup))

        super().__post_init__()

    def update(self):

        self._molgroup.substrate.length = 2.0 * self.overlap.value

        dl_submembrane = 0.0 if self.l_submembrane.value > 0 else -self.l_submembrane.value

        for hg in self._molgroup.headgroups1:
            hg.length = self.l_hg1.value - dl_submembrane

        for hg in self._molgroup.headgroups2:
            hg.length = self.l_hg2.value

        self._molgroup.fnSet(sigma=self.sigma.value,
            bulknsld=self.bulknsld.value * 1e-6,
            global_rough=self.substrate_rough.value,
            rho_substrate=self.rho_substrate.value * 1e-6,
            rho_siox=self.rho_siox.value * 1e-6,
            l_lipid1=self.l_lipid1.value,
            l_lipid2=self.l_lipid2.value,
            l_siox=self.l_siox.value,
            l_submembrane=max(0, self.l_submembrane.value),
            vf_bilayer=self.vf_bilayer.value,
            nf_inner_lipids=[p.value for p in self.inner_lipid_nf],
            nf_outer_lipids=[p.value for p in self.outer_lipid_nf],
            radius_defect=1e8)

        self.normarea.value = self._molgroup.normarea

@dataclass
class TetheredBilayer(BaseGroupInterface):
    """Phospholipid bilayer on a molecular tether (tBLM).

    Models a bilayer tethered to the substrate through lipid-like tether
    molecules mixed with a short filler (typically beta-mercaptoethanol, bME).
    The tether molecule occupies the inner leaflet alongside regular lipids,
    while the filler fills gaps at the substrate surface.

    Attributes:
        tether: :class:`~molgroups.components.Tether` object describing the
            tether molecule chemistry.
        filler: :class:`~molgroups.components.Component` object describing the
            filler molecule (default: bME).
        lipids: List of :class:`~molgroups.components.Lipid` objects defining
            the bilayer composition (same list used for both leaflets).
        inner_lipid_nf: Number fractions of each lipid in the inner leaflet.
        outer_lipid_nf: Number fractions of each lipid in the outer leaflet.
        xray_wavelength: If set (Å), nSL values are calculated for X-ray
            scattering at this wavelength.  ``None`` selects neutron mode.
        rho_substrate: Substrate neutron SLD in 10⁻⁶ Å⁻² (default 2.07).
        vf_bilayer: Bilayer completeness (volume fraction of lipid relative to
            a perfect bilayer), unitless (default 0.9).
        l_hg1: Inner leaflet headgroup layer thickness, Å (default 10).
        l_lipid1: Inner leaflet acyl-chain region thickness, Å (default 10).
        l_lipid2: Outer leaflet acyl-chain region thickness, Å (default 10).
        l_hg2: Outer leaflet headgroup layer thickness, Å (default 10).
        sigma: Roughness applied to all bilayer interfaces, Å (default 5).
        substrate_rough: Global roughness of the substrate surface, Å
            (default 5).
        l_tether: Length of the tether molecule, Å (default 10).
        nf_tether: Number fraction of tether molecules in the inner leaflet,
            unitless (default 0.45).
        mult_tether: Ratio of filler (bME) molecules to tether molecules at
            the substrate surface (default 3).
        normarea: In-plane normalization area in Å².
        overlap: Extent of the substrate box in the positive-z direction, Å.
        nf: Overall number fraction scaling factor (default 1).
        bulknsld: Solvent neutron SLD in 10⁻⁶ Å⁻² units (set automatically
            by :class:`~molgroups.refl1d_interface.MolgroupsLayer`).

    Reference Points:
        substrate_surface: Top of the substrate box, Å.
        filler_surface: Top of the filler (bME) layer, Å.  Marks the bottom
            of the tether headgroup region.
        inner_headgroup_bottom: Bottom of the inner leaflet headgroup layer, Å.
        inner_headgroup_center: Center of the inner leaflet headgroup layer, Å.
        inner_hydrophobic_interface: Interface between inner headgroups and
            acyl chains, Å.
        bilayer_center: Geometric midplane of the bilayer, Å.
        outer_hydrophobic_interface: Interface between outer acyl chains and
            outer headgroups, Å.
        outer_headgroup_center: Center of the outer leaflet headgroup layer, Å.
        outer_headgroup_top: Top of the outer leaflet headgroup layer, Å.

    Example::

        from molgroups.refl1d_interface import TetheredBilayer
        from molgroups import components as cmp

        DPPC = cmp.Lipid(name='DPPC', headgroup=cmp.pc,
                         tails=2 * [cmp.palmitic], methyls=[cmp.methyl])

        blm = TetheredBilayer(
            name='tBLM',
            overlap=overlap,
            tether=cmp.WC14,
            filler=cmp.bme,
            lipids=[DPPC],
            inner_lipid_nf=[1.0],
            outer_lipid_nf=[1.0],
            rho_substrate=silicon.rho,
            l_tether=Parameter(value=10).range(5, 20),
            nf_tether=Parameter(value=0.45).range(0.2, 0.7),
            mult_tether=Parameter(value=3).range(1, 6),
            vf_bilayer=Parameter(value=0.9).range(0, 1),
            sigma=sigma)
    """

    _molgroup: mol.tBLM | None = None
    xray_wavelength: float | None = None

    tether: Tether = field(default_factory=Tether)
    filler: Component = field(default_factory=lambda: bme)
    lipids: List[Lipid] = field(default_factory=list)
    inner_lipid_nf: List[Parameter] = field(default_factory=list)
    outer_lipid_nf: List[Parameter] = field(default_factory=list)
    rho_substrate: Parameter = field(default_factory=lambda: Parameter(name='rho substrate', value=2.07))
    vf_bilayer: Parameter = field(default_factory=lambda: Parameter(name='volume fraction bilayer', value=0.9))
    l_hg1: Parameter = field(default_factory=lambda: Parameter(name='inner headgroup thickness', value=10.0))
    l_lipid1: Parameter = field(default_factory=lambda: Parameter(name='inner acyl chain thickness', value=10.0))
    l_lipid2: Parameter = field(default_factory=lambda: Parameter(name='outer acyl chain thickness', value=10.0))
    l_hg2: Parameter = field(default_factory=lambda: Parameter(name='outer headgroup thickness', value=10.0))
    sigma: Parameter = field(default_factory=lambda: Parameter(name='bilayer roughness', value=5))
    substrate_rough: Parameter = field(default_factory=lambda: Parameter(name ='substrate roughness', value=5))
    l_tether: Parameter = field(default_factory=lambda: Parameter(name='tether length', value=10))
    nf_tether: Parameter = field(default_factory=Parameter(name='number fraction tether', value=0.45)) # number fraction of tether molecules in inner leaflet
    mult_tether: Parameter = field(default_factory=Parameter(name='bME to tether ratio', value=3)) #ratio of bME to tether molecules at surface

    substrate_surface: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='substrate_surface', description='surface of substrate'))
    filler_surface: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='filler_surface', description='surface of filler molecule'))
    bilayer_center: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='bilayer_center', description='center of bilayer'))
    inner_headgroup_bottom: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='inner_headgroup_bottom', description='bottom of inner headgroups'))
    inner_headgroup_center: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='inner_headgroup_center', description='center of inner headgroups'))
    inner_hydrophobic_interface: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='inner_hydrophobic_interface', description='interface between inner headgroups and acyl chains'))
    outer_hydrophobic_interface: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='outer_hydrophobic_interface', description='interface between outer headgroups and acyl chains'))
    outer_headgroup_center: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='outer_headgroup_center', description='center of outer headgroups'))
    outer_headgroup_top: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='outer_headgroup_top', description='top of outer headgroups'))

    def __post_init__(self):
        self._molgroup = mol.tBLM(tether=self.tether,
                              filler=self.filler,
                              inner_lipids=self.lipids,
                              outer_lipids=self.lipids,
                              inner_lipid_nf=[p.value if hasattr(p, 'value') else p for p in self.inner_lipid_nf],
                              outer_lipid_nf=[p.value if hasattr(p, 'value') else p for p in self.outer_lipid_nf],
                              xray_wavelength=self.xray_wavelength,
                              name=self.name)

        n_lipids = len(self.lipids)
        self._group_names = {'substrate': [f'{self.name}.substrate'],
                f'{self.name} bME': [f'{self.name}.bME'],
                f'{self.name} tether': [f'{self.name}.tether_bme', f'{self.name}.tether_free', f'{self.name}.tether_hg'],
                f'{self.name} tether acyl chains': [f'{self.name}.tether_methylene', f'{self.name}.tether_methyl'],
                f'{self.name} inner headgroups': [f'{self.name}.headgroup1_{i}' for i in range(1, n_lipids + 1)],
                f'{self.name} inner acyl chains': [f'{self.name}.methylene1_{i}' for i in range(1, n_lipids + 1)] + [f'{self.name}.methyl1_{i}' for i in range(1, n_lipids + 1)],
                f'{self.name} outer acyl chains': [f'{self.name}.methylene2_{i}' for i in range(1, n_lipids + 1)] + [f'{self.name}.methyl2_{i}' for i in range(1, n_lipids + 1)],
                f'{self.name} outer headgroups': [f'{self.name}.headgroup2_{i}' for i in range(1, n_lipids + 1)],
                }

        self.substrate_surface.set_function(functools.partial(lambda blm: blm.substrate.z + 0.5 * blm.substrate.length, self._molgroup))
        self.filler_surface.set_function(functools.partial(lambda blm: blm.bme.z + 0.5 * blm.bme.length, self._molgroup))
        self.bilayer_center.set_function(self._molgroup.fnGetCenter)
        self.inner_headgroup_bottom.set_function(functools.partial(lambda blm: blm.z_ihc - 0.5 * blm.l_ihc - blm.av_hg1_l, self._molgroup))
        self.inner_headgroup_center.set_function(functools.partial(lambda blm: blm.z_ihc - 0.5 * blm.l_ihc - 0.5 * blm.av_hg1_l, self._molgroup))
        self.inner_hydrophobic_interface.set_function(functools.partial(lambda blm: blm.z_ihc - 0.5 * blm.l_ihc, self._molgroup))
        self.outer_hydrophobic_interface.set_function(functools.partial(lambda blm: blm.z_ohc + 0.5 * blm.l_ohc, self._molgroup))
        self.outer_headgroup_center.set_function(functools.partial(lambda blm: blm.z_ohc + 0.5 * blm.l_ohc + 0.5 * blm.av_hg2_l, self._molgroup))
        self.outer_headgroup_top.set_function(functools.partial(lambda blm: blm.z_ohc + 0.5 * blm.l_ohc + blm.av_hg2_l, self._molgroup))

        super().__post_init__()

    def update(self):

        self._molgroup.substrate.length = 2.0 * self.overlap.value

        for hg in self._molgroup.headgroups1:
            hg.length = self.l_hg1.value

        for hg in self._molgroup.headgroups2:
            hg.length = self.l_hg2.value

        self._molgroup.fnSet(sigma=self.sigma.value,
            bulknsld=self.bulknsld.value * 1e-6,
            global_rough=self.substrate_rough.value,
            rho_substrate=self.rho_substrate.value * 1e-6,
            l_lipid1=self.l_lipid1.value,
            l_lipid2=self.l_lipid2.value,
            l_tether=self.l_tether.value,
            vf_bilayer=self.vf_bilayer.value,
            nf_tether=self.nf_tether.value,
            mult_tether=self.mult_tether.value,
            nf_inner_lipids=[p.value for p in self.inner_lipid_nf],
            nf_outer_lipids=[p.value for p in self.outer_lipid_nf],
            radius_defect=1e8)

        self.normarea.value = self._molgroup.normarea

# ============= Box-type objects ===============
@dataclass
class VolumeBox(MolgroupsInterface):
    """Error-function-broadened box with explicit SLD in H₂O and D₂O.

    Represents a layer of material defined by its volume, thickness, and
    separate SLD values for H₂O and D₂O contrasts.  The box center is set by
    ``z``.  The nSLD at each contrast is computed directly from ``rhoH`` and
    ``rhoD``; for H/D-aware automatic interpolation use
    :class:`VolumeFractionBox` instead.

    Attributes:
        z: Position of the box center, Å (default 0).
        rhoH: Neutron SLD of the material in pure H₂O, 10⁻⁶ Å⁻² (default
            2.07).
        rhoD: Neutron SLD of the material in pure D₂O, 10⁻⁶ Å⁻² (default
            2.07).
        volume: Total molecular volume of the material in the box, Å³
            (default 10).  Together with ``length`` and ``normarea`` this sets
            the area-fraction profile.
        length: Thickness of the box layer, Å (default 10).
        sigma_bottom: Roughness of the bottom (substrate-side) interface, Å
            (default 2.5).
        sigma_top: Roughness of the top (solution-side) interface, Å (default
            2.5).
        nf: Number fraction of the material (scales total occupancy, default
            1).
        bulknsld: Solvent neutron SLD in 10⁻⁶ Å⁻² units (set automatically
            by :class:`~molgroups.refl1d_interface.MolgroupsLayer`).

    Reference Points:
        bottom_surface: Bottom of the box, Å (= ``z`` − ``length``/2).
        top_surface: Top of the box, Å (= ``z`` + ``length``/2).

    Example::

        from molgroups.refl1d_interface import VolumeBox

        oxide = VolumeBox(name='oxide layer',
                          z=blm.substrate_surface + 0.5 * l_oxide,
                          rhoH=4.1,
                          rhoD=4.1,
                          volume=volume_oxide,
                          length=l_oxide,
                          sigma_bottom=sigma_sub,
                          sigma_top=sigma_sub)
    """

    _molgroup: mol.Box2Err | None = None

    z: Parameter = field(default_factory=lambda: Parameter(name='center position', value=0))
    rhoH: Parameter = field(default_factory=lambda: Parameter(name='rho in H2O', value=2.07))
    rhoD: Parameter = field(default_factory=lambda: Parameter(name='rho in D2O', value=2.07))
    volume: Parameter = field(default_factory=lambda: Parameter(name='volume', value=10))
    length: Parameter = field(default_factory=lambda: Parameter(name='length', value=10))
    sigma_bottom: Parameter = field(default_factory=lambda: Parameter(name='roughness of bottom interface', value=2.5))
    sigma_top: Parameter = field(default_factory=lambda: Parameter(name='roughness of top interface', value=2.5))

    bottom_surface: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='bottom_surface', description='bottom of box'))
    top_surface: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='bottom_surface', description='top of box'))

    def __post_init__(self) -> None:
        self._molgroup = mol.Box2Err(name=self.name)

        self.bottom_surface.set_function(functools.partial(lambda box: box.z - 0.5 * box.length, self._molgroup))
        self.top_surface.set_function(functools.partial(lambda box: box.z + 0.5 * box.length, self._molgroup))

        super().__post_init__()

    def update(self):

        self._molgroup.fnSetBulknSLD(self.bulknsld.value * 1e-6)
        self._molgroup.fnSet(volume=self.volume.value,
                             length=self.length.value,
                             position=self.z.value,
                             sigma=(self.sigma_bottom.value,
                                    self.sigma_top.value),
                             nf=self.nf.value,
                             nSL=(self.volume.value * self.rhoH.value * 1e-6,
                                  self.volume.value * self.rhoD.value * 1e-6))

@dataclass
class ComponentBox(MolgroupsInterface):
    """Box layer defined by a list of chemical components with fixed volumes
    and fixed SLDs.

    Each :class:`~molgroups.components.Component` contributes a fixed volume
    and a fixed nSL; their sum determines the box SLD.  Use ``nf`` together
    with the normalization area to set the effective occupancy.  Contrast
    variation is handled component-by-component via ``diff_components``.

    Attributes:
        components: List of :class:`~molgroups.components.Component` objects
            that make up the box material.
        diff_components: Additional components whose SLD changes with contrast
            (e.g. labile-hydrogen components for H/D contrast variation).
        xray_wavelength: If set (Å), nSL values are calculated for X-ray
            scattering at this wavelength.  ``None`` selects neutron mode.
        z: Position of the box center, Å (default 0).  Use a ReferencePoint
            for chained positioning.
        length: Thickness of the box layer, Å (default 10).
        sigma_bottom: Roughness of the bottom interface, Å (default 2.5).
        sigma_top: Roughness of the top interface, Å (default 2.5).
        nf: Number fraction (overall occupancy scaling, default 1).  To set
            a volume fraction from an external parameter use::

                box.nf = vf / (box.volume / (box.length * normarea))

        bulknsld: Solvent neutron SLD in 10⁻⁶ Å⁻² units (set automatically
            by :class:`~molgroups.refl1d_interface.MolgroupsLayer`).

    Reference Points:
        bottom_surface: Bottom of the box, Å.
        top_surface: Top of the box, Å.
        volume: Sum of component volumes, Å³ (read-only, recomputed after
            each update).

    Example::

        from molgroups.refl1d_interface import ComponentBox
        from molgroups import components as cmp

        peptide_box = ComponentBox(name='peptide',
                                   components=[cmp.peptide_component],
                                   z=blm.outer_headgroup_top + dz,
                                   length=l_peptide,
                                   sigma_bottom=sigma,
                                   sigma_top=sigma)
    """

    _molgroup: mol.ComponentBox | None = None
    components: List[Component] = field(default_factory=list)
    diff_components: List[Component] = field(default_factory=list)
    xray_wavelength: float | None = None

    z: Parameter = field(default_factory=lambda: Parameter(name='center position', value=0))
    length: Parameter = field(default_factory=lambda: Parameter(name='length', value=10))
    sigma_bottom: Parameter = field(default_factory=lambda: Parameter(name='roughness of bottom interface', value=2.5))
    sigma_top: Parameter = field(default_factory=lambda: Parameter(name='roughness of top interface', value=2.5))

    bottom_surface: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='bottom_surface', description='bottom of box'))
    top_surface: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='bottom_surface', description='top of box'))
    volume: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='component_volume', description='sum of component volumes'))

    def __post_init__(self) -> None:
        self._molgroup = mol.ComponentBox(name=self.name,
                                      components=self.components,
                                      diffcomponents=self.diff_components,
                                      xray_wavelength=self.xray_wavelength)

        self.bottom_surface.set_function(functools.partial(lambda box: box.z - 0.5 * box.length, self._molgroup))
        self.top_surface.set_function(functools.partial(lambda box: box.z + 0.5 * box.length, self._molgroup))
        self.volume.set_function(functools.partial(lambda box: box.vol, self._molgroup))

        super().__post_init__()

    def update(self):

        self._molgroup.fnSetBulknSLD(self.bulknsld.value * 1e-6)
        self._molgroup.fnSet(length=self.length.value,
                             position=self.z.value,
                             sigma=(self.sigma_bottom.value, self.sigma_top.value),
                             nf=self.nf.value)

@dataclass
class VolumeFractionBox(MolgroupsInterface):
    """H/D-aware volume-fraction box for proteins, peptides, or other
    hydrogenous materials.

    Represents a layer of material at a specified volume fraction with
    automatic interpolation of the neutron SLD between ``rhoH`` (pure H₂O)
    and ``rhoD`` (pure D₂O) according to the current bulk solvent SLD.
    Labile-proton exchange is accounted for via ``proton_exchange_efficiency``.

    This is the recommended class for modelling membrane-associated proteins
    or peptides when only the volume fraction (not the absolute amount) is
    known.

    Attributes:
        z: Position of the box center, Å (default 0).  Use a ReferencePoint
            for chained positioning relative to the bilayer.
        rhoH: Neutron SLD of the material in pure H₂O, 10⁻⁶ Å⁻² (default
            2.07).
        rhoD: Neutron SLD of the material in pure D₂O, 10⁻⁶ Å⁻² (default
            2.07).
        proton_exchange_efficiency: Fraction of labile protons that exchange
            with the solvent, unitless (0–1, default 1.0).  Set < 1 for
            buried or partially exchanged sites.
        volume_fraction: Volume fraction of the material in the box, unitless
            (default 1).
        length: Thickness of the box layer, Å (default 10).
        sigma_bottom: Roughness of the bottom interface, Å (default 2.5).
        sigma_top: Roughness of the top interface, Å (default 2.5).
        nf: Overall number fraction scaling factor (default 1).
        bulknsld: Solvent neutron SLD in 10⁻⁶ Å⁻² units (set automatically
            by :class:`~molgroups.refl1d_interface.MolgroupsLayer`).

    Reference Points:
        bottom_surface: Bottom of the box, Å (= ``z`` − ``length``/2).
        top_surface: Top of the box, Å (= ``z`` + ``length``/2).  Commonly
            used as the start position for groups stacked above this box.

    Example::

        from molgroups.refl1d_interface import SolidSupportedBilayer, VolumeFractionBox
        from refl1d.names import Parameter
        from periodictable.fasta import Sequence

        peptide = Sequence(name='peptide', sequence='ACDEFGHIK', type='aa')

        l_protein = Parameter(name='protein length', value=30).range(10, 60)
        vf_protein = Parameter(name='protein volume fraction', value=0.5).range(0, 1)

        blm = SolidSupportedBilayer(name='bilayer', ...)

        protein = VolumeFractionBox(
            name='protein',
            z=blm.outer_headgroup_top + 0.5 * l_protein,
            rhoH=peptide.sld(contrast='H2O'),
            rhoD=peptide.sld(contrast='D2O'),
            proton_exchange_efficiency=0.9,
            volume_fraction=vf_protein,
            length=l_protein,
            sigma_bottom=sigma,
            sigma_top=sigma)
    """

    _molgroup: mol.ProteinBox | None = None

    z: Parameter = field(default_factory=lambda: Parameter(name='center position', value=0))
    rhoH: Parameter = field(default_factory=lambda: Parameter(name='rho in H2O', value=2.07))
    rhoD: Parameter = field(default_factory=lambda: Parameter(name='rho in D2O', value=2.07))
    proton_exchange_efficiency: Parameter = field(default_factory=lambda: Parameter(name='proton exchange efficiency', value=1.0))
    volume_fraction: Parameter = field(default_factory=lambda: Parameter(name='volume fraction', value=1))
    length: Parameter = field(default_factory=lambda: Parameter(name='length', value=10))
    sigma_bottom: Parameter = field(default_factory=lambda: Parameter(name='roughness of bottom interface', value=2.5))
    sigma_top: Parameter = field(default_factory=lambda: Parameter(name='roughness of top interface', value=2.5))

    bottom_surface: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='bottom_surface', description='bottom of box'))
    top_surface: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='bottom_surface', description='top of box'))

    def __post_init__(self) -> None:
        self._molgroup = mol.ProteinBox(name=self.name)

        self.bottom_surface.set_function(functools.partial(lambda box: box.z - 0.5 * box.length, self._molgroup))
        self.top_surface.set_function(functools.partial(lambda box: box.z + 0.5 * box.length, self._molgroup))

        super().__post_init__()

    def update(self):

        self._molgroup.fnSetBulknSLD(self.bulknsld.value * 1e-6)
        self._molgroup.protexchratio = self.proton_exchange_efficiency.value
        self._molgroup.fnSet(volume_fraction=self.volume_fraction.value,
                             nSLD=(self.rhoH.value * 1e-6,
                                   self.rhoD.value * 1e-6),
                             length=self.length.value,
                             position=self.z.value,
                             sigma=(self.sigma_bottom.value,
                                    self.sigma_top.value),
                             nf=self.nf.value)

@dataclass
class TetheredBox(MolgroupsInterface):
    """Not yet implemented."""
    pass

@dataclass
class TetheredBoxDouble(MolgroupsInterface):
    """Not yet implemented."""
    pass

# ============= Spline objects ================
@dataclass
class Freeform(MolgroupsInterface):
    """Hermite-spline (freeform) density profile for arbitrary layers.

    Represents a layer whose volume-fraction profile is described by a
    Hermite cubic spline over a set of equally spaced knots.  The knot
    positions and volume fractions are fit parameters, giving maximum
    flexibility for modelling adsorbed peptides, polymers, or other
    structurally undefined layers.

    The spline starts at ``startz`` and extends over
    ``len(Dp) * dSpacing`` Å.  Each knot *i* is displaced from its nominal
    position by ``Dp[i]`` Å, and has a volume fraction ``Vf[i]``.

    The SLD at each position is computed from ``rhoH`` and ``rhoD`` by
    interpolating with respect to the bulk solvent SLD
    (``bulknsld``), accounting for ``proton_exchange_efficiency``.

    Attributes:
        dSpacing: Nominal spacing between Hermite knots, Å (default 15, not
            a ``Parameter`` — not fitted).
        startz: z-position of the first knot, Å (default 20).  Use a
            ``ReferencePoint`` from another group to chain positions, e.g.
            ``startz=blm.outer_headgroup_top``.
        Dp: List of knot position offsets relative to the nominal grid, Å.
            Length determines the number of knots.
        Vf: List of volume fractions at each knot, unitless (0–1).  Must have
            the same length as ``Dp``.
        rhoH: Neutron SLD of the spline material in pure H₂O, 10⁻⁶ Å⁻²
            (default 0).
        rhoD: Neutron SLD of the spline material in pure D₂O, 10⁻⁶ Å⁻²
            (default 0).
        proton_exchange_efficiency: Fraction of labile protons that exchange
            with the solvent, unitless (0–1, default 1.0).
        sigma: Roughness applied at the spline boundaries, Å (default 5).
        nf: Overall number fraction scaling factor (default 1).  Can be
            linked to ``vf_bilayer`` of the base bilayer so the spline scales
            consistently with bilayer completeness.
        bulknsld: Solvent neutron SLD in 10⁻⁶ Å⁻² units (set automatically
            by :class:`~molgroups.refl1d_interface.MolgroupsLayer`).

    Reference Points:
        center_of_volume: Area-weighted centroid of the spline profile, Å.
            Updated after each render call.
        rho: H/D-aware nSLD of the spline material at the current bulk
            solvent SLD, 10⁻⁶ Å⁻².

    Example::

        from molgroups.refl1d_interface import Freeform
        from refl1d.names import Parameter

        N = 12
        SPACING = 15.0
        Dp = [Parameter(name=f'dDp{i}', value=0.0) for i in range(N)]
        Vf = [Parameter(name=f'dVf{i}', value=0.0).range(-0.001, 1.0)
              for i in range(N - 1)]
        Vf.append(Parameter(name=f'dVf{N-1}', value=0.0))

        spline = Freeform(name='spline',
                          dSpacing=SPACING,
                          startz=blm.outer_headgroup_top,
                          Dp=Dp,
                          Vf=Vf,
                          rhoH=rhoH_peptide,
                          rhoD=rhoD_peptide,
                          sigma=sigma,
                          nf=vf_bilayer)
    """

    _molgroup: mol.BoxHermite | None = None

    dSpacing: float = 15.0
    startz: Parameter = field(default_factory=lambda: Parameter(name='start position', value=20))
    Dp: List[Parameter] = field(default_factory=[])
    Vf: List[Parameter] = field(default_factory=[])
    rhoH: Parameter = field(default_factory=lambda: Parameter(name='rhoH', value=0.0))
    rhoD: Parameter = field(default_factory=lambda: Parameter(name='rhoD', value=0.0))
    proton_exchange_efficiency: Parameter = field(default_factory=lambda: Parameter(name='proton exchange efficiency', value=1.0))
    sigma: Parameter = field(default_factory=lambda: Parameter(name='roughness', value=5))

    center_of_volume: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='center of volume', description='center of volume'))
    rho: ReferencePoint = field(default_factory=lambda: ReferencePoint(name=f'nSLD', description='H/D aware nSLD of spline'))

    def __post_init__(self):
        self._molgroup = mol.BoxHermite(name=self.name, n_box=21)

         # protects against initial errors calculation self.rho
        self._molgroup.fnSetBulknSLD(0.0)

        self._group_names = {f'{self.name}': [f'{self.name}']}

        self.center_of_volume.set_function(self._center_of_volume)
        self.rho.set_function(functools.partial(lambda self: sld_from_bulk(self.rhoH.value, self.rhoD.value, self._molgroup.bulknsld * 1e6, self.proton_exchange_efficiency.value), self))

        super().__post_init__()

    def update(self) -> None:

        self._molgroup.fnSetBulknSLD(self.bulknsld.value * 1e-6)
        self._molgroup.fnSetRelative(dSpacing=self.dSpacing,
                                     dStart=self.startz.value,
                                     dDp=[d.value for d in self.Dp],
                                     dVf=[d.value for d in self.Vf],
                                     dnSLD=self.rho.value * 1e-6,
                                     dnf=self.nf.value,
                                     sigma=self.sigma.value)

# ============= Euler objects =================
@dataclass
class ContinuousEuler(MolgroupsInterface):
    """Rigid-body protein described by atomic coordinates and Euler rotations.

    Represents a protein (or other macromolecule) as a set of residues, each
    with a fixed volume and nSL determined from atomic composition.  The
    molecule is rotated by two Euler angles (``gamma`` in-plane, ``beta``
    tilt) about a user-specified rotation center, then translated to position
    ``z``.

    Residue data must be supplied as an 8-column array (or file) compatible
    with the ``mol.ContinuousEuler`` backend; typically generated from a PDB
    file using the molgroups pre-processing utilities.

    Attributes:
        residue_data: Array or filename with per-residue geometry (8 columns:
            x, y, z, volume, nSL_H, nSL_D, exchange_fraction, residue_index).
        rotcenter: [x, y, z] rotation center in the molecular frame, Å.
            If ``None``, the center of mass is used.
        gamma: In-plane rotation angle (azimuthal), degrees (default 0).
        beta: Out-of-plane tilt angle (polar), degrees (default 0).
        z: z-position of the rotation center after placement, Å (default 0).
            Use a ``ReferencePoint`` for chained positioning.
        sigma: Roughness applied to the protein envelope, Å (default 5).
        proton_exchange_efficiency: Fraction of labile protons that exchange
            with the solvent, unitless (0–1, default 1.0).
        nf: Number fraction (overall occupancy scaling, default 1).
        bulknsld: Solvent neutron SLD in 10⁻⁶ Å⁻² units (set automatically
            by :class:`~molgroups.refl1d_interface.MolgroupsLayer`).

    Reference Points:
        center_of_volume: Area-weighted centroid of the rendered protein
            profile, Å.  Updated after each render call.

    Example::

        import numpy as np
        from molgroups.refl1d_interface import ContinuousEuler

        residues = np.loadtxt('protein_residues.dat')

        protein = ContinuousEuler(
            name='protein',
            residue_data=residues,
            rotcenter=[0.0, 0.0, 0.0],
            gamma=Parameter(name='gamma', value=0).range(0, 360),
            beta=Parameter(name='beta',  value=0).range(0, 90),
            z=blm.bilayer_center,
            sigma=sigma,
            nf=Parameter(name='protein nf', value=0.5).range(0, 1))
    """

    _molgroup: mol.ContinuousEuler | None = None

    residue_data: list | np.ndarray = None
    rotcenter: list | np.ndarray = None
    gamma: Parameter = field(default_factory=lambda: Parameter(name='gamma rotation', value=0))
    beta: Parameter = field(default_factory=lambda: Parameter(name='beta rotation', value=0))
    z: Parameter = field(default_factory=lambda: Parameter(name='z position', value=0))
    sigma: Parameter = field(default_factory=lambda: Parameter(name='roughness', value=5))
    proton_exchange_efficiency: Parameter = field(default_factory=lambda: Parameter(name='proton exchange efficiency', value=1.0))

    center_of_volume: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='center of volume', description='center of volume'))

    def __post_init__(self):
        self._molgroup = mol.ContinuousEuler(name=self.name, fn8col=self.residue_data, rotcenter=self.rotcenter, xray=False)

         # protects against initial errors calculation self.rho
        self._molgroup.fnSetBulknSLD(0.0)

        self._group_names = {f'{self.name}': [f'{self.name}']}

        self.center_of_volume.set_function(self._center_of_volume)

        super().__post_init__()

    def update(self) -> None:

        self._molgroup.protexchratio = self.proton_exchange_efficiency.value
        self._molgroup.fnSet(gamma=self.gamma.value,
                             beta=self.beta.value,
                             zpos=self.z.value,
                             sigma=self.sigma.value,
                             nf=self.nf.value,
                             bulknsld=self.bulknsld.value * 1e-6)

# ============= Complex objects ===============

@dataclass
class BilayerProteinComplex(BaseGroupInterface):
    """Coupled bilayer + protein complex with automatic area normalization.

    Combines one or more bilayer objects (a mandatory base bilayer plus
    optional additional bilayers) with one or more protein/peptide groups
    into a single composite object.  After ``update()``, the protein area is
    subtracted from the bilayer area so that the total area never exceeds
    ``normarea`` — this is the physically correct treatment when a protein
    displaces lipids.

    The ``base_blm`` must be a :class:`SolidSupportedBilayer` or
    :class:`TetheredBilayer`; it determines ``normarea``.  Additional
    bilayers (e.g. a floating leaflet) are supplied via ``blms``.  Protein
    groups can be :class:`ComponentBox`, :class:`VolumeFractionBox`, or
    :class:`Freeform`.

    Attributes:
        base_blm: Primary bilayer (solid-supported or tethered) that sets the
            normalization area.  Its ``overlap`` is automatically shared with
            this complex.
        blms: Additional :class:`Bilayer` objects (e.g. floating leaflets)
            that are area-normalized along with the base bilayer.
        proteins: Protein / peptide group objects whose area is subtracted
            from the bilayer area at each z-point.
        normarea: In-plane normalization area in Å² (taken from ``base_blm``
            after each update).
        overlap: Substrate overlap, Å — automatically propagated to
            ``base_blm``.

    Example::

        from molgroups.refl1d_interface import (BilayerProteinComplex,
                                                SolidSupportedBilayer,
                                                VolumeFractionBox)

        blm = SolidSupportedBilayer(name='bilayer', ...)
        protein = VolumeFractionBox(name='protein',
                                    z=blm.bilayer_center, ...)

        complex_group = BilayerProteinComplex(
            name='bilayer+protein',
            overlap=overlap,
            base_blm=blm,
            proteins=[protein])

        mollayer = MolgroupsLayer(base_group=complex_group, ...)
    """

    _molgroup: mol.BLMProteinComplex | None = None

    base_blm: SolidSupportedBilayer | TetheredBilayer | None = None
    blms: List[Bilayer] = field(default_factory=list)
    proteins: List[ComponentBox | VolumeFractionBox | Freeform] = field(default_factory=list)

    def __post_init__(self) -> None:

        self._molgroup = mol.BLMProteinComplex(blms=[blm._molgroup for blm in self.all_blms],
                                           proteins=[prot._molgroup for prot in self.proteins])

        self._set_bulknsld(self.bulknsld)

        # compile group names based on
        # BLMProteinComplex.fnWriteGroup2Dict
        _group_names = {}
        for prepend, gplist in zip(['blms', 'proteins'], [self.all_blms, self.proteins]):
            prepend = f'{self.name}.{prepend}'
            for gp in gplist:
                for k, gpnames in gp._group_names.items():
                    gpnames = [f'{prepend}.{gpname}' for gpname in gpnames]
                    _group_names.update({k: gpnames})
        self._group_names = _group_names

        super().__post_init__()

        # tie base group overlap to this overlap, after conversion to a parameter
        self.base_blm.overlap = self.overlap

    def _get_parameters(self) -> Dict[str, Parameter]:

        pars = {}
        for gp in self.all_blms + self.proteins:
            pars.update(gp._get_parameters())
        return pars

    def _set_bulknsld(self, bulknsld):
        super()._set_bulknsld(bulknsld)
        for gp in self.all_blms + self.proteins:
            gp._set_bulknsld(bulknsld)

    @property
    def all_blms(self) -> List[Bilayer | SolidSupportedBilayer | TetheredBilayer]:
        return [self.base_blm] + self.blms if self.base_blm is not None else self.blms

    def update(self) -> None:

        for gp in self.all_blms + self.proteins:
            gp.update()

        self.normarea.value = self.base_blm.normarea.value
        self._molgroup.fnAdjustBLMs()

    def store_profile(self, z: np.ndarray) -> Dict:
        # special profile storage that takes into account excess density.
        # TODO: this is somewhat hackish. It might make more sense to have a subclass of MolgroupsLayer that
        # incorporates this logic
        super().store_profile(z)

        normarea = self.normarea.value

        prot_area = np.zeros_like(z)
        prot_nsl = np.zeros_like(z)
        for gp in self.proteins:
            _, area, nsl = gp.render(z)
            prot_area += area
            prot_nsl += nsl

        blm_area = np.zeros_like(z)
        blm_nsl = np.zeros_like(z)
        for gp in self.all_blms:
            _, area, nsl = gp.render(z)
            blm_area += area
            blm_nsl += nsl

        frac_replacement = np.ones_like(area)
        if len(self.proteins):
            over_filled = (blm_area + prot_area) > normarea
            frac_replacement[over_filled] = (blm_area / (normarea - prot_area))[over_filled]

        for blm in self.all_blms:
            for gplist in blm._group_names.values():
                for gp in gplist:
                    self._stored_profile[f'{self.name}.blms.{gp}']['area'] /= frac_replacement
                    self._stored_profile[f'{self.name}.blms.{gp}']['sl'] /= frac_replacement

        self._stored_profile['area'] = blm_area / frac_replacement + prot_area
        self._stored_profile['sl'] = blm_nsl / frac_replacement + prot_nsl
        self._stored_profile['normarea'] = normarea

# ============= Polymer objects ===============

@dataclass
class PolymerMushroom(MolgroupsInterface):
    """H/D-aware polymer mushroom density profile for low-density grafted polymers.

    Models a grafted polymer chain in the mushroom regime (grafting density
    much less than 1 chain per Rg²).  The volume-fraction profile is computed
    using the analytical self-consistent field result of Adamuţi-Trache,
    McMullen & Douglas, *J. Chem. Phys.* **105**, 4798 (1996), with the
    radius of gyration of the free chain used as the length scale.  The
    profile shape is determined by the radius of gyration and an interaction
    strength parameter (δ in the paper).

    Use :class:`PolymerBrush` for higher grafting densities where chain
    stretching is significant.

    Attributes:
        startz: z-position of the grafting surface, Å (default 0).  Use a
            ``ReferencePoint`` from a bilayer to chain positions.
        rhoH: Neutron SLD of the polymer in pure H₂O, 10⁻⁶ Å⁻² (default
            2.07).
        rhoD: Neutron SLD of the polymer in pure D₂O, 10⁻⁶ Å⁻² (default
            2.07).
        proton_exchange_efficiency: Fraction of labile protons that exchange
            with the solvent, unitless (0–1, default 1.0).
        grafting_density: Surface density of grafted chains, chains/Å²
            (default 0.1).
        radius_of_gyration: Radius of gyration of a free polymer chain in
            solution, Å (default 10).
        interaction_strength: Parameter controlling the self-avoidance
            (interaction strength δ), unitless (default 1).
        sigma: Roughness of the mushroom envelope, Å (default 4).
        normarea: Normalization area in Å² (typically shared with the base
            bilayer).
        nf: Overall number fraction scaling factor (default 1).
        bulknsld: Solvent neutron SLD in 10⁻⁶ Å⁻² units (set automatically
            by :class:`~molgroups.refl1d_interface.MolgroupsLayer`).

    Reference Points:
        rho: H/D-aware nSLD of the polymer at the current bulk solvent SLD,
            10⁻⁶ Å⁻².
        max_density: Peak volume fraction of the mushroom profile, unitless.
        max_position: z-position of the density maximum, Å.
        half_height_position: z-position where the density falls to half its
            maximum, Å.

    Example::

        from molgroups.refl1d_interface import PolymerMushroom
        from refl1d.names import Parameter

        mushroom = PolymerMushroom(
            name='PEG mushroom',
            startz=blm.outer_headgroup_top,
            rhoH=peg_rhoH,
            rhoD=peg_rhoD,
            grafting_density=Parameter(value=0.05).range(0.001, 0.2),
            radius_of_gyration=Parameter(value=15).range(5, 40),
            interaction_strength=Parameter(value=1).range(0.5, 3),
            normarea=blm.normarea)
    """

    _molgroup: mol.PolymerMushroom | None = None


    startz: Parameter = field(default_factory=lambda: Parameter(name='starting position', value=0))
    rhoH: Parameter = field(default_factory=lambda: Parameter(name='rho in H2O', value=2.07))
    rhoD: Parameter = field(default_factory=lambda: Parameter(name='rho in D2O', value=2.07))
    proton_exchange_efficiency: Parameter = field(default_factory=lambda: Parameter(name='proton exchange efficiency', value=1.0))
    grafting_density: Parameter = field(default_factory=lambda: Parameter(name='grafting density', value=0.1))
    radius_of_gyration: Parameter = field(default_factory=lambda: Parameter(name='radius of gyration', value=10))
    interaction_strength: Parameter = field(default_factory=lambda: Parameter(name='interaction strength', value=1))
    sigma: Parameter = field(default_factory=lambda: Parameter(name='roughness', value=4))
    normarea: Parameter = field(default_factory=lambda: Parameter(name='normarea', value=1))

    rho: ReferencePoint = field(default_factory=lambda: ReferencePoint(name=f'nSLD', description='H/D aware nSLD of spline'))
    max_density: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='max_density', description='maximum fractional density'))
    max_position: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='max_position', description='position of maximum density'))
    half_height_position: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='half_height_position', description='position of half density'))

    def __post_init__(self) -> None:
        self._molgroup = mol.PolymerMushroom(name=self.name)
        # protects against initial errors calculation self.rho
        self._molgroup.fnSetBulknSLD(0.0)

        self.rho.set_function(functools.partial(lambda self: sld_from_bulk(self.rhoH.value, self.rhoD.value, self._molgroup.bulknsld * 1e6, self.proton_exchange_efficiency.value), self))
        self.max_position.set_function(functools.partial(lambda gp: gp.fnGetMaxandHalfHeight()[0], self._molgroup))
        self.max_density.set_function(functools.partial(lambda gp: gp.fnGetMaxandHalfHeight()[2], self._molgroup))
        self.half_height_position.set_function(functools.partial(lambda gp: gp.fnGetMaxandHalfHeight()[1], self._molgroup))

        super().__post_init__()

    def update(self):

        self._molgroup.fnSetBulknSLD(self.bulknsld.value * 1e-6)
        self._molgroup.startz = self.startz.value
        self._molgroup.rho = self.rho.value * 1e-6
        self._molgroup.vf = self.grafting_density.value
        self._molgroup.Rg = self.radius_of_gyration.value
        self._molgroup.delta = self.interaction_strength.value
        self._molgroup.sigma = self.sigma.value
        self._molgroup.normarea = self.normarea.value
        self._molgroup.nf = self.nf.value

@dataclass
class PolymerBrush(MolgroupsInterface):
    """H/D-aware parabolic polymer brush density profile.

    Models a dense brush of grafted polymer chains using a parabolic
    density profile.  The profile has two regions: a flat base region of
    thickness ``base_length`` at volume fraction ``volume_fraction``, followed
    by a parabolic decay over ``interface_length`` Å controlled by
    ``thinning_power``.

    Use :class:`PolymerMushroom` for sparse grafting where chains behave as
    isolated mushrooms rather than a brush.

    Attributes:
        startz: z-position of the grafting surface, Å (default 0).  Use a
            ``ReferencePoint`` from a bilayer to chain positions.
        rhoH: Neutron SLD of the polymer in pure H₂O, 10⁻⁶ Å⁻² (default
            2.07).
        rhoD: Neutron SLD of the polymer in pure D₂O, 10⁻⁶ Å⁻² (default
            2.07).
        proton_exchange_efficiency: Fraction of labile protons that exchange
            with the solvent, unitless (0–1, default 1.0).
        volume_fraction: Volume fraction of the polymer in the flat base
            region, unitless (default 0.1).
        base_length: Thickness of the flat (constant volume fraction) region,
            Å (default 20).
        interface_length: Thickness of the parabolic decay region above the
            flat base, Å (default 20).
        thinning_power: Exponent of the parabolic decay; 1 gives linear,
            2 gives parabolic (default 1).
        sigma: Roughness applied at the brush boundaries, Å (default 4).
        normarea: Normalization area in Å² (typically shared with the base
            bilayer).
        nf: Overall number fraction scaling factor (default 1).
        bulknsld: Solvent neutron SLD in 10⁻⁶ Å⁻² units (set automatically
            by :class:`~molgroups.refl1d_interface.MolgroupsLayer`).

    Reference Points:
        rho: H/D-aware nSLD of the polymer at the current bulk solvent SLD,
            10⁻⁶ Å⁻².
        max_density: Peak volume fraction of the brush profile, unitless.
        half_height_position: z-position where the density falls to half its
            maximum, Å.

    Example::

        from molgroups.refl1d_interface import PolymerBrush
        from refl1d.names import Parameter

        brush = PolymerBrush(
            name='PEG brush',
            startz=blm.outer_headgroup_top,
            rhoH=peg_rhoH,
            rhoD=peg_rhoD,
            volume_fraction=Parameter(value=0.1).range(0.01, 0.5),
            base_length=Parameter(value=20).range(5, 60),
            interface_length=Parameter(value=20).range(5, 60),
            thinning_power=Parameter(value=1).range(0.5, 3),
            normarea=blm.normarea)
    """

    _molgroup: mol.PolymerBrush | None = None

    startz: Parameter = field(default_factory=lambda: Parameter(name='starting position', value=0))
    rhoH: Parameter = field(default_factory=lambda: Parameter(name='rho in H2O', value=2.07))
    rhoD: Parameter = field(default_factory=lambda: Parameter(name='rho in D2O', value=2.07))
    proton_exchange_efficiency: Parameter = field(default_factory=lambda: Parameter(name='proton exchange efficiency', value=1.0))
    volume_fraction: Parameter = field(default_factory=lambda: Parameter(name='volume fraction', value=0.1))
    base_length: Parameter = field(default_factory=lambda: Parameter(name='length of base region', value=20))
    interface_length: Parameter = field(default_factory=lambda: Parameter(name='length of interface region', value=20))
    thinning_power: Parameter = field(default_factory=lambda: Parameter(name='thinning power', value=1))
    sigma: Parameter = field(default_factory=lambda: Parameter(name='roughness', value=4))
    normarea: Parameter = field(default_factory=lambda: Parameter(name='normarea', value=1))

    rho: ReferencePoint = field(default_factory=lambda: ReferencePoint(name=f'nSLD', description='H/D aware nSLD of spline'))
    max_density: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='max_density', description='maximum fractional density'))
    half_height_position: ReferencePoint = field(default_factory=lambda: ReferencePoint(name='half_height_position', description='position of half density'))

    def __post_init__(self) -> None:
        self._molgroup = mol.PolymerBrush(name=self.name)

        # protects against initial errors calculation self.rho
        self._molgroup.fnSetBulknSLD(self.bulknsld.value)
        self.rho.set_function(functools.partial(lambda self: sld_from_bulk(self.rhoH.value, self.rhoD.value, self._molgroup.bulknsld * 1e6, self.proton_exchange_efficiency.value), self))
        self.max_density.set_function(functools.partial(lambda gp: gp.fnGetMaxandHalfHeight()[2], self._molgroup))
        self.half_height_position.set_function(functools.partial(lambda gp: gp.fnGetMaxandHalfHeight()[1], self._molgroup))

        super().__post_init__()

    def update(self):

        self._molgroup.fnSetBulknSLD(self.bulknsld.value * 1e-6)
        self._molgroup.startz = self.startz.value
        self._molgroup.rho = self.rho.value * 1e-6
        self._molgroup.vf = self.volume_fraction.value
        self._molgroup.base_length = self.base_length.value
        self._molgroup.interface_length = self.interface_length.value
        self._molgroup.thinning_power = self.thinning_power.value
        self._molgroup.sigma = self.sigma.value
        self._molgroup.normarea = self.normarea.value
        self._molgroup.nf = self.nf.value
