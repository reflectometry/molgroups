"""Assembly of molgroups interface objects into Refl1D layer stacks.

Conceptual model
----------------
A molgroups model occupies a single "canvas" — a slab of thickness
``MolgroupsLayer.thickness`` that sits between the solid substrate layers and
the bulk solvent in the Refl1D layer stack.  Within the canvas, molecular
groups are rendered to area and nSL profiles, summed, and converted to an nSLD
profile that Refl1D integrates via its microslabbing engine.

The canvas has three kinds of group slots:

``base_group``
    The single group that anchors the canvas.  It must be a
    :class:`~molgroups.refl1d_interface.BaseGroupInterface` (i.e.
    :class:`~molgroups.refl1d_interface.SolidSupportedBilayer`,
    :class:`~molgroups.refl1d_interface.TetheredBilayer`, or
    :class:`~molgroups.refl1d_interface.BilayerProteinComplex`).  Its
    ``normarea`` value sets the in-plane normalization area for all other
    groups, and its ``overlap`` value determines how far it extends back into
    the substrate (see *The overlap contract* below).

``add_groups``
    Groups whose area is simply *added* to the base group.  Use these for
    objects that coexist with the bilayer in the same z-region without
    displacing lipids — for example a protein box positioned above the outer
    headgroups, or a floating bilayer overlayer.

``overlay_groups``
    Groups that *displace* material from the base and add groups.  Wherever
    the sum of overlay area and base/add area would exceed ``normarea``, the
    base/add profiles are scaled down proportionally so the total area stays
    at ``normarea``.  Use these for membrane-embedded objects such as a
    freeform spline representing a peptide inserted into the bilayer.

The nSLD at each z-point is::

    nSLD(z) = nsl(z) / (normarea * dz) + (1 - area(z)/normarea) * contrast.rho

so the unfilled fraction is automatically filled with bulk solvent at the
contrast SLD.

---

The overlap contract
--------------------
The ``base_group.overlap`` parameter controls how far the base group's
substrate box extends into the *positive-z* half of the canvas.  The same
distance must be *subtracted* from the thickness of the last substrate ``Slab``
in the Refl1D layer stack, so that the substrate material is not double-counted::

    overlap = 30.0

    # Substrate slabs — last one shortened by overlap
    layer_silicon = Slab(material=silicon, thickness=0.0,    interface=rough)
    layer_siox    = Slab(material=siox,    thickness=d_siox, interface=rough)
    layer_tiox    = Slab(material=tiox,    thickness=d_tiox - overlap, interface=0.0)

    substrate = layer_silicon | layer_siox | layer_tiox

    blm = SolidSupportedBilayer(name='bilayer', overlap=overlap, ...)

If you forget to subtract ``overlap`` from the last substrate slab the
substrate SLD will be double-counted over the overlap region.

---

Multi-contrast fitting
----------------------
Each contrast (H₂O, D₂O, …) requires its own :class:`MolgroupsLayer` and
:class:`MolgroupsStack`, because the bulk SLD and the per-group ``bulknsld``
values differ between contrasts.  Structural parameters should be defined once
as shared :class:`~refl1d.names.Parameter` objects and passed into each
contrast copy.

The recommended pattern is a **factory function** that accepts ``substrate``
and ``contrast`` as required arguments, plus any parameters that vary between
contrasts (e.g. a contrast-specific background or an isotope-labelled
component) as keyword arguments.  Parameters that are the same across contrasts
are captured from the enclosing scope::

    from refl1d.names import Parameter, SLD, Slab, FitProblem, load4
    from molgroups.refl1d_interface import (SolidSupportedBilayer,
                                            VolumeFractionBox,
                                            MolgroupsLayer,
                                            MolgroupsStack,
                                            MolgroupsExperiment)

    # --- Shared structural parameters (defined once, used in all contrasts) ---
    sigma      = Parameter(name='bilayer roughness',  value=5).range(0.5, 9)
    l_lipid1   = Parameter(name='inner acyl chain',   value=10).range(8, 30)
    l_lipid2   = Parameter(name='outer acyl chain',   value=10).range(8, 18)
    vf_bilayer = Parameter(name='volume fraction',    value=0.9).range(0, 1)
    l_protein  = Parameter(name='protein length',     value=30).range(10, 60)
    vf_protein = Parameter(name='protein vf',         value=0.5).range(0, 1)
    overlap    = 30.0
    thickness  = 200.0

    # --- Contrast SLDs ---
    d2o = SLD(name='d2o', rho=6.36)
    h2o = SLD(name='h2o', rho=-0.56)

    # --- Substrate (shared across contrasts) ---
    layer_silicon = Slab(material=silicon, thickness=0.0,          interface=rough)
    layer_tiox    = Slab(material=tiox,    thickness=110 - overlap, interface=0.0)
    substrate     = layer_silicon | layer_tiox

    # --- Protein SLDs from sequence (contrast-independent) ---
    from periodictable.fasta import Sequence
    protein = Sequence(name='protein', sequence='ACDEFGHIKLM', type='aa')
    rhoH_protein = protein.D2Osld(1, 0)   # nSLD in pure H2O, 10^-6 A^-2
    rhoD_protein = protein.D2Osld(1, 1)   # nSLD in pure D2O, 10^-6 A^-2

    # --- Factory function ---
    def make_sample(substrate, contrast):
        \"\"\"Build one MolgroupsStack for a given contrast.

        Structural parameters (sigma, l_lipid1, rhoH_protein, …) are captured
        from the enclosing scope and shared across all contrasts.  Only the
        SLD object and the substrate slab stack differ between calls.
        \"\"\"
        blm = SolidSupportedBilayer(
            name='bilayer',
            overlap=overlap,
            lipids=[DOPC],
            inner_lipid_nf=[1.0],
            outer_lipid_nf=[1.0],
            rho_substrate=tiox.rho,
            vf_bilayer=vf_bilayer,
            l_lipid1=l_lipid1,
            l_lipid2=l_lipid2,
            sigma=sigma)

        protein_box = VolumeFractionBox(
            name='protein',
            z=blm.outer_headgroup_top + 0.5 * l_protein,
            rhoH=rhoH_protein,
            rhoD=rhoD_protein,
            volume_fraction=vf_protein,
            length=l_protein,
            sigma_bottom=sigma,
            sigma_top=sigma)

        mollayer = MolgroupsLayer(
            base_group=blm,
            add_groups=[protein_box],
            thickness=thickness,
            contrast=contrast,
            name='bilayer layer ' + contrast.name)

        return MolgroupsStack(substrate=substrate,
                              molgroups_layer=mollayer)

    # --- Build one stack per contrast ---
    sample_d2o = make_sample(substrate, d2o)
    sample_h2o = make_sample(substrate, h2o)

    # --- Experiments and fit problem ---
    # Use MolgroupsExperiment (not Experiment) to enable molgroups-specific
    # functionality: volume-fraction plots, derived parameter tables, and
    # uncertainty analysis of group positions.
    model_d2o = MolgroupsExperiment(sample=sample_d2o, probe=probe_d2o, dz=0.5)
    model_h2o = MolgroupsExperiment(sample=sample_h2o, probe=probe_h2o, dz=0.5)
    problem = FitProblem([model_d2o, model_h2o])

Because each call to ``make_sample`` constructs new group objects that share
the same ``Parameter`` instances, bumps treats all contrasts as a single
jointly-constrained fit problem.
"""

from typing import List
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import trapezoid

from refl1d.names import Parameter, SLD, Stack, Slab
from refl1d.sample.layers import Layer

from .groups import BaseGroupInterface, MolgroupsInterface

@dataclass(init=False)
class MolgroupsLayer(Layer):
    """A single functional layer that renders a set of molgroups groups to an
    nSLD profile for Refl1D.

    ``MolgroupsLayer`` sits between the solid substrate slabs and the bulk
    solvent slab in the Refl1D layer stack.  On each reflectivity evaluation it:

    1. Calls ``update()`` on all groups (propagates current ``Parameter``
       values into the underlying mol objects).
    2. Renders area and nSL profiles for every group on a z-grid.
    3. Combines them according to the base / add / overlay rules (see module
       docstring).
    4. Fills the remaining volume fraction with bulk solvent at
       ``contrast.rho``.
    5. Returns the nSLD profile to Refl1D's microslabbing engine.

    ``MolgroupsLayer`` is not used directly as a Refl1D ``Slab``; wrap it in a
    :class:`MolgroupsStack` so that the contrast slab is appended
    automatically.

    Attributes:
        base_group: The anchoring molecular group.  Must be a
            :class:`~molgroups.refl1d_interface.BaseGroupInterface` (i.e.
            :class:`~molgroups.refl1d_interface.SolidSupportedBilayer`,
            :class:`~molgroups.refl1d_interface.TetheredBilayer`, or
            :class:`~molgroups.refl1d_interface.BilayerProteinComplex`).
            Determines ``normarea`` for all other groups.
        normarea_group: Optional override for the normarea source.  When set,
            this group's ``normarea`` is used instead of ``base_group``'s, and
            ``base_group`` is rescaled accordingly.  Rarely needed; leave
            ``None`` for standard bilayer models.
        add_groups: Groups whose area is added to the base group without
            displacement.  Suitable for objects coexisting with the bilayer in
            the same z-region (protein boxes above the headgroups, floating
            overlayer bilayers, etc.).
        overlay_groups: Groups that displace base and add material.  Wherever
            their area would cause the total to exceed ``normarea``, the
            base/add profiles are scaled down proportionally.  Use for
            membrane-embedded objects (e.g. a :class:`~molgroups.refl1d_interface.Freeform`
            spline representing an inserted peptide).
        contrast: An :class:`~refl1d.names.SLD` object whose ``rho`` parameter
            is the bulk solvent nSLD in 10⁻⁶ Å⁻².  Setting this wires all
            groups' ``bulknsld`` to ``contrast.rho`` so that H/D-aware SLD
            calculations update automatically.  One ``MolgroupsLayer`` per
            contrast is required for multi-contrast fitting.
        thickness: Thickness of the molgroups canvas, Å.  Should be large
            enough to contain all groups with some margin.  Does not need to
            be fitted.
        name: Layer name used in Refl1D parameter trees.

    Example::

        from molgroups.refl1d_interface import (SolidSupportedBilayer,
                                                Freeform,
                                                MolgroupsLayer,
                                                MolgroupsStack)
        from refl1d.names import SLD

        overlap   = 30.0
        thickness = 200.0
        d2o = SLD(name='d2o', rho=6.36)

        blm    = SolidSupportedBilayer(name='bilayer', overlap=overlap, ...)
        spline = Freeform(name='spline', startz=blm.outer_headgroup_top, ...)

        mollayer = MolgroupsLayer(
            base_group=blm,
            overlay_groups=[spline],   # spline displaces bilayer lipids
            thickness=thickness,
            contrast=d2o,
            name='bilayer layer d2o')

        sample = MolgroupsStack(substrate=substrate, molgroups_layer=mollayer)

    See the module docstring for the multi-contrast factory-function pattern.
    """

    base_group: BaseGroupInterface
    normarea_group: MolgroupsInterface | None = None
    add_groups: List[MolgroupsInterface] = field(default_factory=list)
    overlay_groups: List[MolgroupsInterface] = field(default_factory=list)
    contrast: SLD | None=None
    thickness: float | Parameter = 0.0
    name: str | None = None

    def __init__(self,
                 base_group: BaseGroupInterface,
                 normarea_group: MolgroupsInterface | None = None,
                 add_groups: List[MolgroupsInterface] = [],
                 overlay_groups: List[MolgroupsInterface] = [],
                 contrast: SLD | None=None,
                 thickness: float | Parameter = 0.0,
                 name=None) -> None:

        if not isinstance(base_group, BaseGroupInterface):
            raise TypeError(f'Base group {base_group} must be an instance of BaseGroupInterface')

        self.base_group = base_group
        self.normarea_group = normarea_group
        self.add_groups = add_groups
        self.overlay_groups = overlay_groups

        if name is None:
            name = self.base_group.name

        self.thickness = Parameter.default(thickness, name=name+" thickness")
        self.interface = Parameter.default(0.0, name=name+" interface")

        self.name = name
        self.magnetism = None
        self.contrast = contrast

        if self.contrast is not None:
            for gp in [self.base_group] + self.add_groups + self.overlay_groups:
                gp._set_bulknsld(self.contrast.rho)

        self._penalty = 0.0

    def update(self):

        normarea: float | None = None

        # 0. update normarea_group

        # 1. update base_group (may be required twice if normarea_group is defined differently)
        self.base_group.update()
        if self.normarea_group is None:
            normarea = self.base_group.normarea.value
        else:
            self.normarea_group.update()
            if not hasattr(self.normarea_group, 'normarea'):
                print(f'Warning: {self.normarea_group.name} does not have normarea and cannot be normarea_group. Ignoring.')
            else:
                normarea = self.normarea_group.normarea.value
                if hasattr(self.base_group._molgroup, 'fnSetNormarea') & (self.base_group != self.normarea_group):
                    self.base_group._molgroup.fnSetNormarea(normarea)
                self.base_group.normarea.value = normarea
                self.base_group.update()

        # 2. apply normarea to all remaining objects
        for group in self.add_groups + self.overlay_groups:
            if hasattr(group._molgroup, 'fnSetNormarea') & (group != self.normarea_group):
                group._molgroup.fnSetNormarea(normarea)

        # 3. update all remaining objects
        for group in self.add_groups + self.overlay_groups:
            if group != self.normarea_group:
                group.update()

    def profile(self, z):

        # 1. Write base group
        normarea, area, nsl = self.base_group.render(z)

        # 2. Add up all add_groups
        for group in self.add_groups:
            _, newarea, newnsl = group.render(z)
            area += newarea
            nsl += newnsl

        # 3. Add up all overlay_groups
        overlay_area = np.zeros_like(area)
        overlay_nsl = np.zeros_like(nsl)
        for group in self.overlay_groups:
            _, newarea, newnsl = group.render(z)
            overlay_area += newarea
            overlay_nsl += newnsl

        # 4. Perform overlay
        # 4a. Calculate fraction of base_group + add_groups left after replacement
        frac_replacement = np.ones_like(area)
        if len(self.overlay_groups):
            over_filled = (area + overlay_area) > normarea
            frac_replacement[over_filled] = (area / (normarea - overlay_area))[over_filled]

        # 4b. Scale area, nsl by replacement fraction and add
        # in overlay_area, overlay_nsl
        area /= frac_replacement
        nsl /= frac_replacement
        area += overlay_area
        nsl += overlay_nsl

        # 4c. Scale stored profiles by frac_replacement
        for group in [self.base_group] + self.add_groups:
            group._stored_profile['frac_replacement'] = frac_replacement

        return normarea, area, nsl

    def parameters(self):

        # TODO: figure out how to get all the unique parameters from the sub-objects
        # This will probably break if there are overlapping parameters

        return {k : p
                for group in [self.base_group] + self.add_groups + self.overlay_groups
                    for k, p in group._get_parameters().items()}


    def penalty(self) -> float:

        return self._penalty

    def _filled_profile(self, z):
        """Given area and nSL profiles, fill in the remaining volume with bulk material"""

        self.update()
        normarea, area, nsl = self.profile(z)

        # calculate penalty due to overfilling anywhere
        over_filled = area > normarea
        self._penalty = trapezoid((area - normarea)[over_filled], z[over_filled])

        # Fill in the remaining volume with buffer of appropriate nSLD
        nsld = 1e6 * nsl / (normarea * np.gradient(z)) + (1.0 - area / normarea) * self.contrast.rho.value

        # Return nSLD profile in Refl1D units
        return nsld

    def render(self, probe, slabs) -> None:
        """Adapted from refl1d.sample.flayer.FunctionalProfile
        """
        Pw, Pz = slabs.microslabs(self.thickness)
        if len(Pw) == 0:
            return

        # add molgroups layer slabs
        slabs.extend(rho=[self._filled_profile(Pz)], irho=[np.zeros_like(Pz)], w=Pw)

        # add buffer slab (better done at sample stack level)
        #slabs.append(rho=self.contrast.rho.value, irho=0.0, w=0.0, sigma=0.0)

    def _get_moldat(self) -> dict:
        """Gets moldat object for plotting and statistical analysis

        Returns:
            dict: moldat
        """

        return {group.name: group._stored_profile
                    for group in [self.base_group] + self.add_groups + self.overlay_groups}

# =============

@dataclass(init=False, eq=False, match_args=False)
class MolgroupsStack(Stack):
    """A complete Refl1D sample stack combining substrate slabs with a
    molgroups functional layer.

    ``MolgroupsStack`` assembles the final layer sequence::

        substrate slabs | MolgroupsLayer | contrast slab (zero thickness)

    The trailing zero-thickness contrast slab ensures that Refl1D's
    microslabbing engine fills the semi-infinite bulk region with the correct
    solvent SLD.

    ``MolgroupsStack`` must be used with
    :class:`~molgroups.refl1d_interface.MolgroupsExperiment` rather than the
    standard Refl1D ``Experiment``.  ``MolgroupsExperiment`` adds
    molgroups-specific functionality: volume-fraction and nSLD profile plots,
    derived parameter tables, and uncertainty analysis of group positions and
    reference points.

    Attributes:
        substrate: A Refl1D ``Slab`` or ``Stack`` of slabs representing the
            solid support (silicon, oxide layers, metal films, etc.).  The
            *last* slab in ``substrate`` must have its thickness reduced by
            ``molgroups_layer.base_group.overlap`` so that the substrate
            material is not double-counted in the overlap region (see *The
            overlap contract* in the module docstring).
        molgroups_layer: The :class:`MolgroupsLayer` that defines the
            molecular structure.
        name: Name used for this stack in Refl1D plots and parameter trees.
            Pass ``name=mollayer.name`` to propagate the layer name; if
            omitted the default ``"MolgroupsStack"`` is used for all
            contrasts, which causes plots to be labelled incorrectly.

    Example::

        from refl1d.names import SLD, Slab, Parameter

        # Materials
        silicon = SLD(name='silicon', rho=2.069)
        tiox    = SLD(name='tiox',    rho=2.163)
        d2o     = SLD(name='d2o',     rho=6.36)

        overlap  = 30.0
        d_tiox   = Parameter(name='TiOx thickness', value=110).range(100, 200)
        rough    = Parameter(name='substrate roughness', value=5).range(2, 9)

        # Substrate — last slab shortened by overlap
        layer_silicon = Slab(material=silicon, thickness=0.0,            interface=rough)
        layer_tiox    = Slab(material=tiox,    thickness=d_tiox - overlap, interface=0.0)
        substrate     = layer_silicon | layer_tiox

        # Molecular layer
        blm      = SolidSupportedBilayer(name='bilayer', overlap=overlap, ...)
        mollayer = MolgroupsLayer(base_group=blm, thickness=200.0, contrast=d2o)

        sample = MolgroupsStack(substrate=substrate, molgroups_layer=mollayer)

        # Experiment
        model = MolgroupsExperiment(sample=sample, probe=probe_d2o, dz=0.5)
    """

    substrate: Stack | Slab
    molgroups_layer: MolgroupsLayer

    def __init__(self,
                 substrate: Stack | Slab,
                 molgroups_layer: MolgroupsLayer,
                 name="MolgroupsStack",
                 **kw):
        layer_contrast = Slab(material=molgroups_layer.contrast, thickness=0.0000, interface=0.0000)
        #if isinstance(substrate, Stack):
        #    substrate.layers[-1].thickness = substrate.layers[-1].thickness - molgroups_layer.base_group.overlap
        #elif isinstance(substrate, Slab):
        #    substrate.thickness = substrate.thickness - molgroups_layer.overlap
        layers = substrate | molgroups_layer | layer_contrast
        super().__init__(layers=layers,
                         name=name,
                         interface=None,
                         thickness=None)

        self.substrate, self.molgroups_layer = substrate, molgroups_layer

    def __post_init__(self):
        self.molgroups_layer.update()
