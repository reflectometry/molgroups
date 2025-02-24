"""Defines a Refl1D Experiment object with molgroups support"""

from typing import List, Optional
from dataclasses import dataclass, fields
import copy
import functools

from refl1d.names import Experiment, Stack, SLD, MixedExperiment, Parameter
from bumps.parameter import Expression

from .groups import MolgroupsInterface, ReferencePoint
from .layers import MolgroupsStack, MolgroupsLayer
from .plots import cvo_plot, cvo_uncertainty_plot, results_table

@dataclass(init=False)
class MolgroupsExperiment(Experiment):

    sample: MolgroupsStack | None = None,

    def __init__(self,
                 sample: MolgroupsStack | None = None,
                 probe=None,
                 name=None,
                 roughness_limit=0,
                 dz=None,
                 dA=None,
                 step_interfaces=None,
                 smoothness=None,
                 interpolation=0,
                 constraints=None,
                 version: str | None = None,
                 auto_tag=False):
        super().__init__(sample, probe, name, roughness_limit, dz, dA, step_interfaces, smoothness, interpolation, constraints, version, auto_tag)
        self.register_webview_plot(plot_title='Component Volume Occupancy',
                                   plot_function=functools.partial(cvo_plot, self.sample.molgroups_layer),
                                   change_with='parameter')
        self.register_webview_plot(plot_title='Component Volume Occupancy Uncertainty',
                                   plot_function=functools.partial(cvo_uncertainty_plot, self.sample.molgroups_layer),
                                   change_with='uncertainty')
        self.register_webview_plot(plot_title='Statistics table',
                                   plot_function=functools.partial(results_table, self.sample.molgroups_layer, report_delta=False),
                                   change_with='uncertainty')        
        self.register_webview_plot(plot_title='Difference statistics table',
                                   plot_function=functools.partial(results_table, self.sample.molgroups_layer, report_delta=True),
                                   change_with='uncertainty')        

@dataclass(init=False)
class MolgroupsMixedExperiment(MixedExperiment):
    samples: Optional[List[MolgroupsStack]]

    def __init__(self, samples: Optional[List[MolgroupsStack | Stack]]=None,
                 ratio=None,
                 probe=None,
                 name=None,
                 coherent=False,
                 interpolation=0,
                 **kw):
        super().__init__(samples, ratio, probe, name, coherent, interpolation, **kw)
        for i, (p, s) in enumerate(zip(self.parts, self.samples)):
            
            # if MolgroupsStack samples, use MolgroupsExperiments
            if isinstance(s, MolgroupsStack):
                p = MolgroupsExperiment(s, probe, name=s.name, **kw)

            # experiment inherits registered webview plots
            for key, item in p._webview_plots.items():
                self._webview_plots.update({f'{i}: {key}': item})

def _copy_group(gp: MolgroupsInterface | None) -> MolgroupsInterface | None:
    if gp is not None:
        new_gp = copy.copy(gp)

        # reset ID
        new_gp._generate_id()

        # reset ReferencePoints
        for f in fields(new_gp):
            if f.type == ReferencePoint:
                setattr(new_gp, f.name, f.default_factory())

        # run post init
        new_gp.__post_init__()

        return new_gp
    

def _copy_template(layer_template: MolgroupsLayer, contrast: SLD) -> MolgroupsLayer:
    """Copies a template and associates a contrast with it. This is a
        complex procedure because there can
        be logic linking the ReferencePoint objects in each group to
        parameters via Expression objects. These need to be reconstructed
        in the copy.
    
    Args:
        layer_template (MolgroupsLayer): template layer

    Returns:
        MolgroupsLayer: copied template
    """

    # TODO: this does not work for deep groups like BLMProteinComplex

    # find normarea group if it exists
    normarea_id = layer_template.normarea_group.id if layer_template.normarea_group is not None else None

    # find which group is the normarea group
    if normarea_id is not None:
        all_groups = [layer_template.base_group] + layer_template.add_groups + layer_template.overlay_groups
        normarea_index = [gp.id for gp in all_groups].index(normarea_id)

    # create a dictionary of which attribute corresponds to which reference point
    all_groups = [layer_template.base_group] + layer_template.add_groups + layer_template.overlay_groups
    new_base_group = _copy_group(layer_template.base_group)
    new_add_groups = [_copy_group(gp) for gp in layer_template.add_groups]
    new_overlay_groups = [_copy_group(gp) for gp in layer_template.overlay_groups]
    new_groups = [new_base_group] + new_add_groups + new_overlay_groups

    references = {}

    # recursive function to build up references list
    def _build_references(gps: List[MolgroupsInterface]):
        for i, gp in enumerate(gps):
            for f in fields(gp):
                attr = getattr(gp, f.name)
                if f.type == ReferencePoint:
                    rp: ReferencePoint = getattr(gp, f.name)
                    references[rp.id] = dict(index=i,
                                            group_id=gp.id,
                                            field_name=f.name)
                    
                elif isinstance(attr, MolgroupsInterface):
                    _build_references([attr])

    # create recursive function to replace all Expressions with copies with unique IDs and the appropriate references
    def _replace_references(exp: Expression):
        new_exp = copy.copy(exp)
        new_exp_args = list(new_exp.args)
        for i, arg in enumerate(new_exp_args):
            if isinstance(arg, ReferencePoint):
                ref = references[arg.id]
                new_exp_args[i] = getattr(new_groups[ref['index']], ref['field_name'])
            elif isinstance(arg, Expression):
                new_exp_args[i] = _replace_references(arg)

        new_exp.args = new_exp_args

        return new_exp

    for gp in new_groups:
        for f in fields(gp):
            attr = getattr(gp, f.name)
            if isinstance(attr, Expression):
                setattr(gp, f.name, _replace_references(attr))
                
    return MolgroupsLayer(base_group=new_base_group,
                              normarea_group=new_groups[normarea_index] if normarea_id is not None else None,
                            add_groups=new_add_groups,
                            overlay_groups=new_overlay_groups,
                            contrast=contrast,
                            thickness=layer_template.thickness,
                            name=contrast.name + ' ' + layer_template.name)
    
def make_samples(layer_template: MolgroupsLayer, substrate: Stack, contrasts: List[SLD]) -> List[MolgroupsStack]:

    """Create samples from combining a substrate stack with a molgroups layer
    
        Args:
            layer_template: molgroups layer template
            substrate: Refl1D Stack or Layer object representing the substrate
            contrasts: list of buffer materials, e.g. [d2o, h2o]. One sample will be created for each contrast

        Returns:
            List[MolgroupsStack]: list of samples (MolgroupsStack), one for each contrast
    """
    samples = []

    for contrast in contrasts:
        mollayer = _copy_template(layer_template, contrast)
        samples.append(MolgroupsStack(substrate=substrate,
                                      molgroups_layer=mollayer,
                                      name=mollayer.name))

    return samples