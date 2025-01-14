"""Defines a Refl1D Experiment object with molgroups support"""

from typing import List, Optional
from dataclasses import dataclass
import functools

from refl1d.names import Experiment, Stack, SLD, MixedExperiment

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
        self.parts = [MolgroupsExperiment(s, probe, name=s.name, **kw) for s in samples]
        for i, p in enumerate(self.parts):
            for key, item in p._webview_plots.items():
                self._webview_plots.update({f'{i}: {key}': item})

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
        mollayer = MolgroupsLayer(base_group=layer_template.base_group,
                                  normarea_group=layer_template.normarea_group,
                                  add_groups=layer_template.add_groups,
                                  overlay_groups=layer_template.overlay_groups,
                                  contrast=contrast,
                                  thickness=layer_template.thickness,
                                  name=contrast.name + ' ' + layer_template.name)
        samples.append(MolgroupsStack(substrate=substrate,
                                      molgroups_layer=mollayer,
                                      name=mollayer.name))

    return samples