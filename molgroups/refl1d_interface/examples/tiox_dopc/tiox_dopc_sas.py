"""Example script using molgroups Refl1D interface objects"""

import numpy as np
from refl1d.names import Parameter, SLD, Slab, FitProblem, load4
from refl1d.probe.resolution import divergence
from molgroups import components as cmp
from molgroups.refl1d_interface import (SolidSupportedBilayer,
                                        MolgroupsLayer,
                                        MolgroupsStack,
                                        MolgroupsExperiment)

from molgroups.refl1d_interface import SASReflectivityMolgroupsExperiment, SASReflectivityModel

## === Probes/data files ===
probe_d2o = load4('ch061_d2o_ph7.refl', back_reflectivity=True, name='D2O')
probe_h2o = load4('ch060_h2o_ph7.refl', back_reflectivity=True, name='H2O')

# Probe parameters
probes = [probe_d2o, probe_h2o]

# Probe parameters
intensity = Parameter(name='intensity', value=0.8).range(0.65, 1.0)
sample_broadening = Parameter(name='sample broadening', value=0.0).range(-0.003, 0.02)
theta_offset = Parameter(name='theta offset', value=0.0).range(-0.02, 0.02)

# apply background and intensity to all probes
for probe in probes:
    probe.background.limits = (-np.inf, np.inf)
    probe.background.range(-1e-6, 1e-5)
    probe.intensity = intensity

    # if probes support these
    probe.sample_broadening = sample_broadening
    probe.theta_offset = theta_offset

## === Structural parameters ===

vf_bilayer = Parameter(name='volume fraction bilayer', value=0.9).range(0.0, 1.0)
l_lipid1 = Parameter(name='inner acyl chain thickness', value=10.0).range(8, 30)
l_lipid2 = Parameter(name='outer acyl chain thickness', value=10.0).range(8, 18)
l_submembrane = Parameter(name='submembrane thickness', value=10.0).range(0, 50)
sigma = Parameter(name='bilayer roughness', value=5).range(0.5, 9)
global_rough = Parameter(name ='substrate roughness', value=5).range(2, 9)
tiox_rough = Parameter(name='titanium oxide roughness', value=4).range(2, 9)
d_oxide = Parameter(name='silicon oxide layer thickness', value=10).range(5, 30)
d_tiox =  Parameter(name='titanium oxide layer thickness', value=110).range(100, 200)

## === Materials ===

# Material definitions
d2o = SLD(name='d2o', rho=6.3000, irho=0.0000)
h2o = SLD(name='h2o', rho=-0.56, irho=0.0000)
tiox = SLD(name='tiox', rho=2.1630, irho=0.0000)
siox = SLD(name='siox', rho=4.1000, irho=0.0000)
silicon = SLD(name='silicon', rho=2.0690, irho=0.0000)

# Material SLD parameters
d2o.rho.range(5.3000, 6.36)
h2o.rho.range(-0.56, 0.6)
tiox.rho.range(1.2, 3.2)
siox.rho.range(2.8, 4.8)

## === Molecular groups ===

# overlap between substrate and molgroups layer
overlap = 30.0

# thickness of molgroups layer
thickness = 200.0

# define lipids and number fractions
DOPC = cmp.Lipid(name='DOPC', headgroup=cmp.pc, tails=2 * [cmp.oleoyl], methyls=[cmp.methyl])
lipidlist = [DOPC]
lipid_nf = [1.0]

def bilayer(substrate, contrast):

    blm = SolidSupportedBilayer(name='bilayer',
                        overlap=overlap,
                        lipids=lipidlist,
                        inner_lipid_nf=lipid_nf,
                        outer_lipid_nf=lipid_nf,
                        rho_substrate=tiox.rho,
                        l_siox=0.0,
                        vf_bilayer=vf_bilayer,
                        l_lipid1=l_lipid1,
                        l_lipid2=l_lipid2,
                        l_submembrane=l_submembrane,
                        substrate_rough=tiox_rough,
                        sigma=sigma)
    
    mollayer = MolgroupsLayer(base_group=blm,
                            thickness=thickness,
                            contrast=contrast,
                            name='bilayer layer ' + contrast.name)
    
    return MolgroupsStack(substrate=substrate,
                        molgroups_layer=mollayer,
                        name=mollayer.name)

## == Sample layer stack ==

layer_silicon = Slab(material=silicon, thickness=0.0000, interface=global_rough)
layer_siox = Slab(material=siox, thickness=d_oxide, interface=global_rough)
layer_tiox = Slab(material=tiox, thickness=d_tiox - overlap, interface=0.00)

substrate = layer_silicon | layer_siox | layer_tiox

# Use the bilayer definition function to generate the bilayer SLD profile, passing in the relevant parameters.
sample_d2o, sample_h2o = [bilayer(substrate, contrast) for contrast in [d2o, h2o]]

## === Problem definition ===
## step = True corresponds to a calculation of the reflectivity from an actual profile
## with microslabbed interfaces.  When step = False, the Nevot-Croce
## approximation is used to account for roughness.  This approximation speeds up
## the calculation tremendously, and is reasonably accurate as long as the
## roughness is much less than the layer thickness
step = False
STEPSIZE=0.5

# calculate full transverse divergence (2 * FWHM) for MAGIK reflectometer
S1_transverse = 150.0
S2_transverse = 25.0
L2 = 330.0
L1 = 1403.0 + 330.0
dTl = 2 * np.ones_like(probe_d2o.Q) * divergence(0, (S1_transverse, S2_transverse), (L1, L2))

bilayer_thickness = Parameter(name='SANS bilayer thickness', value=40.0).range(20.0, 80.0)
bilayer_spacing = Parameter(name='SANS bilayer spacing', value=20.0).range(5.0, 50.0)
bilayer_sld = Parameter(name='SANS bilayer sld', value=0.4).range(-0.5, 0.5)
volume_fraction_bilayer = Parameter(name='SANS volume fraction', value=0.00001).range(0.0, 0.0001)
n_bilayers =  Parameter(name='SANS number of bilayers', value=4).range(1, 20)

sans_parameters = {'scale': 1.0,
                    'background': 0.0,
                    'sld': bilayer_sld,
                    'thick_shell': bilayer_thickness,
                    'thick_solvent': bilayer_spacing,
                    'radius': 100.0,
                    'volfraction': volume_fraction_bilayer,
                    'n_shells': n_bilayers * 2 - 1
                    }

sasmodel_d = SASReflectivityModel(sas_model_name='multilayer_vesicle',
                                dtheta_l=dTl,
                                parameters=sans_parameters | {'sld_solvent': d2o.rho})

sasmodel_h = SASReflectivityModel(sas_model_name='multilayer_vesicle',
                                dtheta_l=dTl,
                                parameters=sans_parameters | {'sld_solvent': h2o.rho})


model_d2o = SASReflectivityMolgroupsExperiment(sas_model=sasmodel_d, sample=sample_d2o, probe=probe_d2o, dz=STEPSIZE, step_interfaces = step)
model_h2o = SASReflectivityMolgroupsExperiment(sas_model=sasmodel_h, sample=sample_h2o, probe=probe_h2o, dz=STEPSIZE, step_interfaces = step)

problem = FitProblem([model_d2o, model_h2o])

problem.name = "tiox_dopc_d2o_h2o"
