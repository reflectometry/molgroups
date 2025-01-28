"""Example script using molgroups Refl1D interface objects"""

import numpy as np
from refl1d.names import Parameter, SLD, Slab, FitProblem, load4
from molgroups import components as cmp
from molgroups.refl1d_interface import (SolidSupportedBilayer,
                                        Freeform,
                                        MolgroupsLayer,
                                        MolgroupsStack,
                                        MolgroupsExperiment,
                                        Bilayer,
                                        VolumeFractionBox)

from periodictable.fasta import Sequence

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

dz_overlayer = Parameter(name='separation distance of overlayer', value=10.0).range(0, 30)
vf_overlayer = Parameter(name='volume fraction overlayer', value=0.9).range(0.0, 1.0)

l_surface_group = Parameter(name='surface group length', value=10).range(5, 15)
vf_surface_group = Parameter(name='surface group volume fraction', value=0.5).range(0, 1)
rho_surface_group = Parameter(name='surface group rho', value=4).range(3, 5)

peptide = Sequence(name='peptide',
                   sequence='I AM A PEPTIDE IN LIPIDS',
                   type='aa')
rhoH_peptide = peptide.D2Osld(1, 0)
rhoD_peptide = peptide.D2Osld(1, 1)

CONTROLPOINTS = 12
SPACING = 15.0
dDp = [0.0] * CONTROLPOINTS
dVf = [0.0] * CONTROLPOINTS
for i in range(CONTROLPOINTS):
    dDp[i] = Parameter(name='dDp'+str(i), value=0.0) #.range(-1 * SPACING / 3., SPACING / 3.)
for i in range(0, CONTROLPOINTS-1):
    dVf[i] = Parameter(name='dVf'+str(i), value=0.0).range(-0.001, 1.0)

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

    surface_group = VolumeFractionBox(name='surface group',
                                    z=blm.substrate_surface + 0.5 * l_surface_group,
                                    rhoH=rho_surface_group,
                                    rhoD=rho_surface_group,
                                    volume_fraction=vf_surface_group,
                                    length=l_surface_group,
                                    sigma_bottom=tiox_rough,
                                    sigma_top=tiox_rough)

    blm.l_submembrane = surface_group.length + l_submembrane

    spline = Freeform(name='spline',
                    dSpacing=SPACING,
                    startz=surface_group.top_surface,
                    Dp=dDp,
                    Vf=dVf,
                    rhoH=rhoH_peptide,
                    rhoD=rhoD_peptide,
                    sigma=sigma,
                    nf=vf_bilayer)

    ol = Bilayer(name='overlayer',
                lipids=[DOPC],
                inner_lipid_nf=[1.0],
                outer_lipid_nf=[1.0],
                startz=blm.outer_headgroup_top + dz_overlayer,
                vf_bilayer=vf_overlayer,
                l_lipid1=l_lipid1,
                l_lipid2=l_lipid2,
                sigma=sigma)

    mollayer = MolgroupsLayer(base_group=blm,
                            add_groups=[surface_group, ol],
                            overlay_groups=[spline],
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

model_d2o = MolgroupsExperiment(sample=sample_d2o, probe=probe_d2o, dz=STEPSIZE, step_interfaces = step)
model_h2o = MolgroupsExperiment(sample=sample_h2o, probe=probe_h2o, dz=STEPSIZE, step_interfaces = step)

problem = FitProblem([model_d2o, model_h2o])

problem.name = "tiox_dopc_d2o_h2o"
