from bumps.names import *
from sasmodels.core import load_model
from sasmodels.bumps_model import Model, Experiment
from sasmodels.data import load_data

# IMPORT THE DATA USED
data = load_data('sim.dat')

# DEFINE THE MODEL
kernel = load_model('core_shell_ellipsoid')

pars = dict(scale=1.0, background=0.0005, radius_equat_core=10, x_core=1.5, thick_shell=10, x_polar_shell=1.5, sld_core=3.4, sld_shell=3.5,  sld_solvent=6.4)

model = Model(kernel, **pars)

# PARAMETER RANGES (ONLY THOSE PARAMETERS ARE FITTED)
model.scale.range(0.001, 0.1)
# model.background.range(0, 1)
model.radius_equat_core.range(10, 200)
model.x_core.range(1.0, 10.)
model.thick_shell.range(10., 200.)
model.x_polar_shell.range(1.0, 10.0)
model.sld_core.range(-1, 7)
model.sld_shell.range(-1, 7.)
model.sld_solvent.range(-0.6, 6.4)

M = Experiment(data=data, model=model)
problem = FitProblem(M)
