import numpy

from periodictable.fasta import Molecule, AMINO_ACID_CODES as aa
from periodictable.core import default_table
from periodictable.fasta import xray_sld

def pdb_to_residue_data(pdbfilename, selection='all', center_of_mass=numpy.array([0, 0, 0]),
              deuterated_residues=None, xray_wavelength=1.5418):
    """
    Processes a PDB file into an 8-column residue data array for use with ContinuousEuler from a pdb file\
    with optional selection. "center_of_mass" is the position in space at which to position the
    molecule's center of mass. "deuterated_residues" is a list of residue IDs for which to use deuterated values

    Returns (n_residues x 8) array with columns:
    1. residue number
    2. x coordinate
    3. y coordinate
    4. z coordinate
    5. residue volume
    6. electon scattering length
    7. neutron scattering length (H)
    8. neutron scattering length (D)
    
    """
    import MDAnalysis
    from MDAnalysis.lib.util import convert_aa_code

    if deuterated_residues is None:
        deuterated_residues = []

    elements = default_table()

    molec = MDAnalysis.Universe(pdbfilename)
    sel = molec.select_atoms(selection)
    Nres = sel.n_residues

    if not Nres:
        print('Warning: no atoms selected')

    sel.translate(-sel.center_of_mass() + center_of_mass)

    resnums = []
    rescoords = []
    resscatter = []
    resvol = numpy.zeros(Nres)
    resesl = numpy.zeros(Nres)
    resnslH = numpy.zeros(Nres)
    resnslD = numpy.zeros(Nres)
    deut_header = ''

    for i in range(Nres):
        resnum = sel.residues[i].resid
        resnums.append(resnum)
        rescoords.append(sel.residues[i].atoms.center_of_mass())
        key = convert_aa_code(sel.residues[i].resname)
        if resnum in deuterated_residues:
            resmol = Molecule(name='Dres', formula=aa[key].formula.replace(elements.H, elements.D),
                              cell_volume=aa[key].cell_volume)
        else:
            resmol = aa[key]
        resvol[i] = resmol.cell_volume
        # TODO: Make new column for xray imaginary part (this is real part only)
        resesl[i] = resmol.cell_volume * xray_sld(resmol.formula, wavelength=xray_wavelength)[0] * 1e-6
        resnslH[i] = resmol.cell_volume * resmol.sld * 1e-6
        resnslD[i] = resmol.cell_volume * resmol.Dsld * 1e-6

    resnums = numpy.array(resnums)
    rescoords = numpy.array(rescoords)

    # replace base value in nsl calculation with proper deuterated scattering length
    # resnsl = resscatter[:, 2]

    return numpy.hstack((resnums[:, None], rescoords, resvol[:, None], resesl[:, None],
                                             resnslH[:, None], resnslD[:, None]))


def pdbto8col(pdbfilename, datfilename, selection='all', center_of_mass=numpy.array([0, 0, 0]),
            deuterated_residues=None, xray_wavelength=1.5418):
    """Saves 8-column residue data to a file (for back compatibility)"""

    processed_data = pdb_to_residue_data(pdbfilename, selection, center_of_mass,
                        deuterated_residues, xray_wavelength)
    
    deut_header = ''
    if deuterated_residues is not None:
        deut_header = 'deuterated residues: ' + ', '.join(map(str, deuterated_residues)) + '\n'

    resvol = processed_data[:, 4]
    resnslH = processed_data[:, 6]
    resnslD = processed_data[:, 7]

    average_sldH = numpy.sum(resnslH[:, ]) / numpy.sum(resvol[:, ])
    average_sldD = numpy.sum(resnslD[:, ]) / numpy.sum(resvol[:, ])
    average_header = f'Average nSLD in H2O: {average_sldH}\nAverage nSLD in D2O: {average_sldD}\n'

    numpy.savetxt(datfilename, processed_data, delimiter='\t',
                header=pdbfilename + '\n' + deut_header + average_header + 'resid\tx\ty\tz\tvol\tesl\tnslH\tnslD')

    return datfilename
