import logging
import os

import pandas as pd
import numpy as np

import flyingpace.logginghelper

from ase import Atoms
from ase.units import Bohr, Hartree, Ang, eV
from collections import Counter

log = logging.getLogger(__name__)

def read_cpmd_md(aimd_input_file_path: str, local_dft_dir:str):
    '''
    Reads output data from a CPMD MD run in a given folder 
    and saves it in dft_data.pckl.gzip, which can be processed by pacemaker
    Limitations:
     - Needs GEOMETRY.xyz, ENERGIES, TRAJECTORY, FTRAJECTORY and INPUTFILE to 
       exist in local_dft_dir to function
     - This function will yield nonsense if the aimd run is not done in a cubic cell
    '''

    geometry_file_path = os.path.join(local_dft_dir, 'GEOMETRY.xyz')
    energies_file_path = os.path.join(local_dft_dir, 'ENERGIES')
    trajectory_file_path = os.path.join(local_dft_dir, 'TRAJECTORY')
    forces_file_path = os.path.join(local_dft_dir, 'FTRAJECTORY')

    #Check if all nessesary files exist
    if (os.path.exists(aimd_input_file_path) and\
    os.path.exists(energies_file_path) and\
    os.path.exists(trajectory_file_path) and\
    os.path.exists(forces_file_path)):
        pass
    else: 
        log.warning(f"Check if there are {os.path.basename(aimd_input_file_path)}, ENERGIES,\
        TRAJECTORY and FTRAJECTORY files in {local_dft_dir}")
        raise RuntimeError(f"Check if there are {os.path.basename(aimd_input_file_path)}, ENERGIES,\
        TRAJECTORY and FTRAJECTORY files in {local_dft_dir}")
    
    def get_cell(celldm):
        cell = [[celldm, 0.0, 0.0],
                [0.0, celldm, 0.0],
                [0.0, 0.0, celldm]]
        return cell

    #Data lists for all configurations
    energy_list = []
    forces_list = []
    atoms_list = []

    #Read number of atoms
    with open(geometry_file_path, 'r') as f:
        num_at = len(f.readlines())-2

    #Read number of steps of the trajectory
    with open(trajectory_file_path, 'r') as f:
        num_steps = int(len(f.readlines())/num_at)
        
    #Read stride/sample number of the trajectory
    with open(energies_file_path, 'r') as f:
        stride = int(len(f.readlines())/num_steps)

    #Read the atomic configuration
    f = open(geometry_file_path)
    f.readline()
    f.readline()
    symbols = []
    for lines in range(num_at):
        symbols.append(f.readline().split()[0])

    #Read in relevant files
    ef = open(energies_file_path, 'r')
    tf = open(trajectory_file_path, 'r')
    ff = open(forces_file_path, 'r')

    #Read cell parameter
    cmd1 = f"grep -A1 'CELL' " + aimd_input_file_path + " | tail -1 | awk '{print $1}' "
    celldm = float(os.popen(cmd1).read())
    with open (aimd_input_file_path) as f:
        input_data = f.read()
    if not ('ANGSTROM' in input_data):
        celldm = celldm * Bohr
    cell = get_cell(celldm)

    #Loop through configurations in MD
    for i in range(num_steps):

        pos = np.zeros((num_at, 3))
        force = np.zeros((num_at, 3))
        energy = float(ef.readline().split()[3])

        for j in range(num_at):

            #Read positions and forces per configuration
            tfline = tf.readline()
            ffline = ff.readline()
            pos[j,:] = ([float(k) for k in tfline.split()[1:4]])
            force[j,:] = ([float(k) for k in ffline.split()[7:10]])

        #Convert units
        energy = energy * Hartree
        pos = pos * Bohr
        #pos.tolist()
        force = force * (Hartree/Bohr)
        #force.tolist()

        #Add to lists containing ALL data
        energy_list.append(energy)
        forces_list.append(force)
        atoms_list.append(Atoms(symbols=symbols, positions=pos, cell=cell, pbc=True))

        #Skip not needed lines in ENERGIES file
        for j in range(stride-1):
            ef.readline()

    return symbols, energy_list, forces_list, atoms_list

def read_cpmd_scf(outfile_path: str):
    '''
    Reads a given CPMD outputfile and returns wheter the 
    calculation converged, the chemical symbols,
    the cell, the positions, the forces and the total energy
    Units: Angstrom, eV and eV/Angstrom
    '''
    
    with open(outfile_path, 'r') as f:
        cpmd_lines = f.readlines()

    #Section identifiers
    cpmd_no_convergence = "NO CONVERGENCE"
    cpmd_atoms = "*** ATOMS ***"
    cpmd_cell = "*** SUPERCELL ***"
    cpmd_pos_force = "FINAL RESULTS"
    cpmd_total_energy = "TOTAL ENERGY ="

    indexes = {
        cpmd_no_convergence: [],
        cpmd_atoms: [],
        cpmd_cell: [],
        cpmd_pos_force: [],
        cpmd_total_energy: [],
    }

    for idx, line in enumerate(cpmd_lines):
        for identifier in indexes:
            if identifier in line:
                indexes[identifier].append(idx)

    #returns false if indexes[cpmd_no_convergence] is empty
    # -> if list not empty -> returns true -> convergence = False
    # -> if list empty -> returns false -> convergence = True
    if indexes[cpmd_no_convergence]:
        convergence = False
        symbols = None
        cell = None
        positions = None
        forces = None
        energy = None
    else:
        convergence = True

    #Parse atoms symbols
    symbols = []
    symbols_idx = indexes[cpmd_atoms][0]+2
    symbol_line = cpmd_lines[symbols_idx].strip().split()
    atom_number = symbol_line[0]
    break_line = "****************************************************************"
    while (atom_number != break_line):
        chemical_symbol = symbol_line[1]
        symbols.append(chemical_symbol)
        symbols_idx += 1
        symbol_line = cpmd_lines[symbols_idx].strip().split()
        atom_number = symbol_line[0]
    number_of_atoms = len(symbols)

    #Parse cell
    cell = []
    cell_idx = indexes[cpmd_cell][0]+5
    for i in range(3):
        vec = cpmd_lines[cell_idx].strip().split()[3:6]
        cell.append(vec)
        cell_idx += 1
    cell = np.asarray(cell, dtype=float) * Bohr

    #Parse positions and forces
    positions = []
    forces = []
    pos_force_idx = indexes[cpmd_pos_force][0]+5
    for i in range(number_of_atoms):
        position_line = cpmd_lines[pos_force_idx].strip().split()[2:5]
        force_line = cpmd_lines[pos_force_idx].strip().split()[5:8]
        positions.append(position_line)
        forces.append(force_line)
        pos_force_idx += 1
    
    positions = np.asarray(positions, dtype=float) * Bohr
    forces = np.asarray(forces, dtype=float) * Hartree / Bohr

    #Parse energy 
    energy = None
    energy_idx = indexes[cpmd_total_energy][-1]
    energy = float(cpmd_lines[energy_idx].strip().split()[4]) * Hartree

    return convergence, symbols, cell, positions, forces, energy 


def write_cpmd_input(structure: Atoms, input_file_path: str, dft_dict: dict):

    #Read what is needed from dft_dict
    assert isinstance(dft_dict, dict)
    if "dftParams" in dft_dict:
        input_data_dict = dft_dict["dftParams"]
    else:
        log.warning("No 'dftParams' provided in YAML file, please provide it")
        raise ValueError("No 'dftParams' provided in YAML file, please provide it")

    input_options = {
        'calculation' : {
            'scf' : '    OPTIMIZE WAVEFUNCTION\n',
            'md' : '    MOLECULAR DYNAMICS BO\n',
        },

        'restartMode' : {
            'fromScratch' : '    INITIALIZE WAVEFUNCTION RANDOM\n',
        },

        #Input options regarding moleculra dynamics
        'maxStep' : '    MAXSTEP\n        {maxStep}\n',
        'timeStep' : '    TIMESTEP\n        {timeStep}\n',
        'trajStep' : '    TRAJECTORY SAMPLE XYZ FORCES\n        {trajStep}\n',
        'noseParams' : '    NOSE PARAMETERS\n        {noseParams}\n',
        'temp' : '    NOSE IONS MASSIVE\n        {temp}\n',

        #Input options regarding scf
        'maxIter' : '    MAXITER\n        {maxIter}\n',
        'convOrbital' : '    CONVERGENCE ORBITALS\n        {convOrbital}\n',
        'splinePoints' : '    SPLINE POINTS\n        {splinePoints}\n',
        'functional' :{
            'pbeSol' : '    GRADIENT CORRECTION PBESX PBESC\n',
        },
        'pwCutoff' : '    CUTOFF\n        {pwCutoff}\n',   
    }

    cpmd_technical_block = "    ODIIS NO_RESET=-1\n        10\n    MEMORY BIG\n\
    PRINT FORCES ON\n    RNLSM_AUTOTUNE\n        20\n    USE_BATCHFFT ON\n    ALL2ALL_BATCHSIZE \n       4000\n\
    TUNE_FFT_BATCHSIZE ON\n        10\n    BLOCKSIZE_USPP\n        500\n    PARA_USE_MPI_IN_PLACE\n\
    PARA_BUFF_SIZE\n        0\n    PARA_STACK_BUFF_SIZE\n        0\n    DISTRIBUTED FNL ROT OFF\n\
    USE_OVERLAPPING_COMM_COMP ON\n    USE_ELPA OFF\n"

    #Start with empty input string
    input = ""
    input += "&INFO\n    Automatically generated input file\n&END\n\n"

    #Start with &CPMD section

    input += "&CPMD\n"

    if "calculation" in input_data_dict:
        if input_data_dict["calculation"] in input_options["calculation"]:
            input += input_options["calculation"][input_data_dict["calculation"]]
        else:
            log.warning("'calculation' type in 'dftParams' is not known")
            raise ValueError("'calculation' type in 'dftParams' is not known")
    else:
        log.warning("'calculation' type is not provided in 'dftParams' section, please specify it")
        raise ValueError("'calculation' type is not provided in 'dftParams' section, please specify it")
        
    if "restartMode" in input_data_dict:
        if input_data_dict["restartMode"] in input_options["restartMode"]:
            input += input_options["restartMode"][input_data_dict["restartMode"]]
        else:
            input += input_options["restartMode"]["fromScratch"]
    else: 
        input += input_options["restartMode"]["fromScratch"]

    #For MD simulations
    if (input_data_dict["calculation"] == 'md'):
         
        if "maxStep" in input_data_dict:
            input += input_options["maxStep"]
        else:
            log.warning("'maxStep' type is not provided in 'dftParams' section, please specify it")
            raise ValueError("'maxStep' type in 'dftParams' is not known")
        
        if "timeStep" in input_data_dict:
            input += input_options["timeStep"]
        else:
            log.warning("'timeStep' type is not provided in 'dftParams' section, please specify it")
            raise ValueError("'timeStep' type in 'dftParams' is not known")
        
        if "trajStep" in input_data_dict:
            input += input_options["trajStep"]
        else:
            log.warning("'trajStep' type is not provided in 'dftParams' section, please specify it")
            raise ValueError("'trajStep' type in 'dftParams' is not known")
        
        if "noseParams" in input_data_dict:
            input += input_options["noseParams"]
        else:
            log.warning("'noseParams' type is not provided in 'dftParams' section, please specify it")
            raise ValueError("'noseParams' type in 'dftParams' is not known")
        
        if "temp" in input_data_dict:
            input += input_options["temp"]
        else:
            log.warning("'temp' type is not provided in 'dftParams' section, please specify it")
            raise ValueError("'temp' type in 'dftParams' is not known")
        
        input += "    EXTRAPOLATE WFN STORE\n        6\n"
        input += "    STORE\n        {maxStep}\n"
        input += f"    SUBTRACT COMVEL\n        {int(int(input_data_dict['maxStep'])/100)}\n"

    if "maxIter" in input_data_dict:
        input += input_options["maxIter"]
    else:
        input_data_dict["maxIter"] = 100
        input += input_options["maxIter"]
    
    if "convOrbital" in input_data_dict:
        input += input_options["convOrbital"]
    else:
        input_data_dict["convOrbital"] = '1.d-6'
        input += input_options["convOrbital"]
    
    if "splinePoints" in input_data_dict:
        input += input_options["splinePoints"]
    else:
        input_data_dict["splinePoints"] = 5000
        input += input_options["splinePoints"]

    input += "    REAL SPACE WFN KEEP\n"
        
    input += cpmd_technical_block
    input += "&END\n\n"

    #Start with &SYSTEM section

    input += "&SYSTEM\n    ANGSTROM\n    CELL VECTORS\n"

    #Get cell vectors as strings from the structure
    cell = []
    for i in range(3):
        cell.append("{:10.8f} {:10.8f} {:10.8f}".format(structure.get_cell()[i][0], structure.get_cell()[i][1], structure.get_cell()[i][2]))
        input += f"        {cell[i]}\n"
    
    if "pwCutoff" in input_data_dict:
        input += input_options["pwCutoff"]
    else:
        input_data_dict["pwCutoff"] = 30
        input += input_options["pwCutoff"]

    input += "&END\n\n"
    
    #Start with &DFT section

    input += "&DFT\n    OLDCODE\n"

    if "functional" in input_data_dict:
        if input_data_dict["functional"] in input_options["functional"]:
            input += input_options["functional"][input_data_dict["functional"]]
        else:
            log.warning("'functional' type in dft options is not known")
            raise ValueError("'functional' type in dft options is not known")

    input += "    EXCHANGE CORRELATION TABLE NO\n&END\n\n"

    #Start with the &ATOMS section

    element_dict = Counter(structure.get_chemical_symbols())
    coords = structure.get_positions()
    pseudopotentials = input_data_dict["pseudopotentials"]

    input += "&ATOMS\n"

    offset = 0
    for key in pseudopotentials:
        #CPMD is sensitive that there is no space before the *, otherwise it cant read the atoms
        input += f"*{pseudopotentials[key]} BINARY\n  LMAX=F\n"
        species_number = element_dict[key]
        input += '  {}\n'.format(species_number) 
        for j in range(species_number):
            input += '  {:10.6f} {:10.6f} {:10.6f} \n'.format(coords[j+offset,0],coords[j+offset,1],coords[j+offset,2])
        offset += species_number
        input += '\n'
    input += '&END\n'

    #Set values in template string
    input = input.format(**input_data_dict)

    with open(input_file_path, "w") as f:
            print(input, file=f)

    return

