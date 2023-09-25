import ase.io
import glob
import logging
import numpy as np
import os
import pandas as pd
import random
import sys
import time
import yaml

from ase import Atoms
from ase.io.lammpsrun import read_lammps_dump_text
from ase.units import Bohr, Hartree, Ang, eV
from collections import Counter
from fabric import Connection
from patchwork.files import exists

import flyingpace.dirmanager
import flyingpace.logginghelper

from flyingpace.constants import implemented_dft_codes
from flyingpace.input import DataReader

log = logging.getLogger(__name__)

#TODO: How is this done when casting the script as a python package
template_directory = '/ccc160/mgossler/phd/work/project-ace/flyingPACE/templates/'

#########################################################################################
##                                                                                     ##
##                          Functions to check for idempotency                         ##
##                                                                                     ##
#########################################################################################

def calc_done_in_local_dir(local_dir: str):
    '''
    Checks if the 'CALC_DONE' file exists in the local folder,
    returns a boolean
    '''

    calc_done_file_path = os.path.join(local_dir, 'CALC_DONE')

    if os.path.exists(calc_done_file_path):
        calc_done = True
    else:
        calc_done = False

    return calc_done

def calc_done_in_remote_dir(remote_dir: str, remote_connection: Connection):
    '''
    Checks if the 'CALC_DONE' file exists in the remote folder,
    returns a boolean
    '''
    calc_done_file_path = os.path.join(remote_dir, 'CALC_DONE')

    if exists(remote_connection, calc_done_file_path):
        calc_done = True
    else:
        calc_done = False

    return calc_done

def calc_ongoing_in_local_dir(local_dir: str):
    '''
    Checks if the 'CALC_ONGOING' file exists in the local folder,
    returns a boolean
    '''

    calc_ongoing_file_path = os.path.join(local_dir, 'CALC_ONGOING')

    if os.path.exists(calc_ongoing_file_path):
        calc_ongoing = True
    else:
        calc_ongoing = False

    return calc_ongoing

def calc_ongoing_in_remote_dir(remote_dir: str, remote_connection: Connection):
    '''
    Checks if the 'CALC_ONGOING' file exists in the remote folder,
    returns a boolean
    '''
    calc_ongoing_file_path = os.path.join(remote_dir, 'CALC_ONGOING')

    if exists(remote_connection, calc_ongoing_file_path):
        calc_ongoing = True
    else:
        calc_ongoing = False

    return calc_ongoing


def wait_for_calc_done(directory: str, remote_connection: Connection):
    '''
    Checks every 10 seconds if the file 'CALC_DONE' exists in 
    'remote_dir' and waits if it does not
    '''

    calc_done_file_path = os.path.join(directory, 'CALC_DONE')

    if (remote_connection == None):
        while not (os.path.exists(calc_done_file_path)):
            time.sleep(10)
    else:
        while not (exists(remote_connection, calc_done_file_path)):
            time.sleep(10)

    return

def check_for_calc_done_in_script(run_script_name: str):
    '''
    Checks if the statement 'touch CALC_DONE' exists in the given file 
    and if not it appends it with this statement and rewrites the file
    '''

    with open (run_script_name) as f:
        run_script_str = f.read()

    if ('touch CALC_DONE' in run_script_str):
        pass
    else: 
        run_script_str += '\n touch CALC_DONE'
        with open(run_script_name, "w") as f:
            print(run_script_str, file=f)
    return

        

#########################################################################################
##                                                                                     ##
##                          Functions for handling pacemaker I/O                       ##
##                                                                                     ##
#########################################################################################

def generate_pace_input(dataset_file_path: str, directory_dict: dict, InputData: DataReader):
    '''
    Generate a inputfile for pacemaker based on data from the master input file
    and save it in the current generation local_train_dir
    '''

    pacemaker_dict = InputData.pacemaker_dict


    #Read what is needed from pacemaker_dict
    assert isinstance(pacemaker_dict, dict)
    if "testSize" in pacemaker_dict:
        test_size = pacemaker_dict["testSize"]
        log.info(f"Testsize for pacemaker run: {test_size}")
    else:
        log.warning("No 'testSize' provided in YAML file, the default is 0")
        test_size = 0

    if "numberOfFunctions" in pacemaker_dict:
        num_functions = pacemaker_dict["numberOfFunctions"]
        log.info(f"Number of functions for pacemaker run: {num_functions}")
    else:
        log.warning("No 'numberOfFunctions' provided in YAML file, the default is 700")
        num_functions = 700

    if "cutoff" in pacemaker_dict:
        cutoff = pacemaker_dict["cutoff"]
        log.info(f"Cutoff for pacemaker run: {cutoff}")
    else:
        log.warning("No 'cutoff' provided in YAML file, the default is 7.0")
        cutoff = 7.0

    if "weighting" in pacemaker_dict:
        weighting_inp = pacemaker_dict["weighting"]
        log.info(f"Weighting for pacemaker run: {weighting_inp}")
    else:
        log.warning("No 'weighting' provided in YAML file, the default is 'uniform'")
        weighting_inp = 'uniform'

    if "kappa" in pacemaker_dict:
        kappa = pacemaker_dict["kappa"]
        log.info(f"Kappa for pacemaker run: {kappa}")
    else:
        log.warning("No 'kappa' provided in YAML file, the default is 0.3")
        weighting = 0.3
    
    if "maxNumIterations" in pacemaker_dict:
        max_num_iter = pacemaker_dict["maxNumIterations"]
        log.info(f"Maximum number of iterations for pacemaker run: {max_num_iter}")
    else:
        log.warning("No 'maxNumIterations' provided in YAML file, the default is 500")
        max_num_iter = 500

    #Get relevant directories from directory_dict
    assert isinstance(directory_dict, dict)
    local_train_dir = directory_dict["local_train_dir"]

    #Checking dataset
    df = pd.read_pickle(dataset_file_path, compression="gzip")
    
    if 'ase_atoms' in df.columns:
        log.info("Determining available elements...")
        elements_set = set()
        df["ase_atoms"].map(lambda at: elements_set.update(at.get_chemical_symbols()));
        elements = sorted(elements_set)
        log.info(f"Found elements: {elements}")
    else:
        log.info("ERROR! No `ase_atoms` column found")
        sys.exit(1)

    log.info(f"Number of elements: {len(elements)}")
    log.info("Elements: {elements}")

    #Weighting scheme
    default_energy_based_weighting = """{ type: EnergyBasedWeightingPolicy, DElow: 1.0, 
                                    DEup: 10.0, DFup: 50.0, DE: 1.0, DF: 1.0, wlow: 0.75, 
                                    energy: convex_hull, reftype: all,seed: 42}"""
    weighting = None
    if weighting_inp not in ['uniform', 'energy']:
        log.info("ERROR! No valid weighting_scheme")
        sys.exit(1)
    if weighting_inp == "energy":
        weighting = default_energy_based_weighting
        log.info("Use EnergyBasedWeightingPolicy: ", weighting)
    else:
        weighting = None
        log.info("Use UniformWeightingPolicy")

    with open(os.path.join(template_directory, "input.yaml"), "r") as f:
        input_yaml_text = f.read()

    input_yaml_text = input_yaml_text.replace("{{ELEMENTS}}", str(elements))
    input_yaml_text = input_yaml_text.replace("{{CUTOFF}}", str(cutoff))
    input_yaml_text = input_yaml_text.replace("{{DATAFILENAME}}", os.path.basename(dataset_file_path))
    input_yaml_text = input_yaml_text.replace("{{number_of_functions_per_element}}",
                                              f"number_of_functions_per_element: {num_functions}")
    input_yaml_text = input_yaml_text.replace("{{KAPPA}}", str(kappa))
    input_yaml_text = input_yaml_text.replace("{{MAXITER}}", str(max_num_iter))

    if weighting:
        input_yaml_text = input_yaml_text.replace("{{WEIGHTING}}", f"weighting: {weighting}")
    else:
        input_yaml_text = input_yaml_text.replace("{{WEIGHTING}}", "")

    if test_size > 0:
        input_yaml_text = input_yaml_text.replace("{{test_size}}", f"test_size: {test_size}")
    else:
        input_yaml_text = input_yaml_text.replace("{{test_size}}", "")

    with open(os.path.join(local_train_dir, "input.yaml"), "w") as f:
        print(input_yaml_text, file=f)

    log.info(f"Pacemaker input file is written into {os.path.join(local_train_dir, 'input.yaml')}")

    return

#########################################################################################
##                                                                                     ##
##                          Functions for handling lammps I/O                          ##
##                                                                                     ##
#########################################################################################

def generate_lammps_input(datafile_path: str, directory_dict: dict, InputData: DataReader):
    '''
    Generate a inputfile for lammps based on data from the master input file
    and save it in the current generation local_exploration_dir
    '''

    exploration_dict = InputData.exploration_dict

    #Read what is needed from pacemaker_dict
    assert isinstance(exploration_dict, dict)
    if "elementList" in exploration_dict:
        element_list = exploration_dict["elementList"]
    else:
        log.warning("No 'elementList' provided in YAML file, please specify it")
        raise ValueError("No 'elementList' provided in YAML file, please specify it")
    
    if "temp" in exploration_dict:
        temp = exploration_dict["temp"]
    else:
        temp = 400.0

    if "steps" in exploration_dict:
        steps = exploration_dict["steps"]
    else:
        steps = 50000

    if "lowerGamma" in exploration_dict:
        lower_gamma = exploration_dict["lowerGamma"]
    else:
        lower_gamma = 5

    if "upperGamma" in exploration_dict:
        upper_gamma = exploration_dict["upperGamma"]
    else:
        upper_gamma = 25

    #Read what is needed from directory_dict
    assert isinstance(directory_dict, dict)
    local_exploration_dir = directory_dict["local_exploration_dir"]

    lammps_inp_template_file_path = os.path.join(template_directory, "INP-lammps")
    lammps_inp_save_file_path = os.path.join(local_exploration_dir, 'INP-lammps')

    with open(lammps_inp_template_file_path, "r") as f:
        input_lammps_text = f.read()

    input_lammps_text = input_lammps_text.replace("{{DATAFILE}}", os.path.basename(datafile_path))
    input_lammps_text = input_lammps_text.replace("{{ELEMENTLIST}}", element_list)
    input_lammps_text = input_lammps_text.replace("{{TEMP}}", str(temp))
    input_lammps_text = input_lammps_text.replace("{{NUM_STEPS}}", str(steps))
    input_lammps_text = input_lammps_text.replace("{{LOWERGAMMA}}", str(lower_gamma))
    input_lammps_text = input_lammps_text.replace("{{UPPERGAMMA}}", str(upper_gamma))

    with open(lammps_inp_save_file_path, "w") as f:
        print(input_lammps_text, file=f)

    log.info(f"Input file is written into {lammps_inp_save_file_path}")

    return

#########################################################################################
##                                                                                     ##
##                          Functions for handling dft I/O                             ##
##                                                                                     ##
#########################################################################################


def check_dft_input_file_type(dft_input_file_path: str, dft_code: str):
    '''
    Tries to check wheter the specidied DFT input file matches the
    chosen DFT code, throws an error if they do not match
    '''

    with open (dft_input_file_path) as f:
        dft_input_str = f.read()

    if (dft_code == 'CPMD'):
        if ('&CPMD' in dft_input_str):
            log.info(f"{dft_input_file_path} is propably a CPMD input file, it fits the chosen code")
        else: 
            log.info(f"{dft_input_file_path} can't be a valid CPMD input file, please check it")
            raise ValueError(f"{dft_input_file_path} can't be a valid CPMD input file, please check it")
    else:
        log.info(f"The type of the input file {dft_input_file_path} is not recognized,\
                please check again if it fits the chosen DFT code")
        raise ValueError(f"The type of the input file {dft_input_file_path} is not recognized,\
                please check again if it fits the chosen DFT code")
    
    return
    
def check_dft_job_type(dft_input_file_path: str, dft_code: str):
    '''
    Returns a string describing the job type, based on which DFT code is used
    Implemented job types:
    'md' : Molecular dynamics simulation
    '''

    with open (dft_input_file_path) as f:
        dft_input_str = f.read()

    if (dft_code == 'CPMD'):
        if ('MOLECULAR DYNAMICS' in dft_input_str):
            job_type = 'md'
        else: 
            log.info(f"The used DFT code is {dft_code} and the job type of {dft_input_file_path} not recognized")
            raise ValueError(f"The used DFT code is {dft_code} and the job type of {dft_input_file_path} not recognized")
        
        return job_type
    
    else:
        log.info(f"The chosen DFT code '{dft_code}' is not implemented")
        raise NotImplementedError(f"The chosen DFT code '{dft_code}' is not implemented")
    
    return

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

#########################################################################################
##                                                                                     ##
##                          Functions for handling pickle files                        ##
##                                                                                     ##
#########################################################################################

def merge_pickle(directory_dict: dict):
    '''
    Merges the pickle file 'dft_data.pckl.gzip' from prev_local_dft_dir
    with 'new_dft_data.pckl.gzip' from local_dft_dir and saves the resulting 
    pickle file as 'dft_data.pckl.gzip' in local_dft_dir
    'dft_data.pckl.gzip' in prev_local_dft_dir must contain a column containing
    the reference energy
    '''

    #Get relevant directories from directory_dict
    assert isinstance(directory_dict, dict)
    local_dft_dir = directory_dict["local_dft_dir"]
    prev_local_dft_dir = flyingpace.dirmanager.get_prev_path(local_dft_dir)

    new_pickle_file_path = os.path.join(local_dft_dir, 'new_dft_data.pckl.gzip')
    prev_pickle_file_path = os.path.join(prev_local_dft_dir, 'dft_data.pckl.gzip')
    pickle_file_path = os.path.join(local_dft_dir, 'dft_data.pckl.gzip')

    new_data = pd.read_pickle(new_pickle_file_path, compression="gzip")
    old_data = pd.read_pickle(prev_pickle_file_path, compression="gzip")

    reference_energy = old_data.loc[:,'reference_energy'][0]
    energy_corrected = new_data.loc[:,'energy']-reference_energy

    new_data['energy_corrected'] = energy_corrected
    new_data['reference_energy'] = reference_energy

    combined_data = pd.concat([old_data,new_data], ignore_index=True)

    combined_data.to_pickle(pickle_file_path, compression='gzip', protocol=4)
    log.info(f"Saved new combined pickle file as'{pickle_file_path}'")

    return

def datafile_from_pickle(pickle_file: str, pickle_dir: str, mode:str):
    '''
    Reads the pickle file 'pickle_file' from 'pickle_dir', 
    chooses a random structure and saves it as a lammps data file
    called 'random_struc.data' in 'pickle_dir', returns the file path
    '''
    pickle_path = os.path.join(pickle_dir, pickle_file)

    if (os.path.exists(pickle_path)):
        pass
    else:
        log.warning(f"The pickle file {pickle_path} does not exist!")
        raise RuntimeError(f"The pickle file {pickle_path} does not exist!")
    
    df = pd.read_pickle(pickle_path, compression="gzip")

    if (mode == 'random'):
        struc_path = os.path.join(pickle_dir, 'random_structure.data')
        num_structures = df.shape[0]
        structure_id = random.randint(0,num_structures-1)
        structure = df.loc[structure_id,'ase_atoms']

    if (mode == 'last'):
        struc_path = os.path.join(pickle_dir, 'last_structure.data')
        structure = df.loc[-1,'ase_atoms']

    ase.io.write(struc_path, images=structure, format='lammps-data', masses=True)

    return struc_path

def extrapolative_to_pickle(directory_dict: dict, InputData: DataReader):
    '''
    Read a lammps dump file called "extrapolative_structures.dump" from this 
    generations local_exploration_dir and saves the structures in a pickle file
    in local_exploration_dir
    '''

    exploration_dict = InputData.exploration_dict

    #Read what is needed from dft_dict
    assert isinstance(exploration_dict, dict)
    if "elementList" in exploration_dict:
        element_list = exploration_dict["elementList"]
    else:
        log.warning("No 'elementList' provided in YAML file, please specify it")
        raise ValueError("No 'elementList' provided in YAML file, please specify it")
    
    #Get relevant directories from directory_dict
    assert isinstance(directory_dict, dict)
    local_exploration_dir = directory_dict["local_exploration_dir"]
    #Construct absolute paths for files
    extrapolative_structures_path = os.path.join(local_exploration_dir, "extrapolative_structures.dump")
    pickle_file_path = os.path.join(local_exploration_dir, "extrapolative_structures.pckl.gzip")

    if os.path.exists(pickle_file_path):
        log.info(f"The pickle file {pickle_file_path} already exists")
        return

    if os.path.exists(extrapolative_structures_path):
        pass
    else:
        log.warning(f"The file {extrapolative_structures_path}, does not exist, cannot gather extrapolative structures to pickle file")
        raise RuntimeError(f"The file {extrapolative_structures_path}, does not exist, cannot gather extrapolative structures to pickle file")
    
    species_to_element_dict = {i + 1: e for i, e in enumerate(element_list.split())}
    
    with open(extrapolative_structures_path) as f:
        structures = read_lammps_dump_text(f, index=slice(None))

    if not structures:
        log.warning(f"No extrapolative structures found!")
        raise RuntimeError(f"No extrapolative structures found!")

    for at in structures:
        new_symb = [species_to_element_dict[i] for i in at.get_atomic_numbers()]
        at.set_chemical_symbols(new_symb)
        if np.linalg.det(at.get_cell()) > 0:
            at.set_pbc(True)

    df = pd.DataFrame({"ase_atoms": structures})
    df.to_pickle(pickle_file_path, compression='gzip', protocol=4)
    log.info(f"Saved {pickle_file_path}")

    return

def prepare_scf_calcs_from_pickle(dft_dict: dict, directory_dict: dict):
    '''
    Reads 'extrapolative_structures.pckl.gzip' from prev_local_exploration_dir
    and constructs CPMD scf input files from them in local_dft_dir, one folder for each calculation
    '''

    #Read what is needed from dft_dict
    assert isinstance(dft_dict, dict)
    if "scfInput" in dft_dict:
        scf_input_file = dft_dict["scfInput"]
    else:
        log.warning("No 'scfInput' provided in YAML file, the default is 100")
        raise ValueError("No 'scfInput' provided in YAML file, please specify it")

    if "maxScfRuns" in dft_dict:
        max_scf_runs = dft_dict["maxScfRuns"]
    else:
        log.warning("No 'maxScfRuns' provided in YAML file, the default is 100")
        max_scf_runs = 100

    if "pseudopotentials" in dft_dict:
        pseudopotentials = dft_dict["pseudopotentials"]
        assert isinstance(pseudopotentials, dict)
    else:
        log.warning("No 'pseudopotentials' provided in YAML file, please specify it")
        raise ValueError("No 'pseudopotentials' provided in YAML file, please specify it")
    
    if "ecut" in dft_dict:
        ecut = dft_dict["ecut"]
    else:
        log.warning("No 'ecut' provided in YAML file, the default is 30")
        ecut = 30


    #Get relevant directories from directory_dict
    assert isinstance(directory_dict, dict)
    local_working_dir = directory_dict["local_working_dir"]
    local_dft_dir = directory_dict["local_dft_dir"]
    prev_local_exploration_dir = flyingpace.dirmanager.get_prev_path(directory_dict["local_exploration_dir"])
    #Construct absolute paths for files
    pickel_file_path = os.path.join(prev_local_exploration_dir, "extrapolative_structures.pckl.gzip")
    scf_input_template_file_path = os.path.join(local_working_dir, scf_input_file)

    data = pd.read_pickle(pickel_file_path, compression='gzip')

    if (data.shape[0] == 0):
        log.warning(f"No extrapolative structures found!")
        raise RuntimeError(f"No extrapolative structures found!")

    with open(scf_input_template_file_path, "r") as f:
        input_text = f.read()

    structures = data.loc[:,'ase_atoms']

    scf_dir_num = 1

    for i in structures:
        structure = i
        celldm = i.get_cell()[0][0]
        element_dict = Counter(structure.get_chemical_symbols())
        coords = structure.get_positions()

        structure_string = '&ATOMS\n'
        offset = 0
        for key in pseudopotentials:
            structure_string += '*' + pseudopotentials[key] + ' BINARY\n LMAX=F\n'
            species_number = element_dict[key]
            structure_string += ' {}\n'.format(species_number) 
            for j in range(species_number):
                structure_string += ' {:10.6f} {:10.6f} {:10.6f} \n'.format(coords[j+offset,0],coords[j+offset,1],coords[j+offset,2])
            offset += species_number
            structure_string += '\n'
        structure_string += '&END\n'

        scf_dir = os.path.join(local_dft_dir, "scf." + str(scf_dir_num))
        scf_input_file_path = os.path.join(scf_dir, "INP")
        os.mkdir(scf_dir)
        output_text = input_text + structure_string
        output_text = output_text.replace("{{CELLDM}}", "{:10.8f}".format(celldm))
        output_text = output_text.replace("{{ECUT}}", str(ecut))

        with open(scf_input_file_path, "w") as f:
            print(output_text, file=f)

        scf_dir_num += 1
        if (scf_dir_num > max_scf_runs):
            break

    return

def scfs_to_pickle(output_file_dir: str, file_pattern: str, directory_dict: dict, InputData: DataReader):
    '''
    Reads a number of SCF output files in the directory output_file_dir positioned in 
    local_dft_dir. The filenames follow the fromat of {file_pattern}.* with * being a 
    consecutive numbering. Saves the data as 'new_dft_data.pckl.gzip' in local_dft_dir
    without the corrected_energy column
    '''

    dft_dict = InputData.dft_dict

    #Read what is needed from dft_dict
    assert isinstance(dft_dict, dict)
    if "dftCode" in dft_dict:
        dft_code = dft_dict["dftCode"]
        if (dft_code in implemented_dft_codes):
            pass
        else:
            log.warning("The chosen DFT code is not implemented")
            raise NotImplementedError("The chosen DFT code is not implemented")
    else:
        log.warning("No 'dftCode' provided in YAML file, please specify it")
        raise ValueError("No 'dftCode' provided in YAML file, please specify it")

    assert isinstance(directory_dict, dict)
    local_dft_dir = directory_dict["local_dft_dir"]

    pickle_file_path = os.path.join(local_dft_dir, 'new_dft_data.pckl.gzip')
    output_file_pattern_path = os.path.join(local_dft_dir, output_file_dir, file_pattern)
    
    num_scf_output_files = len(glob.glob(output_file_pattern_path + '.*'))

    atoms_list = []
    forces_list = []
    energy_list = []
    
    for i in range(num_scf_output_files):
        
        convergence, symbols, cell, positions, forces, energy = read_scf_map[dft_code](output_file_pattern_path + "." + str(i+1))
        
        if (convergence):
            atoms_list.append(Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True))
            forces_list.append(forces)
            energy_list.append(energy)
        else: 
            pass

    data = {'energy': energy_list,
            'forces': forces_list,
            'ase_atoms': atoms_list}
    df = pd.DataFrame(data)

    if (df.shape[0] == 0):
        log.warning(f"No extrapolative structure SCF calculation has converged!")
        raise RuntimeError(f"No extrapolative structure SCF calculation has converged!")

    df.to_pickle(pickle_file_path, compression='gzip', protocol=4)
    log.info(f"Saved '{pickle_file_path}'")

def aimd_to_pickle(directory_dict: dict, InputData: DataReader):
    '''
    Reads output data from a AIMD run in the dft folder of the current generation 
    and saves it in 'dft_data.pckl.gzip', which can be processed by pacemaker
    It chooses the right parsing function with the dft_code keyword
    '''

    dft_dict = InputData.dft_dict
    pacemaker_dict = InputData.pacemaker_dict

    #Read what is needed from dft_dict
    assert isinstance(dft_dict, dict)
    if "dftCode" in dft_dict:
        dft_code = dft_dict["dftCode"]
        if (dft_code in implemented_dft_codes):
            pass
        else:
            log.warning("The chosen DFT code is not implemented")
            raise NotImplementedError("The chosen DFT code is not implemented")
    else:
        log.warning("No 'dftCode' provided in YAML file, please specify it")
        raise ValueError("No 'dftCode' provided in YAML file, please specify it")
    
    if "aimdInput" in dft_dict:
        aimd_input_file = dft_dict["aimdInput"]
    else:
        log.warning("No 'aimdInput' provided in YAML file, please specify it")
        raise ValueError("No 'aimdInput' provided in YAML file, please specify it")
    
    #Read what is needed from pacemaker_dict
    assert isinstance(pacemaker_dict, dict)
    if "referenceEnergyMode" in pacemaker_dict:
        reference_energy_mode = pacemaker_dict["referenceEnergyMode"]
        log.info(f"Reference energy mode: {reference_energy_mode}")
    else:
        log.warning("No 'referenceEnergyMode' provided in YAML file, the default is auto")
        reference_energy_mode = 'auto'

    if (reference_energy_mode == 'auto'):
        reference_energy = None
    elif (reference_energy_mode == 'singleAtomEnergies'):
        if "referenceEnergies" in pacemaker_dict:
            reference_energy = pacemaker_dict["referenceEnergies"]
            log.info(f"Reference energies given as atomic energies")
        else:
            log.warning("No 'referenceEnergies' provided in YAML file, please specify it")
            raise ValueError("No 'referenceEnergies' provided in YAML file, please specify it")
    
    #Get relevant directories from directory_dict
    assert isinstance(directory_dict, dict)
    local_working_dir = directory_dict["local_working_dir"]
    local_dft_dir = directory_dict["local_dft_dir"]
    #Construct absolute paths for files
    pickle_file_path = os.path.join(local_dft_dir, 'dft_data.pckl.gzip')
    aimd_input_file_path = os.path.join(local_dft_dir, aimd_input_file)

    #Check wheter a pickle file already exists and skip the function if it does
    if os.path.exists(pickle_file_path):
        log.warning(f"There already is a file called 'dft_data.pckl.gzip' in {local_dft_dir}, skipping the rest")
        return

    #Check if it there is a completed calculation
    calc_done = calc_done_in_local_dir(local_dft_dir)
    if calc_done:
        pass
    else:
        log.warning(f"There is no completed calculation in {local_dft_dir}, cannot create a pickle file")
        raise RuntimeError(f"There is no completed calculation in {local_dft_dir}, cannot create a pickle file")

    #Choose the right function depending on dft_code
    symbols, energy_list, forces_list, atoms_list =\
    read_md_map[dft_code](aimd_input_file_path, local_dft_dir)

    #Reference energies
    if (reference_energy_mode == 'auto'):
        energies_corr = np.array(energy_list)
        ref_e= energies_corr.max()
        energies_corr = energies_corr - ref_e
        ref_e = np.full(shape=len(energy_list),fill_value=ref_e)
        #energies_corr.tolist()
    if (reference_energy_mode == 'singleAtomEnergies'):
        assert isinstance(reference_energy, dict)
        ref_e = 0 
        element_dict = Counter(symbols)
        for element in element_dict:
            ref_e += element_dict[element]*reference_energy[element]
        
    #Save all data in Data Frame
    data = {'energy': energy_list,
            'forces': forces_list,
            'ase_atoms': atoms_list,
            'energy_corrected': energies_corr,
            'reference_energy' : ref_e
            }

    df = pd.DataFrame(data)
    df.to_pickle(pickle_file_path, compression='gzip', protocol=4)
    log.info(f"Saved '{pickle_file_path}'")

    return

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

read_md_map = {
    "CPMD" :    read_cpmd_md 
}

read_scf_map = {
    "CPMD" :    read_cpmd_scf
}