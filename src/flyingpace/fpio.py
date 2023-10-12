import ase.io
import glob
import logging
import numpy as np
import os
import pandas as pd
import random
import sys
import time

from ase import Atoms
from ase.io.lammpsdata import read_lammps_data
from ase.io.lammpsrun import read_lammps_dump_text
from ase.units import Bohr, Hartree, Ang, eV
from collections import Counter
from fabric import Connection
from patchwork.files import exists

import flyingpace.cpmd_io
import flyingpace.dirmanager
import flyingpace.logginghelper

from flyingpace.constants import implemented_dft_codes
from flyingpace.input import DataReader

log = logging.getLogger(__name__)

template_directory = os.path.join(os.path.dirname(__file__), 'templates')

#########################################################################################
##                                                                                     ##
##                             Functions for general I/O                               ##
##                                                                                     ##
#########################################################################################

def check_start_file_type(start_file: str):

    if start_file.endswith(".data"):
        start_file_type = 'lammps_datafile'
    elif start_file.endswith(".yaml"):
        start_file_type = "potential_file"
    elif start_file.endswith(".pckl.gzip"):
        start_file_type = "pickle_file"
    else:
        start_file_type = "dft_input_file"
    
    return start_file_type

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
        log.warning("No 'testSize' provided in input file, the default is 0")
        test_size = 0

    if "numberOfFunctions" in pacemaker_dict:
        num_functions = pacemaker_dict["numberOfFunctions"]
        log.info(f"Number of functions for pacemaker run: {num_functions}")
    else:
        log.warning("No 'numberOfFunctions' provided in input file, the default is 700")
        num_functions = 700

    if "cutoff" in pacemaker_dict:
        cutoff = pacemaker_dict["cutoff"]
        log.info(f"Cutoff for pacemaker run: {cutoff}")
    else:
        log.warning("No 'cutoff' provided in input file, the default is 7.0")
        cutoff = 7.0

    if "weighting" in pacemaker_dict:
        weighting_inp = pacemaker_dict["weighting"]
        log.info(f"Weighting for pacemaker run: {weighting_inp}")
    else:
        log.warning("No 'weighting' provided in input file, the default is 'uniform'")
        weighting_inp = 'uniform'

    if "kappa" in pacemaker_dict:
        kappa = pacemaker_dict["kappa"]
        log.info(f"Kappa for pacemaker run: {kappa}")
    else:
        log.warning("No 'kappa' provided in input file, the default is 0.3")
        weighting = 0.3
    
    if "maxNumIterations" in pacemaker_dict:
        max_num_iter = pacemaker_dict["maxNumIterations"]
        log.info(f"Maximum number of iterations for pacemaker run: {max_num_iter}")
    else:
        log.warning("No 'maxNumIterations' provided in input file, the default is 500")
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

def generate_lammps_input(InputData: DataReader):
    '''
    Generate a inputfile for lammps based on data from the master input file
    and save it in the current generation local_exploration_dir
    '''

    exploration_dict = InputData.exploration_dict
    manager_dict = InputData.manager_dict

    directory_dict = InputData.directory_dict

    #Read what is needed from pacemaker_dict
    assert isinstance(exploration_dict, dict)
    if "explorationParams" in exploration_dict:
        input_data_dict = exploration_dict["explorationParams"]
    else:
        log.warning("No 'explorationParams' provided in input file, please provide it")
        raise ValueError("No 'explorationParams' provided in input file, please provide it")

    #Read what is needed from directory_dict
    assert isinstance(directory_dict, dict)
    local_exploration_dir = directory_dict["local_exploration_dir"]
    input_file_path = os.path.join(local_exploration_dir, 'INP-lammps')

    input_options = {
    "datafile" : "read_data ${{submitdir}}/{datafile}\n",
    "thermo" : "thermo {thermo}\n",
    "timestep" : "timestep {timestep}\n",
    "explorationStyle" : {
        
        "stop" : '\n#dump extrapolative structures if c_max_pace_gamma > {lowerGamma},\
 skip otherwise, check every {gammaStride} steps\n\
variable dump_skip equal "c_max_pace_gamma < {lowerGamma}"\n\
dump pace_dump all custom {gammaStride} extrapolative_structures.dump id type x y z f_pace_gamma\n\
dump_modify pace_dump skip v_dump_skip\n\
\n#stop simulation if maximum extrapolation grade exceeds {upperGamma}\n\
variable max_pace_gamma equal c_max_pace_gamma\n\
fix extreme_extrapolation all halt {gammaStride} v_max_pace_gamma > {upperGamma}\n\n',
        
        "noStop" : '\n#dump extrapolative structures if {upperGamma} > c_max_pace_gamma > {lowerGamma},\
 skip otherwise, check every {gammaStride} steps\n\
variable dump_skip equal "(c_max_pace_gamma < {lowerGamma}) || (c_max_pace_gamma > {upperGamma})"\n\
dump pace_dump all custom {gammaStride} extrapolative_structures.dump id type x y z f_pace_gamma\n\
dump_modify pace_dump skip v_dump_skip\n'

    },
    
    "trjStep" : '\n#dump trajectory every {trjStep} steps\n\
dump dmp_trj all custom {trjStep} trj.dmp id element x y z f_pace_gamma\n\
dump_modify dmp_trj format line "%d %s %20.5g %20.15g %20.15g %20.15g" element {elementList} first no sort id\n',
    
    "runType" : {
        
        'NVT' : '\nvelocity all create {startTemp} 67 dist gaussian\n\
fix 6 all nvt temp {startTemp} {endTemp} $(100.0*dt)\n\
run {steps}',
        
        'NPT' : '\nvelocity all create {startTemp} 67 dist gaussian\n\
fix 6 all npt temp {startTemp} {endTemp} $(100.0*dt) iso {startPress} {startPress} $(1000.0*dt)\n\
run {steps}',
    },           
}

    input = ""
    input += "#-----------------------------------\n\
#Automatically generated input file\n\
#-----------------------------------\n\n"

    #Start with the general section

    input += "units metal\natom_style atomic\nboundary p p p\n"

    #Start with data section 

    input += "\n#-----------------------------------\n\
#Box and regions\n\
#-----------------------------------\n\n"

    datafile_path = manager_dict["dataFilePath"]
    datafile = os.path.basename(datafile_path)
    input_data_dict["datafile"] = datafile
            
    input += input_options["datafile"]
        
    #Start with the potential section
    
    input += "\n#-----------------------------------\n\
#Potentials\n\
#-----------------------------------\n\n"

    input += "pair_style pace/extrapolation\n"

    #Read elementlist from manager_dict
    input_data_dict["elementList"] = manager_dict["elementList"]
    
    input += "pair_coeff * * ${{submitdir}}/output_potential.yaml\
 ${{submitdir}}/output_potential.asi {elementList}\n"

    #Start with the simulation section
    
    input += "\n#-----------------------------------\n\
#Simulation\n\
#-----------------------------------\n\n"

    #Print thermodynamic information
    if not "thermo" in input_data_dict:
        input_data_dict["thermo"] = 100
    input += input_options["thermo"]
    input += "thermo_style custom step temp press vol etotal ke pe\n"
    
    if not "timestep" in input_data_dict:
        input_data_dict["timestep"] = 5.0e-4
    input += input_options["timestep"]

    #Extrapolation grade assessment    
    if not "gammaStride" in input_data_dict:
        input_data_dict["gammaStride"] = 10
    if not "lowerGamma" in input_data_dict:
        input_data_dict["lowerGamma"] = 5
    if not "upperGamma" in input_data_dict:
        input_data_dict["upperGamma"] = 25
    
    input += "\n#compute per-atom extrapolation grade every {gammaStride} steps\n"
    input += "fix pace_gamma all pair {gammaStride} pace/extrapolation gamma 1\n"
    input += "\n#compute maximum extrapolation grade over complete structure\n"
    input += "compute max_pace_gamma all reduce max f_pace_gamma\n"

    #Exploration style
    if "explorationStyle" in input_data_dict:
        if input_data_dict["explorationStyle"] in input_options["explorationStyle"]: 
            input += input_options["explorationStyle"][input_data_dict["explorationStyle"]]
        else: 
            log.warning("'explorationStyle' type in 'explorationParams' is not known")
            raise ValueError("'explorationStyle' type in 'explorationParams' is not known")
    else:
        input_data_dict["explorationStyle"] = "stop"
        input += input_options["explorationStyle"][input_data_dict["explorationStyle"]]
    
    #Dump trajectory
    if not "trjStep" in input_data_dict:
        input_data_dict["trjStep"] = 100
    input += input_options["trjStep"]

    #Temperature for run
    if "temp" in input_data_dict:
        input_data_dict["startTemp"] = input_data_dict["temp"]
        input_data_dict["endTemp"] = input_data_dict["temp"]
    elif "tempRamp" in input_data_dict:
        input_data_dict["startTemp"] = input_data_dict["tempRamp"].split()[0]
        input_data_dict["endTemp"] = input_data_dict["tempRamp"].split()[1]
    else: 
        input_data_dict["temp"] = 400.0
        input_data_dict["startTemp"] = input_data_dict["temp"]
        input_data_dict["endTemp"] = input_data_dict["temp"]

    #Pressure for NPT run
    if (input_data_dict["runType"] == "NPT"):
        if "press" in input_data_dict:
            input_data_dict["startPress"] = input_data_dict["press"]
            input_data_dict["endPress"] = input_data_dict["press"]
        elif "pressRamp" in input_data_dict:
            input_data_dict["startPress"] = input_data_dict["tempPress"].split()[0]
            input_data_dict["endPress"] = input_data_dict["tempPress"].split()[1]
        

    #Type of run
    input += "\n#Start run"
    if "runType" in input_data_dict:
        if input_data_dict["runType"] in input_options["runType"]:
            input += input_options["runType"][input_data_dict["runType"]]
        else: 
            log.warning("'runType' type in 'explorationParams' is not known")
            raise ValueError("'runType' type in 'explorationParams' is not known")
    else:
        input_data_dict["runType"] = "NVT"
        input += input_options["runType"][input_data_dict["runType"]]

    #Set values in template string
    input = input.format(**input_data_dict)

    with open(input_file_path, "w") as f:
        print(input, file=f)

    log.info(f"Input file is written into {input_file_path}")

    return


def element_list_from_datafile(InputData: DataReader):

    manager_dict = InputData.manager_dict

    datafile_path = manager_dict["dataFilePath"]

    with open(datafile_path, 'r') as f:
        datafile = f.readlines()

    for idx, line in enumerate(datafile):
        if "atom types" in line:
            num_types = int(line.split()[0])
        if "Masses" in line:
            masses_idx = idx

    element_list = []        
    for i in range(num_types):
        element_list.append(datafile[i+masses_idx+2].split()[-1])

    element_list = " ".join(element_list)

    InputData.change_data("manager_dict", "elementList", element_list)

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

def write_aimd_input_file(InputData: DataReader):
    '''
    Writes a AIMD input file from the current DataFile and
    writes it to local_working_dir as 'INP0'
    '''

    dft_dict = InputData.dft_dict
    manager_dict = InputData.manager_dict

    directory_dict = InputData.directory_dict

    #Read what is needed from dft_dict
    assert isinstance(dft_dict, dict)
    if "dftCode" in dft_dict:
        dft_code = dft_dict["dftCode"]
        if (dft_code in implemented_dft_codes):
            pass
        else:
            log.warning("The chosen DFT code is not implemented")
            raise NotImplementedError("The chosen DFT code is not implemented")
        
    assert isinstance(manager_dict, dict)
    if "dataFilePath" in manager_dict:
        datafile_path = manager_dict["dataFilePath"]
    else:
        log.warning("'DataFilePath' not yet assigned")
        raise ValueError("'DataFilePath' not yet assigned")
    
    #Overwrite value to ensure scf calculations
    dft_dict["dftParams"]["calculation"] = "md"
    
    #Get relevant directories from directory_dict
    assert isinstance(directory_dict, dict)
    local_working_dir = directory_dict["local_working_dir"]

    with open(datafile_path) as f:
        structure = read_lammps_data(f)

    scf_input_file_path = os.path.join(local_working_dir, "INP0")
    write_input_map[dft_code](structure, scf_input_file_path, dft_dict)

    log.info(f"Read {os.path.basename(datafile_path)} and write DFT input file 'INP0'")

    return

def datafile_from_dft_input(InputData: DataReader):
    '''
    Reads the 'startFile' (assumes it is a DFT input file) and writes
    a lammps datafile to local_working_dir named after 'systemName'
    '''

    dft_dict = InputData.dft_dict
    manager_dict = InputData.manager_dict

    directory_dict = InputData.directory_dict

    assert isinstance(dft_dict, dict)
    if "dftCode" in dft_dict:
        dft_code = dft_dict["dftCode"]
        log.info(f"The DFT code used is: {dft_code}")
        if (dft_code in implemented_dft_codes):
            pass
        else:
            log.warning("The chosen DFT code is not implemented")
            raise NotImplementedError("The chosen DFT code is not implemented")
    else:
        log.warning("No 'dftCode' provided in input file, please specify it")
        raise ValueError("No 'dftCode' provided in input file, please specify it")
        
    assert isinstance(manager_dict, dict)
    start_file = manager_dict["startFile"]
    system_name = manager_dict["systemName"]

    #Get relevant directories from directory_dict
    assert isinstance(directory_dict, dict)
    local_working_dir = directory_dict["local_working_dir"]

    datafile_path = os.path.join(local_working_dir, f"{system_name}.data")
    start_file_path = os.path.join(local_working_dir, start_file)

    structure = read_input_map[dft_code](start_file_path)

    ase.io.write(datafile_path, images=structure, format='lammps-data', masses=True)

    log.info(f"Read DFT input file and write lammps datafile '{system_name}.data'")

    return


#########################################################################################
##                                                                                     ##
##                          Functions for handling pickle files                        ##
##                                                                                     ##
#########################################################################################

def merge_pickle(InputData: DataReader):
    '''
    Merges the pickle file 'dft_data.pckl.gzip' from prev_local_dft_dir
    with 'new_dft_data.pckl.gzip' from local_dft_dir and saves the resulting 
    pickle file as 'dft_data.pckl.gzip' in local_dft_dir
    'dft_data.pckl.gzip' in prev_local_dft_dir must contain a column containing
    the reference energy
    '''

    directory_dict = InputData.directory_dict

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
    log.info(f"Merging pickle files {new_pickle_file_path} and {prev_pickle_file_path} to {pickle_file_path}")

    return

def update_datafile(mode:str, InputData: DataReader):
    '''
    Reads the pickle file 'dft_data.pckl.gzip' from local_train_dir, 
    chooses a random structure and saves it as a lammps data file
    in local_train_dir and updates its path to manager_dict["dataFilePath"]
    '''

    directory_dict = InputData.directory_dict
    local_train_dir = directory_dict["local_train_dir"]
    pickle_file_path = os.path.join(local_train_dir, "dft_data.pckl.gzip")

    if (os.path.exists(pickle_file_path)):
        pass
    else:
        log.warning(f"The pickle file {pickle_file_path} does not exist!")
        raise RuntimeError(f"The pickle file {pickle_file_path} does not exist!")
    
    df = pd.read_pickle(pickle_file_path, compression="gzip")

    if (mode == "random"):
        datafile_path = os.path.join(local_train_dir, "random_structure.data")
        num_structures = df.shape[0]
        structure_id = random.randint(0,num_structures-1)
        structure = df.loc[structure_id,"ase_atoms"]

    if (mode == "last"):
        datafile_path = os.path.join(local_train_dir, "last_structure.data")
        num_structures = df.shape[0]
        structure = df.loc[num_structures-1,"ase_atoms"]

    ase.io.write(datafile_path, images=structure, format="lammps-data", masses=True)

    InputData.change_data("manager_dict", "dataFilePath", datafile_path)

    log.info(f"Updated datafile to {datafile_path}")

    return

def extrapolative_dump_to_pickle(InputData: DataReader):
    '''
    Read a lammps dump file called "extrapolative_structures.dump" from this 
    generations local_exploration_dir and saves the structures in a pickle file
    in local_exploration_dir
    '''

    log.info(f"Gathering extrapolative structures in 'extrapolative_structures.pckl.gzip'")

    manager_dict = InputData.manager_dict

    directory_dict = InputData.directory_dict

    #Read elementlist from manager_dict
    element_list = manager_dict["elementList"]
    
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

def prepare_scf_calcs_from_pickle(InputData: DataReader):
    '''
    Reads 'extrapolative_structures.pckl.gzip' from prev_local_exploration_dir
    and constructs CPMD scf input files from them in local_dft_dir, one folder for each calculation
    '''

    dft_dict = InputData.dft_dict

    directory_dict = InputData.directory_dict

    #Read what is needed from dft_dict
    assert isinstance(dft_dict, dict)
    if "dftCode" in dft_dict:
        dft_code = dft_dict["dftCode"]
        if (dft_code in implemented_dft_codes):
            pass
        else:
            log.warning("The chosen DFT code is not implemented")
            raise NotImplementedError("The chosen DFT code is not implemented")
        
    if "maxScfRuns" in dft_dict:
        max_scf_runs = dft_dict["maxScfRuns"]
    else:
        log.warning("No 'maxScfRuns' provided in input file, the default is 100")
        max_scf_runs = 100

    #Overwrite value to ensure scf calculations
    dft_dict["dftParams"]["calculation"] = "scf"
    
    #Get relevant directories from directory_dict
    assert isinstance(directory_dict, dict)
    local_dft_dir = directory_dict["local_dft_dir"]
    prev_local_exploration_dir = flyingpace.dirmanager.get_prev_path(directory_dict["local_exploration_dir"])
    #Construct absolute paths for files
    pickel_file_path = os.path.join(prev_local_exploration_dir, "extrapolative_structures.pckl.gzip")

    data = pd.read_pickle(pickel_file_path, compression='gzip')

    if (data.shape[0] == 0):
        log.warning(f"No extrapolative structures found!")
        raise RuntimeError(f"No extrapolative structures found!")

    structures = data.loc[:,'ase_atoms']

    log.info(f"Prepare scf calculations in {local_dft_dir}")

    scf_dir_num = 1
    for i in structures:

        scf_dir = os.path.join(local_dft_dir, f"scf.{str(scf_dir_num)}")
        scf_input_file_path = os.path.join(scf_dir, "INP")
        os.mkdir(scf_dir)
        write_input_map[dft_code](i, scf_input_file_path, dft_dict)

        scf_dir_num += 1
        if (scf_dir_num > max_scf_runs):
            break

    return

def scfs_to_pickle(InputData: DataReader):
    '''
    Reads a number of SCF output files in the directory output_file_dir positioned in 
    local_dft_dir. The filenames follow the format of OUT.* with * being a 
    consecutive numbering. Saves the data as 'new_dft_data.pckl.gzip' in local_dft_dir
    without the corrected_energy column
    '''

    dft_dict = InputData.dft_dict

    directory_dict = InputData.directory_dict

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
        log.warning("No 'dftCode' provided in input file, please specify it")
        raise ValueError("No 'dftCode' provided in input file, please specify it")

    assert isinstance(directory_dict, dict)
    local_dft_dir = directory_dict["local_dft_dir"]

    output_file_dir = "scf_results"
    file_pattern = "OUT"
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
    
    log.info(f"Gathering results of scf calculations in {pickle_file_path}")

    df.to_pickle(pickle_file_path, compression='gzip', protocol=4)
    log.info(f"Saved '{pickle_file_path}'")

def aimd_to_pickle(InputData: DataReader):
    '''
    Reads output data from a AIMD run in local_dft_dir 
    and saves it in local_dft_dir as 'dft_data.pckl.gzip'
    '''

    log.info(f"Gathering data from AIMD run")

    dft_dict = InputData.dft_dict
    manager_dict = InputData.manager_dict
    pacemaker_dict = InputData.pacemaker_dict

    directory_dict = InputData.directory_dict

    #Get relevant directories from directory_dict
    assert isinstance(directory_dict, dict)
    local_dft_dir = directory_dict["local_dft_dir"]
    #Construct absolute paths for files
    pickle_file_path = os.path.join(local_dft_dir, 'dft_data.pckl.gzip')

    #Check wheter a pickle file already exists and skip the function if it does
    if os.path.exists(pickle_file_path):
        log.warning(f"There already is a file called 'dft_data.pckl.gzip' in {local_dft_dir}, skipping the rest")
        return

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
        log.warning("No 'dftCode' provided in input file, please specify it")
        raise ValueError("No 'dftCode' provided in input file, please specify it")
    
    #Read what is needed from pacemaker_dict
    assert isinstance(pacemaker_dict, dict)
    if "referenceEnergyMode" in pacemaker_dict:
        reference_energy_mode = pacemaker_dict["referenceEnergyMode"]
        log.info(f"Reference energy mode: {reference_energy_mode}")
    else:
        log.warning("No 'referenceEnergyMode' provided in input file, the default is auto")
        reference_energy_mode = 'auto'

    if (reference_energy_mode == 'auto'):
        reference_energy = None
    elif (reference_energy_mode == 'singleAtomEnergies'):
        if "referenceEnergies" in pacemaker_dict:
            reference_energy = pacemaker_dict["referenceEnergies"]
            log.info(f"Reference energies given as atomic energies")
        else:
            log.warning("No 'referenceEnergies' provided in input file, please specify it")
            raise ValueError("No 'referenceEnergies' provided in input file, please specify it")

    #Construct absolute paths for files  
    aimd_input_file_path = manager_dict["startInputFilePath"]

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

read_md_map = {
    "CPMD" :    flyingpace.cpmd_io.read_cpmd_md 
}

read_scf_map = {
    "CPMD" :    flyingpace.cpmd_io.read_cpmd_scf
}

write_input_map = {
    "CPMD" :    flyingpace.cpmd_io.write_cpmd_input
}

read_input_map = {
    "CPMD" :    flyingpace.cpmd_io.read_cpmd_input
}

