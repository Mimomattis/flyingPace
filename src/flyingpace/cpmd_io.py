import logging

import pandas as pd
import numpy as np

import flyingpace.logginghelper

from ase import Atoms
from collections import Counter

log = logging.getLogger(__name__)

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
            'aimd' : '    MOLECULAR DYNAMICS BO\n',
        },

        'restart_mode' : {
            'from_scratch' : '    INITIALIZE WAVEFUNCTION RANDOM\n',
        },

        'max_iter' : '    MAXITER\n        {max_iter}\n',
        'conv_orbital' : '    CONVERGENCE ORBITALS\n        {conv_orbital}\n',
        'spline_points' : '    SPLINE POINTS\n        {spline_points}\n',
        'functional' :{
            'pbe_sol' : '    GRADIENT CORRECTION PBESX PBESC\n',
        },
    
        'pw_cutoff' : '    CUTOFF\n        {pw_cutoff}\n',   
    }

    cpmd_technical_block = "    ODIIS NO_RESET=-1\n        10\n    MEMORY BIG\n    REAL SPACE WFN KEEP\n\
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
            log.warning("'calculation' type in dft options is not known")
            raise ValueError("'calculation' type in dft options is not known")
    else:
        log.warning("'calculation' type is not provided in 'dftParams' section, please specify it")
        raise ValueError("'calculation' type is not provided in 'dftParams' section, please specify it")
        
    if "restart_mode" in input_data_dict:
        if input_data_dict["restart_mode"] in input_options["restart_mode"]:
            input += input_options["restart_mode"][input_data_dict["restart_mode"]]
        else:
            input += input_options["restart_mode"]["from_scratch"]
    else: 
        input += input_options["restart_mode"]["from_scratch"]

    if "max_iter" in input_data_dict:
        input += input_options["max_iter"]
    else:
        input_data_dict["max_iter"] = 100
        input += input_options["max_iter"]
    
    if "conv_orbital" in input_data_dict:
        input += input_options["conv_orbital"]
    else:
        input_data_dict["conv_orbital"] = '1.d-6'
        input += input_options["conv_orbital"]
    
    if "spline_points" in input_data_dict:
        input += input_options["spline_points"]
    else:
        input_data_dict["spline_points"] = 5000
        input += input_options["spline_points"]

        
    input += cpmd_technical_block
    input += "&END\n\n"

    #Start with &SYSTEM section

    input += "&SYSTEM\n    ANGSTROM\n    CELL VECTORS\n"

    #Get cell vectors as strings from the structure
    cell = []
    for i in range(3):
        cell.append("{:10.8f} {:10.8f} {:10.8f}".format(structure.get_cell()[i][0], structure.get_cell()[i][1], structure.get_cell()[i][2]))
        input += f"        {cell[i]}\n"
    
    if "pw_cutoff" in input_data_dict:
        input += input_options["pw_cutoff"]
    else:
        input_data_dict["pw_cutoff"] = 30
        input += input_options["pw_cutoff"]

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