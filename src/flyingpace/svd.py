import logging
import os 
import sys

from fabric import Connection
from invoke import run as local #To run local commands like rsync function

import flyingpace.dirmanager
import flyingpace.logginghelper
import flyingpace.sshutils
import flyingpace.fpio

from flyingpace.constants import implemented_dft_codes
from flyingpace.input import DataReader

log = logging.getLogger(__name__)

def run_aimd(cpu_connection: Connection, directory_dict: dict, InputData: DataReader):
    '''Start a AIMD run on the given CPU cluster, given a run script and a input file'''

    log.info(f"*** DFT RUN ***")

    dft_dict = InputData.dft_dict
    manager_dict = InputData.manager_dict

    #Read what is needed from dft_dict
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
        log.warning("No 'dftCode' provided in YAML file, please specify it")
        raise ValueError("No 'dftCode' provided in YAML file, please specify it")

    if "aimdRunScript" in dft_dict:
        run_script_aimd = dft_dict["aimdRunScript"]
        log.info(f"Run script for the AIMD calculation: {run_script_aimd}")
    else:
        log.warning("No 'aimdRunScript' provided in YAML file, please specify it")
        raise ValueError("No 'aimdRunScript' provided in YAML file, please specify it")
    
    #Read what is needed from manager_dict
    assert isinstance(manager_dict, dict)
    start_file = manager_dict["startFile"]
    start_file_type = manager_dict["startFileType"]

    #Get relevant directories from directory_dict
    assert isinstance(directory_dict, dict)
    local_working_dir = directory_dict["local_working_dir"]
    local_dft_dir = directory_dict["local_dft_dir"]
    if (cpu_connection != None):
        remote_dft_dir = directory_dict["remote_dft_dir"]

    if (start_file_type == "dft_input_file"):
        aimd_input_file = start_file
    elif (start_file_type == "lammps_data_file"):
        log.info(f"Start file is a lammps data file, will write AIMD input file automatically")
        flyingpace.fpio.write_aimd_input_file(directory_dict, InputData)
        aimd_input_file = 'INP0'

    #Construct absolute paths for files
    aimd_input_file_path = os.path.join(local_working_dir, aimd_input_file)
    run_script_aimd_path = os.path.join(local_working_dir, run_script_aimd)

    #Check if there is a completed or ongoing calculation in local_dft_dir or remote_dft_dir
    
    if (flyingpace.fpio.calc_done_in_local_dir(local_dft_dir)):
        log.warning(f"There already is a completed calculation in {local_dft_dir}")
        return
        
    elif (flyingpace.fpio.calc_ongoing_in_local_dir(local_dft_dir)):
        log.warning(f"There is an ongoing calculation in {local_dft_dir}, is now waiting for it to finish")
        flyingpace.fpio.wait_for_calc_done(local_dft_dir, cpu_connection)
        local(f"rm -rf {os.path.join(local_dft_dir, 'CALC_ONGOING')}")
        return

    if (cpu_connection != None):
        
        if (flyingpace.fpio.calc_done_in_remote_dir(remote_dft_dir, cpu_connection)):
            log.warning(f"There already is a completed calculation in {remote_dft_dir}")
            log.warning(f"Copying results to {local_dft_dir}")
            flyingpace.sshutils.get_dir_as_archive(local_dft_dir, remote_dft_dir, cpu_connection)
            return
        
        elif (flyingpace.fpio.calc_ongoing_in_remote_dir(remote_dft_dir, cpu_connection)):
            log.warning(f"There is an ongoing calculation in {remote_dft_dir}, is now waiting for it to finish")
            flyingpace.fpio.wait_for_calc_done(remote_dft_dir, cpu_connection)
            cpu_connection.run(f"rm -rf {os.path.join(remote_dft_dir, 'CALC_ONGOING')}", hide='both')
            flyingpace.sshutils.get_dir_as_archive(local_dft_dir, remote_dft_dir, cpu_connection)
            return
    
    #Check if all nessesary files exist
    if (os.path.exists(aimd_input_file_path) and\
    os.path.exists(run_script_aimd_path)):
        pass
    else: 
        log.warning(f"Check if {aimd_input_file_path} and {run_script_aimd_path} are in their place")
        raise RuntimeError(f"Check if {aimd_input_file_path} and {run_script_aimd_path} are in their place")
    
    #Check if the given input file matches the chosen DFT code
    flyingpace.fpio.check_dft_input_file_type(aimd_input_file_path, dft_code)

    #Check if the job type is a MD simulation
    jobtype = flyingpace.fpio.check_dft_job_type(aimd_input_file_path, dft_code)

    if (jobtype == 'md'):
        pass
    else:
        log.warning(f"The given input file {aimd_input_file_path} is not a MD run")
        raise ValueError(f"The given input file {aimd_input_file_path} is not a MD run")

    #Add 'touch CALC_DONE' to run script if not there
    flyingpace.fpio.check_for_calc_done_in_script(run_script_aimd_path)

    #Copy run script and cpmd input to local_dft_dir
    local(f"cp {aimd_input_file_path} {run_script_aimd_path} {local_dft_dir}")
    
    #Copy local_dft_dir to remote_dft_dir
    if (cpu_connection != None):
        flyingpace.sshutils.put_dir_as_archive(local_dft_dir, remote_dft_dir, cpu_connection)

    #Start AIMD run
    if (cpu_connection == None):
        log.info("Starting AIMD run, is now waiting for it to finish")
        local(f"touch {os.path.join(local_dft_dir, 'CALC_ONGOING')}")
        local(f"cd {local_dft_dir} && sbatch {os.path.basename(run_script_aimd_path)}")
        flyingpace.fpio.wait_for_calc_done(local_dft_dir, cpu_connection)
        local(f"rm -rf {os.path.join(local_dft_dir, 'CALC_ONGOING')}")
    elif (cpu_connection != None):
        log.info("Starting AIMD run, is now waiting for it to finish")
        cpu_connection.run(f"touch {os.path.join(remote_dft_dir, 'CALC_ONGOING')}", hide='both')
        with cpu_connection.cd(remote_dft_dir):
            cpu_connection.run(f"sbatch {os.path.basename(run_script_aimd_path)}", hide='both')
        flyingpace.fpio.wait_for_calc_done(remote_dft_dir, cpu_connection)
        cpu_connection.run(f"rm -rf {os.path.join(local_dft_dir, 'CALC_ONGOING')}", hide='both')
    log.info("AIMD run has finished")

    #Copy results to local folder
    if (cpu_connection != None):
        flyingpace.sshutils.get_dir_as_archive(local_dft_dir, remote_dft_dir, cpu_connection)

    return

def run_scf_from_exploration(cpu_connection: Connection, directory_dict: dict, InputData: DataReader):
    '''
    Start a series of scf calculations from a pickle file of extrapolative structures
    called 'extrapolative_structures.pckl.gzip' taken from the previous' generation
    local_exploration_dir
    '''

    log.info(f"*** DFT RUN ***")

    dft_dict = InputData.dft_dict

    #Read what is needed from dft_dict
    assert isinstance(dft_dict, dict)
    if "dftCode" in dft_dict:
        dft_code = dft_dict["dftCode"]
        log.info("The DFT code used is:" + dft_code)
        if (dft_code in implemented_dft_codes):
            pass
        else:
            log.warning("The chosen DFT code is not implemented")
            raise NotImplementedError("The chosen DFT code is not implemented")
    else:
        log.warning("No 'dftCode' provided in YAML file, please specify it")
        raise ValueError("No 'dftCode' provided in YAML file, please specify it")

    if "scfRunScript" in dft_dict:
        run_script_scf = dft_dict["scfRunScript"]
        log.info(f"Run script for the SCF calculations: {run_script_scf}")
    else:
        log.warning("No 'scfRunScript' provided in YAML file, please specify it")
        raise ValueError("No 'scfRunScript' provided in YAML file, please specify it")
    
    #Get relevant directories from directory_dict
    assert isinstance(directory_dict, dict)
    local_working_dir = directory_dict["local_working_dir"]
    local_dft_dir = directory_dict["local_dft_dir"]
    local_scf_results_dir = os.path.join(local_dft_dir, "scf_results")
    if (cpu_connection != None):
        remote_dft_dir = directory_dict["remote_dft_dir"]
        remote_scf_results_dir = os.path.join(remote_dft_dir, "scf_results")
    #Construct absolute paths for files
    run_script_scf_path = os.path.join(local_working_dir, run_script_scf)

    #Check if there is a completed or ongoing calculation in local_dft_dir or remote_dft_dir
    if (flyingpace.fpio.calc_done_in_local_dir(local_dft_dir)):
        log.warning(f"There already is a completed calculation in {local_dft_dir}")
        flyingpace.sshutils.gather_files(local_dft_dir, local_scf_results_dir, "scf", ["OUT"], cpu_connection)
        return
        
    elif (flyingpace.fpio.calc_ongoing_in_local_dir(local_dft_dir)):
        log.warning(f"There is an ongoing calculation in {local_dft_dir}, is now waiting for it to finish")
        flyingpace.fpio.wait_for_calc_done(local_dft_dir, cpu_connection)
        local(f"rm -rf {os.path.join(local_dft_dir, 'CALC_ONGOING')}")
        flyingpace.sshutils.gather_files(local_dft_dir, local_scf_results_dir, "scf", ["OUT"], cpu_connection)
        return

    if (cpu_connection != None):
        
        if (flyingpace.fpio.calc_done_in_remote_dir(remote_dft_dir, cpu_connection)):
            log.warning(f"There already is a completed calculation in {remote_dft_dir}")
            log.warning(f"Copying results to {local_dft_dir}")
            flyingpace.sshutils.gather_files(remote_dft_dir, remote_scf_results_dir, "scf", ["OUT"], cpu_connection)
            flyingpace.sshutils.get_dir_as_archive(local_scf_results_dir, remote_scf_results_dir, cpu_connection)
            return
        
        elif (flyingpace.fpio.calc_ongoing_in_remote_dir(remote_dft_dir, cpu_connection)):
            log.warning(f"There is an ongoing calculation in {remote_dft_dir}, is now waiting for it to finish")
            flyingpace.fpio.wait_for_calc_done(remote_dft_dir, cpu_connection)
            cpu_connection.run(f"rm -rf {os.path.join(remote_dft_dir, 'CALC_ONGOING')}", hide='both')
            flyingpace.sshutils.gather_files(remote_dft_dir, remote_scf_results_dir, "scf", ["OUT"], cpu_connection)
            flyingpace.sshutils.get_dir_as_archive(local_scf_results_dir, remote_scf_results_dir, cpu_connection)
            return
    
    #Check if all nessesary files exist
    if (os.path.exists(run_script_scf_path)):
        pass
    else: 
        log.warning(f"Check if {run_script_scf_path} is in its place")
        raise RuntimeError(f"Check if {run_script_scf_path} is in its place")
    
    #Add 'touch CALC_DONE' to run script if not there
    flyingpace.fpio.check_for_calc_done_in_script(run_script_scf_path)
    
    #Copy 'extrapolative_structures.pckl.gzip' from and runs script to local_dft_dir
    local(f"cp {run_script_scf_path} {local_dft_dir}")

    #Prepare all a folder and an input file for each scf calculation
    flyingpace.fpio.prepare_scf_calcs_from_pickle(directory_dict, InputData)

    #Copy local_dft_dir to remote_dft_dir
    if (cpu_connection != None):
        flyingpace.sshutils.put_dir_as_archive(local_dft_dir, remote_dft_dir, cpu_connection)

    #Start SCF runs
    if (cpu_connection == None):
        log.info("Starting SCF runs, is now waiting for them to finish")
        local(f"touch {os.path.join(local_dft_dir, 'CALC_ONGOING')}")
        local(f"cd {local_dft_dir} && sbatch {os.path.basename(run_script_scf_path)}")
        flyingpace.fpio.wait_for_calc_done(local_dft_dir, cpu_connection)
        local(f"rm -rf {os.path.join(local_dft_dir, 'CALC_ONGOING')}")
    elif (cpu_connection != None):
        log.info("Starting SCF runs, is now waiting for them to finish")
        cpu_connection.run(f"touch {os.path.join(remote_dft_dir, 'CALC_ONGOING')}", hide='both')
        with cpu_connection.cd(remote_dft_dir):
            cpu_connection.run(f"sbatch {os.path.basename(run_script_scf_path)}", hide='both')
        flyingpace.fpio.wait_for_calc_done(remote_dft_dir, cpu_connection)
        cpu_connection.run(f"rm -rf {os.path.join(local_dft_dir, 'CALC_ONGOING')}", hide='both')
    log.info("SCF runs have finished")

    flyingpace.sshutils.gather_files(remote_dft_dir, remote_scf_results_dir, "scf", ["OUT"], cpu_connection)
    
    if (cpu_connection != None):
        flyingpace.sshutils.get_dir_as_archive(local_scf_results_dir, remote_scf_results_dir, cpu_connection)
   





    
    

    


    
