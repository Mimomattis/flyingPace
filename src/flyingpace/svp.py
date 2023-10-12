import logging
import os

import pandas as pd

from fabric import Connection
from invoke import run as local #To run local commands like rsync function
from patchwork.files import exists

import flyingpace.fpio
import flyingpace.logginghelper
import flyingpace.sshutils
import flyingpace.dirmanager

from flyingpace.input import DataReader

log = logging.getLogger(__name__) 

def run_pacemaker(InputData: DataReader, **kwargs):
    '''
    Start a pacemaker run on the given GPU cluster, given a run script
    '''

    log.info(f"*** PACEMAKER RUN ***")

    pacemaker_dict = InputData.pacemaker_dict

    directory_dict = InputData.directory_dict

    gpu_connection = InputData.gpu_connection

    #Read what is needed from pacemaker_dict
    assert isinstance(pacemaker_dict, dict)
    if "pacemakerRunScript" in pacemaker_dict:
        run_script_pacemaker = pacemaker_dict["pacemakerRunScript"]
        log.info(f"Run script for pacemaker run: {run_script_pacemaker}")
    else:
        log.warning("No 'pacemakerRunScript' provided in input file, please specify it")
        raise ValueError("No 'pacemakerRunScript' provided in input file, please specify it")

    #Read what is needed from directory_dict
    assert isinstance(directory_dict, dict)
    local_working_dir = directory_dict["local_working_dir"]
    local_train_dir = directory_dict["local_train_dir"]
    local_dft_dir = directory_dict["local_dft_dir"]
    if (gpu_connection != None):
        remote_train_dir = directory_dict["remote_train_dir"]
    #Construct absolute paths for files
    dataset_file = 'dft_data.pckl.gzip'
    dataset_file_path = os.path.join(local_dft_dir, dataset_file)
    run_script_pacemaker_path = os.path.join(local_working_dir, run_script_pacemaker)

    #Read variables in case of a restart, old potential is then reused

    if "restart" in kwargs:
        restart = kwargs["restart"]
    else:
        restart = False

    if restart:
        prev_local_train_dir = flyingpace.dirmanager.get_prev_path(local_train_dir)
    
    #Check if there is a completed or ongoing calculation in local_train_dir or remote_train_dir

    if (flyingpace.fpio.calc_done_in_local_dir(local_train_dir)):
        log.warning(f"There already is a completed calculation in {local_train_dir}")
        run_activeset(InputData)
        return
        
    elif (flyingpace.fpio.calc_ongoing_in_local_dir(local_train_dir)):
        log.warning(f"There is an ongoing calculation in {local_train_dir}, is now waiting for it to finish")
        flyingpace.fpio.wait_for_calc_done(local_train_dir, gpu_connection)
        local(f"rm -rf {os.path.join(local_train_dir, 'CALC_ONGOING')}")
        run_activeset(InputData)
        return

    if (gpu_connection != None):
        
        if (flyingpace.fpio.calc_done_in_remote_dir(remote_train_dir, gpu_connection)):
            log.warning(f"There already is a completed calculation in {remote_train_dir}")
            log.warning(f"Copying results to {local_train_dir}")
            run_activeset(InputData)
            flyingpace.sshutils.get_dir_as_archive(local_train_dir, remote_train_dir, gpu_connection)
            return
        
        elif (flyingpace.fpio.calc_ongoing_in_remote_dir(remote_train_dir, gpu_connection)):
            log.warning(f"There is an ongoing calculation in {remote_train_dir}, is now waiting for it to finish")
            flyingpace.fpio.wait_for_calc_done(remote_train_dir, gpu_connection)
            gpu_connection.run(f"rm -rf {os.path.join(remote_train_dir, 'CALC_ONGOING')}", hide='both')
            run_activeset(InputData)
            flyingpace.sshutils.get_dir_as_archive(local_train_dir, remote_train_dir, gpu_connection)
            return
    
    #Check if all nessesary files exist
    if (os.path.exists(dataset_file_path) and\
    os.path.exists(run_script_pacemaker_path)):
        pass
    else: 
        log.warning(f"Check if {dataset_file_path} and {run_script_pacemaker_path} are in their place")
        raise RuntimeError(f"Check if {dataset_file_path} and {run_script_pacemaker_path} are in their place")

    #Add 'touch CALC_DONE' to run script if not there
    flyingpace.fpio.check_for_calc_done_in_script(run_script_pacemaker_path)

    #Copy run script to local_train_dir
    local(f"cp {run_script_pacemaker_path} {local_train_dir}")

    #Copy dataset to local_train_dir
    local(f"cp {dataset_file_path} {local_train_dir}")

    #In case of restart, copy the old potential file
    if restart:
        local(f"cp {os.path.join(prev_local_train_dir, 'output_potential.yaml')} {os.path.join(local_train_dir, 'cont.yaml')}")
    
    #Create input file and save it to local_train_dir
    flyingpace.fpio.generate_pace_input(dataset_file_path, directory_dict, InputData)

    #Copy local_train_dir to remote_train_dir
    if (gpu_connection != None):
        flyingpace.sshutils.put_dir_as_archive(local_train_dir, remote_train_dir, gpu_connection)

    #Start pacemaker run
    if (gpu_connection == None):
        log.info("Starting pacemaker run, is now waiting for it to finish")
        local(f"touch {os.path.join(local_train_dir, 'CALC_ONGOING')}")
        local(f"cd {local_train_dir} && sbatch {os.path.basename(run_script_pacemaker_path)}")
        flyingpace.fpio.wait_for_calc_done(local_train_dir, gpu_connection)
        local(f"rm -rf {os.path.join(local_train_dir, 'CALC_ONGOING')}")
    elif (gpu_connection != None):
        log.info("Starting pacemaker run, is now waiting for it to finish")
        gpu_connection.run(f"touch {os.path.join(remote_train_dir, 'CALC_ONGOING')}", hide='both')
        with gpu_connection.cd(remote_train_dir):
            gpu_connection.run(f"sbatch {os.path.basename(run_script_pacemaker_path)}", hide='both')
        flyingpace.fpio.wait_for_calc_done(remote_train_dir, gpu_connection)
        gpu_connection.run(f"rm -rf {os.path.join(remote_train_dir, 'CALC_ONGOING')}", hide='both')
    log.info("pacemaker run has finished")

    #Run pace_activeset
    run_activeset(InputData)

    #Copy results to local folder
    if (gpu_connection != None):
        flyingpace.sshutils.get_dir_as_archive(local_train_dir, remote_train_dir, gpu_connection)

    return

def run_activeset(InputData: DataReader):
    '''
    Run pace_activeset in remote_train_dir or local_train_dir if gpu_connection is None
    '''

    log.info(f"Run 'pace_activeset' on 'output_potential.yaml'")

    manager_dict = InputData.manager_dict

    directory_dict = InputData.directory_dict

    gpu_connection = InputData.gpu_connection

    #read what is needed from manager_dict
    assert isinstance(manager_dict, dict)
    if "GPUpaceDir" in manager_dict:
        pace_dir = manager_dict["GPUpaceDir"]
    else:
        log.warning("No 'GPUpaceDir' provided in input file, please specify it")
        raise ValueError("No 'GPUpaceDir' provided in input file, please specify it")

    #Read what is needed from directory_dict an construct paths
    assert isinstance(directory_dict, dict)
    local_train_dir = directory_dict["local_train_dir"]
    local_active_set_file_path = os.path.join(local_train_dir, 'output_potential.asi')
    local_fitting_data_file_path = os.path.join(local_train_dir, 'fitting_data_info.pckl.gzip')
    local_output_potential_file_path = os.path.join(local_train_dir, 'output_potential.yaml')
    if (gpu_connection != None):
        remote_train_dir = directory_dict["remote_train_dir"]
        remote_active_set_file_path = os.path.join(remote_train_dir, 'output_potential.asi')
        remote_fitting_data_file_path = os.path.join(remote_train_dir, 'fitting_data_info.pckl.gzip')
        remote_output_potential_file_path = os.path.join(remote_train_dir, 'output_potential.yaml')
    ace_activeset_file_path = os.path.join(pace_dir, 'pace_activeset')

    if os.path.exists(local_active_set_file_path):
        log.warning(f"There already exists the .asi file {local_active_set_file_path}")
        return
    if (gpu_connection != None):
        if exists(gpu_connection, remote_active_set_file_path):
            log.warning(f"There already exists the .asi file {remote_active_set_file_path}")
            return

    #Check if all nessesary files exist
    if (gpu_connection == None):
        if (os.path.exists(local_fitting_data_file_path) and\
        os.path.exists(local_output_potential_file_path)):
            pass
        else: 
            log.warning(f"Check if {local_fitting_data_file_path} and {local_output_potential_file_path} are in their place")
            raise RuntimeError(f"Check if {local_fitting_data_file_path} and {local_output_potential_file_path} are in their place")
    elif (gpu_connection != None):
        if (exists(gpu_connection, remote_fitting_data_file_path) and\
        exists(gpu_connection, remote_output_potential_file_path)):
            pass
        else: 
            log.warning(f"Check if {remote_fitting_data_file_path} and {remote_output_potential_file_path} are in their place")
            raise RuntimeError(f"Check if {remote_fitting_data_file_path} and {remote_output_potential_file_path} are in their place")

    if (gpu_connection == None):
        local(f"cd {local_train_dir} && {ace_activeset_file_path} -d {os.path.basename(local_fitting_data_file_path)} {os.path.basename(local_output_potential_file_path)}")
    elif (gpu_connection != None):
        with gpu_connection.cd(remote_train_dir):
            gpu_connection.run(f"{ace_activeset_file_path} -d {os.path.basename(remote_fitting_data_file_path)} {os.path.basename(remote_output_potential_file_path)}", hide='both')

    return
