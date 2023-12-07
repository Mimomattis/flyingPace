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

    manager_dict = InputData.manager_dict

    pacemaker_dict = InputData.pacemaker_dict

    directory_dict = InputData.directory_dict

    train_connection = InputData.train_connection

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
    if (train_connection != None):
        remote_train_dir = directory_dict["remote_train_dir"]

    #Construct absolute paths for files
    pickle_file_path = manager_dict["pickleFilePath"]
    run_script_pacemaker_path = os.path.join(local_working_dir, run_script_pacemaker)

    #Read variables in case of a restart, old potential is then reused
    if "restart" in kwargs:
        restart = kwargs["restart"]
    else:
        restart = False

    #Check if there is a completed or ongoing calculation in local_train_dir or remote_train_dir
    if (flyingpace.fpio.calc_done_in_local_dir(local_train_dir)):
        log.warning(f"There already is a completed calculation in {local_train_dir}")
        return
        
    elif (flyingpace.fpio.calc_ongoing_in_local_dir(local_train_dir)):
        log.warning(f"There is an ongoing calculation in {local_train_dir}, is now waiting for it to finish")
        flyingpace.fpio.wait_for_calc_done(local_train_dir, train_connection)
        local(f"rm -rf {os.path.join(local_train_dir, 'CALC_ONGOING')}")
        return

    if (train_connection != None):
        
        if (flyingpace.fpio.calc_done_in_remote_dir(remote_train_dir, train_connection)):
            log.warning(f"There already is a completed calculation in {remote_train_dir}")
            log.warning(f"Copying results to {local_train_dir}")
            flyingpace.sshutils.get_dir_as_archive(local_train_dir, remote_train_dir, train_connection)
            return
        
        elif (flyingpace.fpio.calc_ongoing_in_remote_dir(remote_train_dir, train_connection)):
            log.warning(f"There is an ongoing calculation in {remote_train_dir}, is now waiting for it to finish")
            flyingpace.fpio.wait_for_calc_done(remote_train_dir, train_connection)
            train_connection.run(f"rm -rf {os.path.join(remote_train_dir, 'CALC_ONGOING')}", hide='both')
            flyingpace.sshutils.get_dir_as_archive(local_train_dir, remote_train_dir, train_connection)
            return
    
    #Check if all nessesary files exist
    if (os.path.exists(pickle_file_path) and\
    os.path.exists(run_script_pacemaker_path)):
        pass
    else: 
        log.warning(f"Check if {pickle_file_path} and {run_script_pacemaker_path} are in their place")
        raise RuntimeError(f"Check if {pickle_file_path} and {run_script_pacemaker_path} are in their place")

    #Add 'touch CALC_DONE' to run script if not there
    flyingpace.fpio.check_for_calc_done_in_script(run_script_pacemaker_path)

    #Copy run script to local_train_dir
    local(f"cp {run_script_pacemaker_path} {local_train_dir}")

    #Copy dataset to local_train_dir
    local(f"cp {pickle_file_path} {local_train_dir}")

    #In case of restart, copy the old potential file
    if restart:
        local(f"cp {manager_dict['potentialFilePath']} {os.path.join(local_train_dir, 'cont.yaml')}")
    
    #Create input file and save it to local_train_dir
    flyingpace.fpio.generate_pace_input(pickle_file_path, directory_dict, InputData)

    #Copy local_train_dir to remote_train_dir
    if (train_connection != None):
        flyingpace.sshutils.put_dir_as_archive(local_train_dir, remote_train_dir, train_connection)

    #Start pacemaker run
    if (train_connection == None):
        log.info("Starting pacemaker run, is now waiting for it to finish")
        local(f"touch {os.path.join(local_train_dir, 'CALC_ONGOING')}")
        local(f"cd {local_train_dir} && sbatch {os.path.basename(run_script_pacemaker_path)}")
        flyingpace.fpio.wait_for_calc_done(local_train_dir, train_connection)
        local(f"rm -rf {os.path.join(local_train_dir, 'CALC_ONGOING')}")
    elif (train_connection != None):
        log.info("Starting pacemaker run, is now waiting for it to finish")
        train_connection.run(f"touch {os.path.join(remote_train_dir, 'CALC_ONGOING')}", hide='both')
        with train_connection.cd(remote_train_dir):
            train_connection.run(f"sbatch {os.path.basename(run_script_pacemaker_path)}", hide='both')
        flyingpace.fpio.wait_for_calc_done(remote_train_dir, train_connection)
        train_connection.run(f"rm -rf {os.path.join(remote_train_dir, 'CALC_ONGOING')}", hide='both')
    log.info("pacemaker run has finished")

    #Copy results to local folder
    if (train_connection != None):
        flyingpace.sshutils.get_dir_as_archive(local_train_dir, remote_train_dir, train_connection)

    return

def follow_up_pacemaker(InputData: DataReader):
    '''
    Renames output_potential.yaml to {systemName}.yaml, runs pace_activeset
    to produce the file {systemName}.asi and adds both of them to InputData
    '''

    manager_dict = InputData.manager_dict

    directory_dict = InputData.directory_dict

    #read what is needed from manager_dict
    assert isinstance(manager_dict, dict)
    if "paceDir" in manager_dict:
        pace_dir = manager_dict["paceDir"]
    else:
        log.warning("No 'paceDir' provided in input file, please specify it")
        raise ValueError("No 'paceDir' provided in input file, please specify it")
    
    system_name = manager_dict["systemName"]

    #Read what is needed from directory_dict
    assert isinstance(directory_dict, dict)
    local_train_dir = directory_dict["local_train_dir"]

    #Rename the output potential an add it to InputData
    output_potential_file = f"output_potential.yaml"
    new_potential_file = f"{system_name}.yaml"
    local(f"cp {os.path.join(local_train_dir, output_potential_file)} {os.path.join(local_train_dir, new_potential_file)}")
    local_output_potential_file_path = os.path.join(local_train_dir, new_potential_file)
    InputData.change_data("manager_dict", "potentialFilePath", local_output_potential_file_path)

    #Run pace_activeset
    log.info(f"Run 'pace_activeset' on '{new_potential_file}'")

    local_active_set_file_path = os.path.join(local_train_dir, f"{system_name}.asi")
    local_fitting_data_file_path = os.path.join(local_train_dir, 'fitting_data_info.pckl.gzip')
    ace_activeset_file_path = os.path.join(pace_dir, 'pace_activeset')

    if os.path.exists(local_active_set_file_path):
        log.warning(f"There already exists the .asi file {local_active_set_file_path}")
        return

    #Check if all nessesary files exist
    
    if (os.path.exists(local_fitting_data_file_path) and\
    os.path.exists(local_output_potential_file_path)):
        pass
    else: 
        log.warning(f"Check if {local_fitting_data_file_path} and {local_output_potential_file_path} are in their place")
        raise RuntimeError(f"Check if {local_fitting_data_file_path} and {local_output_potential_file_path} are in their place")

    local(f"cd {local_train_dir} && {ace_activeset_file_path} -d {os.path.basename(local_fitting_data_file_path)} {os.path.basename(local_output_potential_file_path)}")

    #Add active set file path to InputData
    InputData.change_data("manager_dict", "activeSetFilePath", local_active_set_file_path)

    return