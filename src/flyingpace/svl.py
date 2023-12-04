import argparse
import glob
import logging
import os
import sys

import pandas as pd

from ase.io.lammpsrun import read_lammps_dump_text
from fabric import Connection
from invoke import run as local #To run local commands like rsync function
from patchwork.files import exists
from shutil import copyfile

import flyingpace.dirmanager
import flyingpace.fpio
import flyingpace.logginghelper
import flyingpace.sshutils

from flyingpace.input import DataReader

log = logging.getLogger(__name__) 

def run_explorative_md(InputData: DataReader):

    log.info(f"*** EXPLORATION RUN ***")

    exploration_dict = InputData.exploration_dict
    manager_dict = InputData.manager_dict

    directory_dict = InputData.directory_dict

    cpu_connection = InputData.cpu_connection

    #Read what is needed from exploration_dict
    assert isinstance(exploration_dict, dict)    
    if "explorationRunScript" in exploration_dict:
        run_script_exploration = exploration_dict["explorationRunScript"]
        log.info(f"Run script for exploration run: {run_script_exploration}")
    else:
        log.warning("No 'explorationRunScript' provided in input file, please specify it")
        raise ValueError("No 'explorationRunScript' provided in input file, please specify it")

    #Read what is needed from directory_dict
    assert isinstance(directory_dict, dict)
    local_working_dir = directory_dict["local_working_dir"]
    local_exploration_dir = directory_dict["local_exploration_dir"]
    if (cpu_connection != None):
        remote_exploration_dir = directory_dict["remote_exploration_dir"]
    #Construct absolute paths for files
    run_script_exploration_path = os.path.join(local_working_dir, run_script_exploration)
    #Get file paths from manager_dict
    potential_file_path = manager_dict["potentialFilePath"]
    active_set_file_path = manager_dict["activeSetFilePath"]
    datafile_path = manager_dict["dataFilePath"]

    #Check if there is a completed or ongoing calculation in local_exploration_dir or remote_exploration_dir

    if (flyingpace.fpio.calc_done_in_local_dir(local_exploration_dir)):
        log.warning(f"There already is a completed calculation in {local_exploration_dir}")
        return
    elif (flyingpace.fpio.calc_ongoing_in_local_dir(local_exploration_dir)):
        log.warning(f"There is an ongoing calculation in {local_exploration_dir}, is now waiting for it to finish")
        flyingpace.fpio.wait_for_calc_done(local_exploration_dir, cpu_connection)
        local(f"rm -rf {os.path.join(local_exploration_dir, 'CALC_ONGOING')}")
        return

    if (cpu_connection != None):
        
        if (flyingpace.fpio.calc_done_in_remote_dir(remote_exploration_dir, cpu_connection)):
            log.warning(f"There already is a completed calculation in {remote_exploration_dir}")
            log.warning(f"Copying results to {remote_exploration_dir}")
            flyingpace.sshutils.get_dir_as_archive(local_exploration_dir, remote_exploration_dir, cpu_connection)
            return
        
        elif (flyingpace.fpio.calc_ongoing_in_remote_dir(remote_exploration_dir, cpu_connection)):
            log.warning(f"There is an ongoing calculation in {remote_exploration_dir}, is now waiting for it to finish")
            flyingpace.fpio.wait_for_calc_done(remote_exploration_dir, cpu_connection)
            cpu_connection.run(f"rm -rf {os.path.join(remote_exploration_dir, 'CALC_ONGOING')}", hide='both')
            flyingpace.sshutils.get_dir_as_archive(local_exploration_dir, remote_exploration_dir, cpu_connection)
            return
    
    #Check if all nessesary files exist
    if (os.path.exists(datafile_path) and\
    os.path.exists(run_script_exploration_path) and\
    os.path.exists(potential_file_path) and\
    os.path.exists(active_set_file_path)):
        pass
    else: 
        log.warning(f"Check if {datafile_path}, {run_script_exploration_path}, \
        {potential_file_path} and {active_set_file_path} are in their place")
        raise RuntimeError(f"Check if {datafile_path}, {run_script_exploration_path}, \
        {potential_file_path} and {active_set_file_path} are in their place")


    #Add 'touch CALC_DONE' to run script if not there
    flyingpace.fpio.check_for_calc_done_in_script(run_script_exploration_path)

    #Copy all files to local_exploration_dir
    local(f"cp {datafile_path} {run_script_exploration_path} {potential_file_path} {active_set_file_path} {local_exploration_dir}")

    #Create input file and save it to local_train_dir
    flyingpace.fpio.generate_lammps_input(InputData)

    #Copy local_exploration_dir to remote_exploration_dir
    if (cpu_connection != None):
        flyingpace.sshutils.put_dir_as_archive(local_exploration_dir, remote_exploration_dir, cpu_connection)

    #Start Exploration_run run
    if (cpu_connection == None):
        log.info("Starting exploration run, is now waiting for it to finish")
        local(f"touch {os.path.join(local_exploration_dir, 'CALC_ONGOING')}")
        local(f"cd {local_exploration_dir} && sbatch {os.path.basename(run_script_exploration_path)}")
        flyingpace.fpio.wait_for_calc_done(local_exploration_dir, cpu_connection)
        local(f"rm -rf {os.path.join(local_exploration_dir, 'CALC_ONGOING')}")
    elif (cpu_connection != None):
        log.info("Starting exploration run, is now waiting for it to finish")
        cpu_connection.run(f"touch {os.path.join(remote_exploration_dir, 'CALC_ONGOING')}", hide='both')
        with cpu_connection.cd(remote_exploration_dir):
            cpu_connection.run(f"sbatch {os.path.basename(run_script_exploration_path)}", hide='both')
        flyingpace.fpio.wait_for_calc_done(remote_exploration_dir, cpu_connection)
        cpu_connection.run(f"rm -rf {os.path.join(remote_exploration_dir, 'CALC_ONGOING')}", hide='both')
    log.info("Exploration run has finished")

    #Copy results to local folder
    if (cpu_connection != None):
        flyingpace.sshutils.get_dir_as_archive(local_exploration_dir, remote_exploration_dir, cpu_connection)

    return

def follow_up_exploration(InputData: DataReader):
    '''
    If 'select' is in InputData, pace_select is run and the result is saved as 'extrapolative_structures.pckl.gzip'
    if 'select' is not in InputData, extrapolative_dump_to_pickle is run to save 'extrapolative_structures.dump' directly
    to 'extrapolative_structures.pckl.gzip' without running pace_select
    '''

    exploration_dict = InputData.exploration_dict
    manager_dict = InputData.manager_dict

    directory_dict = InputData.directory_dict

    #read what is needed from exploration_dict
    assert isinstance(exploration_dict, dict)
    if "select" in exploration_dict:
        select = True
        select_num = exploration_dict["select"]
    else:
        select = False

    #read what is needed from manager_dict
    assert isinstance(manager_dict, dict)
    if "paceDir" in manager_dict:
        pace_dir = manager_dict["paceDir"]
    else:
        log.warning("No 'paceDir' provided in input file, please specify it")
        raise ValueError("No 'paceDir' provided in input file, please specify it")
    
    log.info(f"Selecting extrapolative structures via pace_select")

    #Read what is needed from directory_dict an construct paths
    assert isinstance(directory_dict, dict)
    local_exploration_dir = directory_dict["local_exploration_dir"]
    local_active_set_file_path = os.path.join(local_exploration_dir, os.path.basename(manager_dict["potentialFilePath"]))
    local_potential_file_path = os.path.join(local_exploration_dir, os.path.basename(manager_dict["activeSetFilePath"]))
    local_extrapolative_dump_file_path = os.path.join(local_exploration_dir, 'extrapolative_structures.dump')
    local_selected_file_path = os.path.join(local_exploration_dir, 'selected.pkl.gz')
    local_output_pickle_file_path = os.path.join(local_exploration_dir, 'extrapolative_structures.pckl.gzip')
    pace_select_file_path = os.path.join(pace_dir, 'pace_select')

    if select:

        #Read elementlist from manager_dict
        element_list = manager_dict["elementList"]

        if os.path.exists(local_output_pickle_file_path):
            log.info(f"The pickle file {local_output_pickle_file_path} already exists")
            return

        #Check if all nessesary files exist
    
        if (os.path.exists(local_active_set_file_path) and\
        os.path.exists(local_potential_file_path) and\
        os.path.exists(local_extrapolative_dump_file_path)):
            pass
        else: 
            log.warning(f"Check if {local_active_set_file_path}, {local_potential_file_path} and {local_extrapolative_dump_file_path} are in their place")
            raise RuntimeError(f"Check if {local_active_set_file_path}, {local_potential_file_path} and {local_extrapolative_dump_file_path} are in their place")

        local(f'cd {local_exploration_dir} && {pace_select_file_path} -p {local_potential_file_path} -a {local_active_set_file_path} -e "{element_list}"\
        -m {select_num} {local_extrapolative_dump_file_path}')
        local(f"yes | cp {local_selected_file_path} {local_output_pickle_file_path}")

        return
    
    if not select:

        flyingpace.fpio.extrapolative_dump_to_pickle(InputData)