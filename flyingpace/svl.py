import argparse
import glob
import logging
import os
import sys

import pandas as pd

from ase.io.lammpsrun import read_lammps_dump_text
from fabric import Connection
from invoke import run as local #To run local commands like rsync function
from shutil import copyfile

import flyingpace.dirmanager
import flyingpace.fpio
import flyingpace.logginghelper
import flyingpace.sshutils

from flyingpace.input import DataReader

log = logging.getLogger(__name__) 

def run_explorative_md(cpu_connection: Connection, directory_dict: dict, InputData: DataReader):

    exploration_dict = InputData.exploration_dict

    #Read what is needed from pacemaker_dict
    assert isinstance(exploration_dict, dict)
    if "datafile" in exploration_dict:
        datafile = exploration_dict["datafile"]
        log.info(f"Data file for exploration: {datafile}")
    else:
        log.warning("No 'datafile' provided in YAML file, the defalut is 'last'")
        datafile = 'last'

    if "explorationRunScript" in exploration_dict:
        run_script_exploration = exploration_dict["explorationRunScript"]
        log.info(f"Run script for exploration run: {run_script_exploration}")
    else:
        log.warning("No 'explorationRunScript' provided in YAML file, please specify it")
        raise ValueError("No 'explorationRunScript' provided in YAML file, please specify it")

    #Read what is needed from directory_dict
    assert isinstance(directory_dict, dict)
    local_working_dir = directory_dict["local_working_dir"]
    local_exploration_dir = directory_dict["local_exploration_dir"]
    prev_local_exploration_dir = flyingpace.dirmanager.get_prev_path(directory_dict["local_exploration_dir"])
    local_train_dir = directory_dict["local_train_dir"]
    if (cpu_connection != None):
        remote_exploration_dir = directory_dict["remote_exploration_dir"]
    #Construct absolute paths for files
    run_script_exploration_path = os.path.join(local_working_dir, run_script_exploration)
    potential_file_path = os.path.join(local_train_dir, 'output_potential.yaml')
    active_set_file_path = os.path.join(local_train_dir, 'output_potential.asi')
    #Generate datafile from trainingset in local_train_dir or copy it from local_working_dir
    if (datafile == 'randomFromTrain'):
        datafile_path = flyingpace.fpio.datafile_from_pickle('dft_data.pckl.gzip', local_train_dir, 'random')
    elif (datafile == 'last'):
        if (os.path.exists(os.path.join(prev_local_exploration_dir, 'last_structure.data'))):
            datafile_path = os.path.join(local_working_dir, datafile)
        else: 
            datafile_path = flyingpace.fpio.datafile_from_pickle('dft_data.pckl.gzip', local_train_dir, 'last')
    else:
        datafile_path = os.path.join(local_working_dir, datafile)

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
    flyingpace.fpio.generate_lammps_input(datafile_path, directory_dict, InputData)

    #Copy local_exploration_dir to remote_exploration_dir
    if (cpu_connection != None):
        flyingpace.sshutils.put_dir_as_archive(local_exploration_dir, remote_exploration_dir, cpu_connection)

    #Start AIMD run
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

