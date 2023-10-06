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

def run_explorative_md(cpu_connection: Connection, directory_dict: dict, InputData: DataReader):

    log.info(f"*** EXPLORATION RUN ***")

    exploration_dict = InputData.exploration_dict

    #Read what is needed from pacemaker_dict
    assert isinstance(exploration_dict, dict)
    if "datafile" in exploration_dict:
        datafile = exploration_dict["datafile"]
        log.info(f"Data file for exploration: {datafile}")
    else:
        log.warning("No 'datafile' provided in YAML file, the defalut is 'last'")
        datafile = 'last'

    if "select" in exploration_dict:
        select = True
    else:
        select = False
    
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
    #Generate datafile from trainingset in local_train_dir 
    #or copy it from prev_local_exploration_dir if existent 
    #or copy it fromlocal_working_dir
    if (datafile == 'randomFromTrain'):
        if (os.path.exists(os.path.join(prev_local_exploration_dir, 'random_structure.data'))):
            datafile_path = os.path.join(prev_local_exploration_dir, 'random_structure.data')
        else: 
            datafile_path = flyingpace.fpio.datafile_from_pickle('dft_data.pckl.gzip', local_train_dir, 'random')
    elif (datafile == 'last'):
        if (os.path.exists(os.path.join(prev_local_exploration_dir, 'last_structure.data'))):
            datafile_path = os.path.join(prev_local_exploration_dir, 'last_structure.data')
        else: 
            datafile_path = flyingpace.fpio.datafile_from_pickle('dft_data.pckl.gzip', local_train_dir, 'last')
    else:
        datafile_path = os.path.join(local_working_dir, datafile)

    #Check if there is a completed or ongoing calculation in local_exploration_dir or remote_exploration_dir

    if (flyingpace.fpio.calc_done_in_local_dir(local_exploration_dir)):
        log.warning(f"There already is a completed calculation in {local_exploration_dir}")
        if select:
            run_pace_select(cpu_connection, directory_dict, InputData)
        return
    elif (flyingpace.fpio.calc_ongoing_in_local_dir(local_exploration_dir)):
        log.warning(f"There is an ongoing calculation in {local_exploration_dir}, is now waiting for it to finish")
        flyingpace.fpio.wait_for_calc_done(local_exploration_dir, cpu_connection)
        local(f"rm -rf {os.path.join(local_exploration_dir, 'CALC_ONGOING')}")
        if select:
            run_pace_select(cpu_connection, directory_dict, InputData)
        return

    if (cpu_connection != None):
        
        if (flyingpace.fpio.calc_done_in_remote_dir(remote_exploration_dir, cpu_connection)):
            log.warning(f"There already is a completed calculation in {remote_exploration_dir}")
            if select:
                run_pace_select(cpu_connection, directory_dict, InputData)
            log.warning(f"Copying results to {remote_exploration_dir}")
            flyingpace.sshutils.get_dir_as_archive(local_exploration_dir, remote_exploration_dir, cpu_connection)
            return
        
        elif (flyingpace.fpio.calc_ongoing_in_remote_dir(remote_exploration_dir, cpu_connection)):
            log.warning(f"There is an ongoing calculation in {remote_exploration_dir}, is now waiting for it to finish")
            flyingpace.fpio.wait_for_calc_done(remote_exploration_dir, cpu_connection)
            cpu_connection.run(f"rm -rf {os.path.join(remote_exploration_dir, 'CALC_ONGOING')}", hide='both')
            if select:
                run_pace_select(cpu_connection, directory_dict, InputData)
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

    if select:
        run_pace_select(cpu_connection, directory_dict, InputData)

    #Copy results to local folder
    if (cpu_connection != None):
        flyingpace.sshutils.get_dir_as_archive(local_exploration_dir, remote_exploration_dir, cpu_connection)

    return

def run_pace_select(cpu_connection: Connection, directory_dict: dict, InputData: DataReader):

    log.info(f"Selecting extrapolative structures via pace_select")

    exploration_dict = InputData.exploration_dict
    manager_dict = InputData.manager_dict

    #read what is needed from exploration_dict
    assert isinstance(exploration_dict, dict)
    if "datafile" in exploration_dict:
        datafile = exploration_dict["datafile"]
        log.info(f"Data file for exploration: {datafile}")
    else:
        log.warning("No 'datafile' provided in YAML file, the defalut is 'last'")
        datafile = 'last'
    
    if "select" in exploration_dict:
        select_num = exploration_dict["select"]
    else:
        raise ValueError("Tried to run pace_select without 'select' being specified in the YAML file")

    #read what is needed from manager_dict
    assert isinstance(manager_dict, dict)
    if "CPUpaceDir" in manager_dict:
        pace_dir = manager_dict["CPUpaceDir"]
    else:
        log.warning("No 'CPUpaceDir' provided in YAML file, please specify it")
        raise ValueError("No 'CPUpaceDir' provided in YAML file, please specify it")

    #Read what is needed from directory_dict an construct paths
    assert isinstance(directory_dict, dict)
    local_exploration_dir = directory_dict["local_exploration_dir"]
    local_active_set_file_path = os.path.join(local_exploration_dir, 'output_potential.asi')
    local_potential_file_path = os.path.join(local_exploration_dir, 'output_potential.yaml')
    local_extrapolative_dump_file_path = os.path.join(local_exploration_dir, 'extrapolative_structures.dump')
    local_selected_file_path = os.path.join(local_exploration_dir, 'selected.pkl.gz')
    local_output_pickle_file_path = os.path.join(local_exploration_dir, 'extrapolative_structures.pckl.gzip')
    if (cpu_connection != None):
        remote_exploration_dir = directory_dict["remote_exploration_dir"]
        remote_active_set_file_path = os.path.join(remote_exploration_dir, 'output_potential.asi')
        remote_potential_file_path = os.path.join(remote_exploration_dir, 'output_potential.yaml')
        remote_extrapolative_dump_file_path = os.path.join(remote_exploration_dir, 'extrapolative_structures.dump')
        remote_selected_file_path = os.path.join(remote_exploration_dir, 'selected.pkl.gz')
        remote_output_pickle_file_path = os.path.join(remote_exploration_dir, 'extrapolative_structures.pckl.gzip')
    ace_select_file_path = os.path.join(pace_dir, 'pace_select')

    #Get element list from datafile
    local_working_dir = directory_dict["local_working_dir"]
    local_train_dir = directory_dict["local_train_dir"]
    prev_local_exploration_dir = flyingpace.dirmanager.get_prev_path(directory_dict["local_exploration_dir"])
    if (datafile == 'randomFromTrain'):
        if (os.path.exists(os.path.join(prev_local_exploration_dir, 'random_structure.data'))):
            datafile_path = os.path.join(prev_local_exploration_dir, 'random_structure.data')
        else: 
            datafile_path = flyingpace.fpio.datafile_from_pickle('dft_data.pckl.gzip', local_train_dir, 'random')
    elif (datafile == 'last'):
        if (os.path.exists(os.path.join(prev_local_exploration_dir, 'last_structure.data'))):
            datafile_path = os.path.join(prev_local_exploration_dir, 'last_structure.data')
        else: 
            datafile_path = flyingpace.fpio.datafile_from_pickle('dft_data.pckl.gzip', local_train_dir, 'last')
    else:
        datafile_path = os.path.join(local_working_dir, datafile)
    with open(datafile_path, 'r') as f:
        data_file = f.readlines()
    for idx, line in enumerate(data_file):
        if "atom types" in line:
            num_types = int(line.split()[0])
        if "Masses" in line:
            masses_idx = idx
    element_list = []        
    for i in range(num_types):
        element_list.append(data_file[i+masses_idx+2].split()[-1])
    element_string = " ".join(element_list)

    if os.path.exists(local_output_pickle_file_path):
            log.info(f"The pickle file {local_output_pickle_file_path} already exists")
            return
    if (cpu_connection != None):
        if exists(cpu_connection, remote_output_pickle_file_path):
            log.info(f"The pickle file {remote_output_pickle_file_path} already exists")
            return

    #Check if all nessesary files exist
    if (cpu_connection == None):
        if (os.path.exists(local_active_set_file_path) and\
        os.path.exists(local_potential_file_path) and\
        os.path.exists(local_extrapolative_dump_file_path)):
            pass
        else: 
            log.warning(f"Check if {local_active_set_file_path}, {local_potential_file_path} and {local_extrapolative_dump_file_path} are in their place")
            raise RuntimeError(f"Check if {local_active_set_file_path}, {local_potential_file_path} and {local_extrapolative_dump_file_path} are in their place")
    elif (cpu_connection != None):
        if (exists(cpu_connection, remote_active_set_file_path) and\
            exists(cpu_connection, remote_potential_file_path) and\
        exists(cpu_connection, remote_extrapolative_dump_file_path)):
            pass
        else: 
            log.warning(f"Check if {remote_active_set_file_path}, {remote_potential_file_path} and {remote_extrapolative_dump_file_path} are in their place")
            raise RuntimeError(f"Check if {remote_active_set_file_path}, {remote_potential_file_path} and {remote_extrapolative_dump_file_path} are in their place")

    if (cpu_connection == None):
        local(f'cd {local_exploration_dir} && {ace_select_file_path} -p {local_potential_file_path} -a {local_active_set_file_path} -e "{element_string}"\
        -m {select_num} {local_extrapolative_dump_file_path}')
        local(f"yes | cp {local_selected_file_path} {local_output_pickle_file_path}")
    elif (cpu_connection != None):
        with cpu_connection.cd(remote_exploration_dir):
            cpu_connection.run(f'cd {remote_exploration_dir} && {ace_select_file_path} -p {remote_potential_file_path} -a {remote_active_set_file_path} -e "{element_string}"\
            -m {select_num} {remote_extrapolative_dump_file_path}', hide='both')
            cpu_connection.run(f"yes | cp {remote_selected_file_path} {remote_output_pickle_file_path}", hide='both')


    return