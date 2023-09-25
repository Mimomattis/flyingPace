import logging
import os 
import re

from fabric import Connection
from invoke import run as local
from patchwork.files import exists

import flyingpace.logginghelper

log = logging.getLogger(__name__)

def create_gen_directories(gen: int, cpu_connection: Connection, gpu_connection: Connection, manager_dict: dict):
    '''
    Creates all local and remote working directories for learning gerneration 'gen'
    and returns the paths for all directories in a dict
    '''

    directory_dict = {}

    #Read what is needed from manager_dict
    if "localWorkingDir" in manager_dict:
        directory_dict["local_working_dir"] = manager_dict["localWorkingDir"]
        log.info(f"Local working directory: {directory_dict['local_working_dir']}")
    else:
        log.warning("No 'localWorkingDir' provided in YAML file, please specify it")
        raise ValueError("No 'localWorkingDir' provided in YAML file, please specify it")
    
    if (cpu_connection != None):
        if "CPUWorkingDir" in manager_dict:
            directory_dict["cpu_working_dir"] = manager_dict["CPUWorkingDir"]
            log.info(f"CPU working directory: {directory_dict['cpu_working_dir']}")
        else:
            log.warning("No 'CPUWorkingDir' provided in YAML file, please specify it")
            raise ValueError("No 'CPUWorkingDir' provided in YAML file, please specify it")
        
    if (gpu_connection != None):
        if "GPUWorkingDir" in manager_dict:
            directory_dict["gpu_working_dir"] = manager_dict["GPUWorkingDir"]
            log.info(f"GPU working directory: {directory_dict['gpu_working_dir']}")
        else:
            log.warning("No 'GPUWorkingDir' provided in YAML file, please specify it")
            raise ValueError("No 'GPUWorkingDir' provided in YAML file, please specify it")
    
    #Create loacal generation directory by combining the local working directory with the generation number
    directory_dict["local_gen_dir"] = os.path.join(directory_dict["local_working_dir"], f"gen{str(gen)}")
        
    #Create the paths for all directories
    directory_dict["local_train_dir"] = os.path.join(directory_dict["local_gen_dir"], "train")
    directory_dict["local_exploration_dir"] = os.path.join(directory_dict["local_gen_dir"], "exploration")
    directory_dict["local_dft_dir"] = os.path.join(directory_dict["local_gen_dir"], "dft")
    
    #Create all directories if connections!=None
    if (cpu_connection != None):
        directory_dict["cpu_gen_dir"] = os.path.join(directory_dict["cpu_working_dir"], f"gen{str(gen)}")
        directory_dict["remote_exploration_dir"] = os.path.join(directory_dict["cpu_gen_dir"], "exploration")
        directory_dict["remote_dft_dir"] = os.path.join(directory_dict["cpu_gen_dir"], "dft")
    if (gpu_connection != None):
        directory_dict["gpu_gen_dir"] = os.path.join(directory_dict["gpu_working_dir"], f"gen{str(gen)}")
        directory_dict["remote_train_dir"] = os.path.join(directory_dict["gpu_gen_dir"], "train")

    #Create working directories if not there
    if not os.path.exists(directory_dict["local_working_dir"]):
        local(f'mkdir {directory_dict["local_working_dir"]}')
    if ((cpu_connection != None) and not exists(cpu_connection, directory_dict["cpu_working_dir"])):
        cpu_connection.run(f'mkdir {directory_dict["cpu_working_dir"]}', hide='both')
    if ((gpu_connection != None) and not exists(gpu_connection, directory_dict["gpu_working_dir"])):
        gpu_connection.run(f'mkdir {directory_dict["gpu_working_dir"]}', hide='both')

    #Check if the local gen directories already exist
    if os.path.exists(directory_dict["local_gen_dir"]):
        log.warning(f"The local generation {gen} already exists!")
    else:
        #Create all local directories
        local(f'mkdir {directory_dict["local_gen_dir"]} \
        {directory_dict["local_train_dir"]} \
        {directory_dict["local_exploration_dir"]} \
        {directory_dict["local_dft_dir"]}')

    #Check if the remote gen directories already exist
    if (cpu_connection != None):
        if (exists(cpu_connection, directory_dict["cpu_gen_dir"])):
            log.warning(f"The remote generation {gen} already exists on the CPU host!")
        else:
            #Create all remote directories
            cpu_connection.run(f'mkdir {directory_dict["cpu_gen_dir"]}\
            {directory_dict["remote_exploration_dir"]}\
            {directory_dict["remote_dft_dir"]}', hide='both')
            
    if (gpu_connection != None):
        #If CPU and GPU clusters have a sharded file system, gpu_gen_dir=cpu_gen_dir will already
        #exists and an error will be thrown. To avoid this, it is checked if gpu_gen_dir and remote_train_dir
        #exist.
        if (exists(gpu_connection, directory_dict["gpu_gen_dir"]) and not exists(gpu_connection, directory_dict["remote_train_dir"])):
            gpu_connection.run(f'mkdir {directory_dict["remote_train_dir"]}', hide='both')
        elif (exists(gpu_connection, directory_dict["gpu_gen_dir"]) and exists(gpu_connection, directory_dict["remote_train_dir"])):
            log.warning(f"The remote generation {gen} already exists on the GPU host!\n \
            Please check if the right generation was specified")
        else:
            gpu_connection.run(f'mkdir {directory_dict["gpu_gen_dir"]}\
            {directory_dict["remote_train_dir"]}', hide='both')
        
    return directory_dict


def get_prev_path(path: str):

    current_path = path
    gen_string = re.findall('gen[0-9]+', current_path)[-1]
    gen = int(gen_string.replace("gen",""))
    prev_gen = gen -1
    prev_gen_string = f"gen{prev_gen}"
    prev_path = path.replace(gen_string, prev_gen_string)

    return prev_path
        


