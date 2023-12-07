import logging
import os 
import re

from fabric import Connection
from invoke import run as local
from patchwork.files import exists

import flyingpace.logginghelper

from flyingpace.input import DataReader


log = logging.getLogger(__name__)

def create_gen_directories(gen: int, InputData: DataReader, **kwargs):
    '''
    Creates all local and remote working directories for learning gerneration 'gen'
    and returns the paths for all directories in a dict
    '''
    
    manager_dict = InputData.manager_dict

    directory_dict = InputData.directory_dict

    dft_connection = InputData.dft_connection
    train_connection = InputData.train_connection
    exploration_connection = InputData.exploration_connection

    #Read what is needed from manager_dict
    if "localWorkingDir" in manager_dict:
        directory_dict["local_working_dir"] = manager_dict["localWorkingDir"]
        log.info(f"Local working directory: {directory_dict['local_working_dir']}")
    else:
        log.warning("No 'localWorkingDir' provided in input file, please specify it")
        raise ValueError("No 'localWorkingDir' provided in input file, please specify it")
    
    if (dft_connection != None):
        if "DFTWorkingDir" in manager_dict:
            directory_dict["dft_working_dir"] = manager_dict["DFTWorkingDir"]
            log.info(f"DFT working directory: {directory_dict['dft_working_dir']}")
        else:
            log.warning("No 'DFTWorkingDir' provided in input file, please specify it")
            raise ValueError("No 'DFTWorkingDir' provided in input file, please specify it")
        
    if (train_connection != None):
        if "TrainWorkingDir" in manager_dict:
            directory_dict["train_working_dir"] = manager_dict["TrainWorkingDir"]
            log.info(f"Train working directory: {directory_dict['train_working_dir']}")
        else:
            log.warning("No 'TrainWorkingDir' provided in input file, please specify it")
            raise ValueError("No 'TrainWorkingDir' provided in input file, please specify it")
        
    if (exploration_connection != None):
        if "ExplorationWorkingDir" in manager_dict:
            directory_dict["exploration_working_dir"] = manager_dict["ExplorationWorkingDir"]
            log.info(f"Exploration working directory: {directory_dict['exploration_working_dir']}")
        else:
            log.warning("No 'ExplorationWorkingDir' provided in input file, please specify it")
            raise ValueError("No 'ExplorationWorkingDir' provided in input file, please specify it")
    
    #Generate the path for local_gen_dir
    directory_dict["local_gen_dir"] = os.path.join(directory_dict["local_working_dir"], f"gen{str(gen)}")
        
    #Generate the paths for all directories
    directory_dict["local_train_dir"] = os.path.join(directory_dict["local_gen_dir"], "train")
    directory_dict["local_exploration_dir"] = os.path.join(directory_dict["local_gen_dir"], "exploration")
    directory_dict["local_dft_dir"] = os.path.join(directory_dict["local_gen_dir"], "dft")
    
    #Generate the paths for all directories if connections!=None
    if (dft_connection != None):
        directory_dict["dft_gen_dir"] = os.path.join(directory_dict["dft_working_dir"], f"gen{str(gen)}")
        directory_dict["remote_dft_dir"] = os.path.join(directory_dict["dft_gen_dir"], "dft")
    if (train_connection != None):
        directory_dict["train_gen_dir"] = os.path.join(directory_dict["train_working_dir"], f"gen{str(gen)}")
        directory_dict["remote_train_dir"] = os.path.join(directory_dict["train_gen_dir"], "train")
    if (exploration_connection != None):
        directory_dict["exploration_gen_dir"] = os.path.join(directory_dict["exploration_working_dir"], f"gen{str(gen)}")
        directory_dict["remote_exploration_dir"] = os.path.join(directory_dict["exploration_gen_dir"], "exploration")

    #If create_dictonaries=False, only the names are generated
    if "create_dictonaries" in kwargs:
        create_dictonaries = kwargs["create_dictonaries"]
    else:
        create_dictonaries = True

    if create_dictonaries:

        #Check if local directories exist and if not, create them 
        if not os.path.exists(directory_dict["local_gen_dir"]):
            local(f'mkdir {directory_dict["local_gen_dir"]}')
        else: 
            log.warning(f"The local generation {gen} already exists!")
        
        if not os.path.exists(directory_dict["local_dft_dir"]):
            local(f'mkdir {directory_dict["local_dft_dir"]}')
        if not os.path.exists(directory_dict["local_train_dir"]):
            local(f'mkdir {directory_dict["local_train_dir"]}')
        if not os.path.exists(directory_dict["local_exploration_dir"]):
                local(f'mkdir {directory_dict["local_exploration_dir"]}')
            

        #Check if remote directories exist and if not, create them
        if ((dft_connection != None) and not exists(dft_connection, directory_dict["dft_working_dir"])):
            dft_connection.run(f'mkdir {directory_dict["dft_working_dir"]}', hide='both')
        if ((dft_connection != None) and not exists(dft_connection, directory_dict["dft_gen_dir"])):
            dft_connection.run(f'mkdir {directory_dict["dft_gen_dir"]}', hide='both')
        if ((dft_connection != None) and not exists(dft_connection, directory_dict["remote_dft_dir"])):
            dft_connection.run(f'mkdir {directory_dict["remote_dft_dir"]}', hide='both')

        if ((train_connection != None) and not exists(train_connection, directory_dict["train_working_dir"])):
            train_connection.run(f'mkdir {directory_dict["train_working_dir"]}', hide='both')
        if ((train_connection != None) and not exists(train_connection, directory_dict["train_gen_dir"])):
            train_connection.run(f'mkdir {directory_dict["train_gen_dir"]}', hide='both')
        if ((train_connection != None) and not exists(train_connection, directory_dict["remote_train_dir"])):
            train_connection.run(f'mkdir {directory_dict["remote_train_dir"]}', hide='both')

        if ((exploration_connection != None) and not exists(exploration_connection, directory_dict["exploration_working_dir"])):
            exploration_connection.run(f'mkdir {directory_dict["exploration_working_dir"]}', hide='both')
        if ((exploration_connection != None) and not exists(exploration_connection, directory_dict["exploration_gen_dir"])):
            exploration_connection.run(f'mkdir {directory_dict["exploration_gen_dir"]}', hide='both')
        if ((exploration_connection != None) and not exists(exploration_connection, directory_dict["remote_exploration_dir"])):
            exploration_connection.run(f'mkdir {directory_dict["remote_exploration_dir"]}', hide='both')
        
    return directory_dict


def get_prev_path(path: str):

    current_path = path
    gen_string = re.findall('gen[0-9]+', current_path)[-1]
    gen = int(gen_string.replace("gen",""))
    prev_gen = gen -1
    prev_gen_string = f"gen{prev_gen}"
    prev_path = path.replace(gen_string, prev_gen_string)

    return prev_path
        


