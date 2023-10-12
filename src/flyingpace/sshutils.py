import logging
import os
import zipfile

from fabric import Connection
from functools import wraps
from invoke import run as local
from patchwork.files import exists

from flyingpace.input import DataReader

log = logging.getLogger(__name__)

def initialize_connections(InputData: DataReader):
    '''
    Returns two fabric.Connection objects for the CPU and GPU cluster.
    In case when 'local' is provided for either the CPU or GPU connection,
    None is returned
    '''

    manager_dict = InputData.manager_dict

    #Read what is needed from manager_dict
    if "CPUHost" in manager_dict:
        cpu_host = manager_dict["CPUHost"]
        if (cpu_host == 'local'):
            log.info(f"CPU calculations are done locally")
        else:
            log.info(f"CPU host: {cpu_host}")
    else:
        log.warning("No 'CPUHost' provided in input file, please specify it")
        raise ValueError("No 'CPUHost' provided in input file, please specify it")
    
    if "GPUHost" in manager_dict:
        gpu_host = manager_dict["GPUHost"]
        if (gpu_host == 'local'):
            log.info(f"CPU calculations are done locally")
        else:
            log.info(f"GPU host: {gpu_host}")
    else:
        log.warning("No 'GPUHost' provided in input file, please specify it")
        raise ValueError("No 'GPUHost' provided in input file, please specify it")
    
    if (cpu_host != "local"):
        if "CPUUser" in manager_dict:
            cpu_user = manager_dict["CPUUser"]
            log.info(f"CPU user: {cpu_user}")
        else:
            log.warning("No 'CPUUser' provided in input file, please specify it")
            raise ValueError("No 'CPUUser' provided in input file, please specify it")
        
        if "CPUJumpHost" in manager_dict:
            cpu_jump_host = manager_dict["CPUJumpHost"]
            log.info(f"CPU Jump host: {cpu_jump_host}")
        else:
            cpu_jump_host = None
            
    if (gpu_host != "local"):
        if "GPUUser" in manager_dict:
            gpu_user = manager_dict["GPUUser"]
            log.info(f"GPU user: {gpu_user}")
        else:
            log.warning("No 'GPUUser' provided in input file, please specify it")
            raise ValueError("No 'GPUUser' provided in input file, please specify it")
        if "GPUJumpHost" in manager_dict:
            gpu_jump_host = manager_dict["GPUJumpHost"]
            log.info(f"GPU Jump host: {gpu_jump_host}")
        else:
            gpu_jump_host = None

    #Setup for the ssh connection
    if (cpu_host == "local"):
        cpu_connection = None
    elif (cpu_jump_host == None):
        cpu_connection = Connection(host=cpu_host, 
                                    user=cpu_user, 
                                    )
    else:
        cpu_connection = Connection(host=cpu_host, 
                                    user=cpu_user, 
                                    gateway=Connection(
                                            host=cpu_jump_host,
                                            user=cpu_user
                                            ),
                                    )
    if (gpu_host == "local"):
        gpu_connection = None
    elif (gpu_jump_host == None):
        gpu_connection = Connection(host=gpu_host, 
                                    user=gpu_user, 
                                    )
    else:
        gpu_connection = Connection(host=gpu_host, 
                                    user=gpu_user, 
                                    gateway=Connection(
                                            host=gpu_jump_host,
                                            user=gpu_user
                                            ),
                                    )
        
    #Test if both connections are correctly set up, if not throw an exception
    if (cpu_connection != None):
        try:
            cpu_connection.run("echo 'CPU connection is working'", hide='both')
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}\n \
            CPU connection could not be opend, check given details in input file"
            message = template.format(type(ex).__name__, ex.args)
            log.info(message)
    if (gpu_connection != None):       
        try:
            gpu_connection.run("echo 'GPU connection is working'", hide='both')
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}\n \
            GPU connection could not be opend, check given details in input file"
            message = template.format(type(ex).__name__, ex.args)
            log.info(message)
    
    return cpu_connection, gpu_connection

def check_same_dir(func):
    '''
    Decorator to check if source and destination directories are the same for 
    put/get functions. Useful when the local machine and the CPU/GPU hosts have the 
    same file system
    '''
    @wraps(func)
    def wrapper(src_dir, dest_dir, *args, **kwargs):
        if src_dir == dest_dir:
            log.info(f"Source ({src_dir}) and destination ({dest_dir}) are the same. Skipping the copying process.")
        else:
            func(src_dir, dest_dir, *args, **kwargs)
    return wrapper

#Two wrappers for the built in rsync functions
@check_same_dir
def rsync_from_remote(local_dir: str, remote_dir: str, remote_user: str, proxy_host: str, remote_host: str):
    '''
    Copy a directory from 'remote_dir' to 'local_dir' 
    via the connection 'remote_user@remote_host' and the 
    jump host 'proxy_host' using the local rsync functionality
    '''
    local("rsync -azv -e 'ssh -A -J " + remote_user + "@" + 
          proxy_host + "' " + remote_user + '@' + 
          remote_host + ':' + remote_dir + ' ' + local_dir)
    
    return

@check_same_dir  
def rsync_to_remote(local_dir: str, remote_dir: str, remote_user: str, proxy_host: str, remote_host: str):
    '''
    Copy a directory from 'local_dir' to 'remote_dir' 
    via the connection 'remote_user@remote_host' and the 
    jump host 'proxy_host' using the local rsync functionality
    '''
    local("rsync -azv -e 'ssh -A -J " + remote_user + "@" + 
          proxy_host + "' " + ' ' + local_dir + remote_user + '@' + 
          remote_host + ':' + remote_dir)
    
    return

@check_same_dir
def put_dir_as_archive(local_dir: str, remote_dir: str, connection: Connection):
    '''
    Archive the directory 'local_dir' and send it to 'remote_dir'
    using the fabric 'put' function via the fabric connection 
    'connection' and unzip it there, deletes the archive on both ends
    '''
    zip_filename = local_dir + '.zip'
    local(f"cd {local_dir} && zip -r {zip_filename} .")
    connection.put(zip_filename, remote_dir)
    with connection.cd(remote_dir):
        connection.run(f"unzip -o {os.path.basename(zip_filename)}", hide='both')
        connection.run(f"rm -f {os.path.basename(zip_filename)}", hide='both')
    local(f"rm -f {zip_filename}")

    return

@check_same_dir
def get_dir_as_archive(local_dir: str, remote_dir: str, connection: Connection):
    '''
    Archive the directory 'remote_dir' and send it to 'local_dir'
    using the fabric 'get' function via the fabric connection 
    'connection' and unzip it there, deletes the archive on both ends
    '''
    zip_filename = remote_dir + '.zip'
    local_zip_filename = os.path.join(local_dir, os.path.basename(zip_filename))
    with connection.cd(remote_dir):
        connection.run(f"zip -r {zip_filename} .", hide='both')
    connection.get(zip_filename, local_zip_filename)
    connection.run(f"rm -f {zip_filename}", hide='both')
    local(f"cd {local_dir} && \
        unzip -o {os.path.basename(local_zip_filename)} && \
        rm -f {os.path.basename(local_zip_filename)}")
    
    return

def gather_files(dir: str, gather_dir: str, dir_pattern: str, file_list: list, remote_connection: Connection):
    '''
    Collects files with the names given in file_list from subfolders in dir with 
    the name pattern dir_pattern into the directory gather_dir via remote_connection 
    
    Permitted form of dir_pattern: 
    If the pattern is 'dir.*' with * being a consecutive numerical value, then just give 'dir'
    as a dir_pattern
    '''
    #Prepare absolut paths
    folder_pattern = os.path.join(dir, dir_pattern + ".*")

    log.info(f"Gather the files {file_list} from the folders {dir_pattern}.* into {gather_dir}")

    if (remote_connection != None):
        if exists(remote_connection, gather_dir):
            log.warning(f"{gather_dir} already exists!")
            return
            
        with remote_connection as c:
            c.run(f"mkdir {gather_dir}", hide='both')
            #Use the 'ls' command with the folder pattern to list matching folders
            #an -v1 to sort numerically 
            result = c.run(f"ls -v1 -d {folder_pattern}", hide='both')
            remote_directories = result.stdout.strip().split("\n")
            #Loop through the matched folders
            for i, remote_folder in enumerate(remote_directories, start=1):
                for file_to_rename in file_list:
                    #Define the new file name with the folder number
                    new_file_name = f"{file_to_rename}.{i}"

                    #Construct the full paths for the old and new file names
                    old_file_path = os.path.join(remote_folder, file_to_rename)
                    new_file_path = os.path.join(gather_dir, new_file_name)

                    #Copy the file
                    c.run(f"cp {old_file_path} {new_file_path}", hide='both')
    else:
        local(f"mkdir {gather_dir}")
        #Use the 'ls' command with the folder pattern to list matching folders
        #an -v1 to sort numerically 
        result = local(f"ls -v1 -d {folder_pattern}", hide=True)
        directories = result.stdout.strip().split("\n")
        #Loop through the matched folders
        for i, folder in enumerate(directories, start=1):
            for file_to_rename in file_list:
                #Define the new file name with the folder number
                new_file_name = f"{file_to_rename}.{i}"

                #Construct the full paths for the old and new file names
                old_file_path = os.path.join(folder, file_to_rename)
                new_file_path = os.path.join(gather_dir, new_file_name)

                #Copy the file
                local(f"cp {old_file_path} {new_file_path}")

    return