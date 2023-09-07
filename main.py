import getpass  # for getpass.getuser()
import logging 
import socket
import sys

from fabric import Connection
from invoke import run as local #To run local commands like rsync function

def main():

    #Debugging option to not run the calculations
    dry_run = False   

    #TODO: SETUP on big inputfile with configurations for:
    #   - SSH settings
    #   - All working directories (Needed locally)
    #   - PACEMAKER settings (Needed on GPU host)
    #   - DFT settings (Needed on CPU host)
    #   - LAMMPS settings (Needed on CPU host)

    #Setup for the ssh connection
    user_name = 'nfcc010h'

    cpu_host = 'fritz'
    gpu_host = 'tinyx'

    jump_host = 'cshpc.rrze.fau.de'

    cpu_connection = Connection(host=cpu_host, 
                                user=user_name, 
                                gateway=Connection(
                                        host=jump_host,
                                        user=user_name
                                        ),
                                )
    gpu_connection = Connection(host=gpu_host, 
                                user=user_name,
                                gateway=Connection(
                                        host=jump_host,
                                        user=user_name
                                        ),
                                )
    

    #Setup for working in the right directories

    script_directory = '/ccc160/mgossler/phd/work/project-ace/flyingPACE/scripts/'
    template_directory = '/ccc160/mgossler/phd/work/project-ace/flyingPACE/templates/'

    local_working_dir = '/ccc160/mgossler/phd/work/project-ace/test_directory/'
    gpu_working_dir = '/home/titan/nfcc/nfcc010h/work/project-ace/test_directory/'
    cpu_working_dir = '/home/titan/nfcc/nfcc010h/work/project-ace/test_directory/'

    #TODO:Is there a better way to do this
    gpu_python_interpreter = '/home/titan/nfcc/nfcc010h/software/privat/conda/envs/ace/bin/python3'
    cpu_python_interpreter = '/home/titan/nfcc/nfcc010h/software/privat/conda/envs/ace/bin/python3'

    remote_train = gpu_working_dir + 'train/'
    remote_md = cpu_working_dir + 'md/'
    remote_dft = cpu_working_dir + 'dft/'

    local_train = local_working_dir + 'train/'
    local_md = local_working_dir + 'md/'
    local_dft = local_working_dir + 'dft/'

    #Setup logging

    class LoggerWriter:
        def __init__(self, level):
        # self.level is really like using log.debug(message)
        # at least in my case
            self.level = level

        def write(self, message):
        # if statement reduces the amount of newlines that are
        # printed to the logger
            if message != '\n':
                self.level(message)

        def flush(self):
        # create a flush method so things can be flushed when
        # the system wants to. Not sure if simply 'printing'
        # sys.stderr is the correct way to do it, but it seemed
        # to work properly for me.
            self.level(sys.stderr)

    hostname = socket.gethostname()
    username = getpass.getuser()
    LOG_FMT = '%(asctime)s %(levelname).1s - %(message)s'.format(hostname)
    logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%Y/%m/%d %H:%M:%S")
    log = logging.getLogger()
    log_file_name = 'log.txt'
    log.info("Redirecting log into file {}".format(log_file_name))
    fileh = logging.FileHandler(log_file_name, 'a')
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter(LOG_FMT)
    fileh.setFormatter(formatter)
    log.addHandler(fileh)
    sys.stdout = LoggerWriter(log.debug)
    sys.stderr = LoggerWriter(log.warning)

    if not dry_run:
        #Start the first pacemaker run

        gpu_connection.run('if [ ! -d "' + remote_train + '" ]; then mkdir ' + remote_train + '; fi')
        gpu_connection.put(local_working_dir + 'S.pckl.gzip', remote_train)
        gpu_connection.put(template_directory + 'input.yaml', remote_train)

        #The runbatch scripts should properbly also lie in the local working directory

        gpu_connection.put(template_directory + 'runbatch-ace', remote_train)
        #TODO: Is coping the scripts the best way to do it
        gpu_connection.put(script_directory + 'svp.py', remote_train)

        with gpu_connection.cd(remote_train):
            #TODO: Run svp.py with the configurations from the input file
            #Also, it probaply would be better to run the calculation from the script files themselves
            gpu_connection.run(gpu_python_interpreter + ' svp.py --dataset S.pckl.gzip --max_iter 10')
            gpu_connection.run('rm -f svp.py')

        local('if [ ! -d "' + local_train + '" ]; then mkdir ' + local_train + '; fi')
        rsync_from_remote(user_name, jump_host, gpu_host, local_train, remote_train)

    #Start the first Lammps run

    cpu_connection.run('if [ ! -d "' + remote_md + '" ]; then mkdir ' + remote_md + '; fi')
    cpu_connection.put(local_working_dir + 's.data', remote_md)
    cpu_connection.put(template_directory + 'INP-lammps', remote_md)
    cpu_connection.put(template_directory + 'runbatch-lammps', remote_md)
    cpu_connection.put(local_train + 'output_potential.yaml', remote_md)
    cpu_connection.put(local_train + 'output_potential.asi', remote_md)
    cpu_connection.put(script_directory + 'svl.py', remote_md)

    with cpu_connection.cd(remote_md):
        cpu_connection.run(cpu_python_interpreter + ' svl.py --datafile s.data --potentialfile output_potential --element_list "S"')
        cpu_connection.run('rm -f svl.py')

    local('if [ ! -d "' + local_md + '" ]; then mkdir ' + local_md + '; fi')
    rsync_from_remote(user_name, jump_host, gpu_host, local_md, remote_md)


    

#Two wrappers for the built in rsync functions

def rsync_from_remote(remote_user, proxy_host, remote_host, local_dir, remote_dir):
    local("rsync -azv -e 'ssh -A -J " + remote_user + "@" + 
          proxy_host + "' " + remote_user + '@' + 
          remote_host + ':' + remote_dir + ' ' + local_dir)
    
def rsync_to_remote(remote_user, proxy_host, remote_host, local_dir, remote_dir):
    local("rsync -azv -e 'ssh -A -J " + remote_user + "@" + 
          proxy_host + "' " + ' ' + local_dir + remote_user + '@' + 
          remote_host + ':' + remote_dir)
    

#Parse input
    

if __name__ == "__main__":
    main()
