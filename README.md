
# flyingPace

flyingPace is a tool for fitting ACE potentials on-the-fly using pacemaker for the fitting,  
a DFT code for data generation and LAMMPS for structural exploration  

## Installation 

Clone the repo, move to the directory and type 

`pip install --upgrade .`

## Usage

Invoke the main script by creating a new directory with a flyingPace input file,  
slurm scripts for the initial AIMD, SCF, pacemaker and LAMMPS runs (see examples directory).  
Move to the directory and type

`flyingPace -ip flyingPACE-in.yaml`

with 'flyingPACE-in.yaml' being the name of the flyingPace input file

## Input file

The following sections and input parameters can be given or are required in the flyingPace input file

### 'manager' section 

the manager section is started with  

`manager:`

Possible input parameters:  

`  startGen: 0` (required, default: 0)  

- Defines the learning generation from which learning is started. If 'gen{n}' is already existent in the  
local working directory, learning can be started from 'gen{n+1}'

'  finalGen : 10' (required, default: 10)  

- Defines the final learning generation

'  systemName: str' (required)

- Short descriptive name for the system

'  CPUHost: str, None' (required)

- `str` Hostname of the machine, on which the cpu calculations are done (DFT and LAMMPS calculations).  
Assumes that there is a functioning installation of LAMMPS and the used DFT code on the cpu machine,  
there are workingslurm scripts 'explorationRunScript' for LAMMPS and 'aimdRunScript' and 'scfRunScript' 
for the DFT code. Assumes also a working authentification on the cpu machine with ssh-key

- `None` The cpu calculations are done (DFT and LAMMPS calculations) are done locally

'  CPUUser: str' (required)

- Username for the cpu machine 

'  CPUJumpHost' (optional)

- If the connection to the cpu machine is only opssible via a jump server, provide the hostname with this option 

