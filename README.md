
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

Possible input parameter:  

`startGen: 0` (required, default: 0)
