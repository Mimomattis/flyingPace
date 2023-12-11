# flyingPace

flyingPace is a tool for fitting ACE potentials on-the-fly using pacemaker for the fitting,  
a DFT code for data generation and LAMMPS for structural exploration  

## Installation 

Clone the repo, move to the directory and type 

`pip install --upgrade .`

## Usage

Invoke the main script by creating a new directory with a flyingPace input file, the necessary starting files
(description below) and slurm scripts for the initial AIMD, SCF, pacemaker and LAMMPS runs (see examples directory).  
Move to the directory and type

`flyingPace -ip flyingPACE-in.yaml`

with 'flyingPACE-in.yaml' being the name of the flyingPace input file

## Input file

The following sections and input parameters can be given or are required in the flyingPace input file

### 'manager' section 

the manager section is started with  

`manager:`

Possible input parameters:  

`startGen: int` (default: 0)

- Defines the learning generation from which learning is started. If 'gen{n}' is already existent in the  
local working directory, learning can be started from 'gen{n+1}'

` finalGen: int` (default: 10)  
not
- Defines the final learning generation

`systemName: str` (required)

- Short descriptive name for the system

`startFile: str` (required if `startGen = 0`)

- File from which the flyingPace run is started. Depending on the file type, other files are required

    - `{filename}.data: LAMMPS data file` If the run is started from a LAMMPS data file, an input file for an initial AIMD run is written depending on the parameters given in the `dft` section, to generate an initial dataset.
    - `DFT input file` The run is started from an initial AIMD calculation to generate the initial dataset.
    - `{filename}.pckl.gzip: dataset file` Other required files: `{filename}.data: LAMMPS data file`. The run is started from a pickled dataframe of datapoint in the format required by pacemaker. An initial potential is trained from this dataset. The subsequent exploration run is started from the given LAMMPS data file.
    - `{filename}.yaml: ACE potential` Other required files: `{filename}.data: LAMMPS data file`, `{filename}.pckl.gzip: dataset file`, `{filename}.asi: active set file`. The run is started from a given ACE potential with an exploration run from the given LAMMPS data file.

`localWorkingDir: str` (required) 

- Path of directory where flyingPace is executed and all needed files are stored

`DFTWorkingDir/ExplorationWorkingDir/TrainWorkingDir: str` (required)

- Path to the working dir on the DFTHost/TrainHost/ExplorationHost

`DFTHost/TrainHost/ExplorationHost: str, 'local'` (default 'local')

- `str` Hostname of the machine, on which the DFT/train/exploration calculations are done. 
        Assumes a working authentification on the remote machine with ssh-key

- `'local'` The DFT/train/exploration calculations are done locally

`DFTUser/TrainUser/ExplorationUser: str` (required if `DFTHost/TrainHost/ExplorationHost` not `'local'`)

- Username for the remote machine machine 

`DFTJumpHost/TrainJumpHost/ExplorationJumpHost: str` (optional)

- If the connection to the remote machine is only possible via a jump server, provide the hostname with this option

`DFTPython/TrainPython/ExplorationPython: str` (required)

- Path to the python interpreter on DFT/train/exploration host

### 'pacemaker' section 

the manager section is started with  

`pacemaker:`

Possible input parameters:  

`pacemakerRunScript: str` (required)

- Name of the submission script for the pacemaker run

`cutoff: float` (default = 7.0)

- Cutoff radius in angstrom for the ACE potential

`numberOfFunctions: int` (default = 700)

- Number of basis functions per atom species in for the ACE potential

`testSize: float` (default = 0)

- Fraction of the dataset used as a test set for the ACE potential


`weighting: 'uniform' or 'energy'` (default = 'uniform')

- Refer to the pacemaker manual

`kappa: float` (default = 0.3)

- Relative force weight, kappa = 1.0 is forces only fit, kappa = 0.0 is energies only fit

`maxNumIterations: int` (default = 500)

- Maximum number of iterations during the training cycle

`batchSize: int` (default = 500)

- Batch size for the pacemaker run

`referenceEnergyMode: 'auto' or 'singleAtomEnergies'` (default = 'auto')

- Refer to the pacemaker manual

if `referenceEnergyMode == 'singleAtomEnergies'`:

- `referenceEnergies: dict` (required)

    - Dictionary of atomic species and corresponding reference energies in eV:

        Bi: -1920.6066

### 'exploration' section 

the exploration section is started with  

`exploration:`

Possible input parameters:  

`explorationRunScript: str` (required)

- Name of the submission script for the exploration run

`select: int` (optional)

- If given, the pace_select utility is used to select the given number of structures from all found extrapoative structures

#### 'explorationParams' subsection 

the explorationParams subsection is started with  

`explorationParams:`

within the exploration section 

Possible input parameters:

`steps: int` (default: 50000)

- Number of steps of the exploration run

`timestep: float` (default 5.0e-4)

- Timestep (in units 'metal') of the exploration run

`trjStep: int` (default 100)

- Trajectory is written every 'trjStep' timesteps

`thermo: int` (default: 100)

- Thermodynamic info is printed every 'thermo' timesteps

`explorationStyle: 'stop' or 'noStop'` (default: 'stop')

- Wheter to stop or continue the exploration run after the first structure is encountered where the upper gamma value is exceeded

`gammaStride: int` (default: 10)

- The gamma parameter is evaulauted every `gammaStride` time steps

`lowerGamma: int` (default: 5)

- Lower boundary of the gamma range of extrapolative structures

`upperGamma: int` (default: 25)

- Upper boundary of the gamma range of extrapolative structures 

`runType: 'NVT' or NPT` (required)

- `NVT:` Constant volume md
- `NPT:` Constant pressure md

if `runType == 'NPT'`:

- `nptMode: 'iso' or 'aniso'` (default: 'iso')

    - `iso`: Pressure is controlled isotropically
    - `aniso`: Pressure is controlled anisotropically
    

- `press: float`  (default: 1.0)

    - Pressure of the exploration run
 
- `pressRamp: string` (optional)

    - String of two pressures like '1.0 2.0', marking the start and end pressure of the exploration run. If not in use, defaulting to the `press` parameter

`temp: float` (default: 400.0)

- Temperature of the exploration run

`tempRamp: string` (optional)

- String of two temperatures like '300.0 600.0', marking the start and end temperature of the exploration run. If not in use, defaulting to the `temp` parameter

### 'dft' section 

the dft section is started with  

`dft:`

Possible input parameters:  

`dftCode: 'CPMD'` (required)

- Decides which DFT code is used to generate data

`scfRunScript: str` (required)

- Name of the submission script for the SCF runs

if `startFile == 'DFT input file'`:

- `aimdRunScript: str` (required)

    - Name of the submission script for the initial AIMD run
 
`maxScfRuns: int` (default: 100)

- Maximum of SCF calculations performed each learning generation

#### 'dftParams' subsection 

the dftParams subsection is started with  

`dftParams:`

within the dft section 

Possible input parameters:

**For SCF and AIMD caclulations**:

`maxIter: int` (default: 100)

- Maximum number of iterations for each SCF calculation

`convOrbital: float` (default: 1.0e-6)

- Convergence criterion for each SCF calculation

`splinePoints: int` (default: 5000)

- Number of spline points

`pwCutoff: float` (default: 30)

- Energy cutoff for the plane wave basis in rydberg

`functional: 'pbe' or 'pbeSol'` (required)

- Functional for DFT calculations

`vdwCorrection: bool` (default: false)

- Wheter to use VdW corecction or not

`pseudopotentials: dict` (required)

- Dictionary of atomic species and corresponding pseudopotential names:

    Bi: 'Bi.uspp736.pbesol'

**Only for AIMD calculation**:

`maxStep: int` (required)

- Maximum number of steps for AIMD

`timeStep: int` (required)

- Time step for AIMD

`trajStep: int` (required)

- Trajectory is written every 'trjStep' timesteps

`temp: float` (required)

- Temperature for AIMD

`noseParams: str` (required)

- Nose parameter for AIMD