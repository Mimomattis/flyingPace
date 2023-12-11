#!/bin/bash -l
#
#SBATCH --job-name=runbatch-exploration
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
#SBATCH --partition=batch3
#
#SBATCH --cpus-per-task=1
#SBATCH --distribution=block:block:block
##SBATCH --hint=nomultithread
#SBATCH --export=none
#SBATCH --get-user-env
#
#-----------------------------------------------------------------------
  input="INP-lammps"
  output="OUT"
#
# set workdir: Only change if you know what you are doing
#
#  workdir="."
  workdir="/dev/shm/${SLURM_JOB_ID}"
#
#-----------------------------------------------------------------------
# nothing needs to be changed below this line
#-----------------------------------------------------------------------
  unset SLURM_EXPORT_ENV
#
# set number of processes per node:
#
  export ppn=${SLURM_NTASKS_PER_NODE}
#
  echo `scontrol show hostnames ${SLURM_JOB_NODELIST}`
  echo ${SLURM_JOB_PARTITION}
  echo "Number of OpenMP threads:     " ${SLURM_CPUS_PER_TASK}
  echo "Number of MPI procs per node: " ${SLURM_NTASKS_PER_NODE}
  echo "Number of MPI processes:      " ${SLURM_NTASKS}
#
# load modules
#
module purge
module load openmpi/intel/4.1.2-2021.4.0.3422-hpcx2.10
module load ucx/2.10
module load hcoll/2.10
module load compiler-rt/latest
module load tbb/latest

#
# set path for LAMMPS executable
#
  export LAMMPS="/path/to/lmp/"
  echo ${LAMMPS}
  export PAR_RUN="srun"
#  export PAR_RUN="mpirun"
  which ${PAR_RUN}
#
  export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
  export OMPI_MCA_hwloc_base_report_bindings=true
#
# copy to local workdir:
#
  if [ ${workdir} != "." ]; then
    mkdir -p ${workdir}
    cp -a * ${workdir}
    cd ${workdir}
    rm *.o${SLURM_JOB_ID} *.e${SLURM_JOB_ID}
  fi

#
  echo -n "Starting run at: "
  date

  $PAR_RUN $LAMMPS -var submitdir "${SLURM_SUBMIT_DIR}" -in ${input} > ${output}
  
  touch CALC_DONE

  echo -n "Stopping run at: "
  date
#
#
# move results and tidy up:
#
  echo
  if [ ${workdir} != "." ]; then
    cp -a * ${SLURM_SUBMIT_DIR}
    cd ${SLURM_SUBMIT_DIR}
    rm -rf ${workdir}
  fi
