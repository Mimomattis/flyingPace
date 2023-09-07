import argparse
import glob
import os
import sys

import pandas as pd

from invoke import run as local #To run local commands like rsync function
from shutil import copyfile

def main(args):
    parser = argparse.ArgumentParser(prog="SvL", description="Script to start lammps run within the active learning scheme")

    parser.add_argument("--datafile", help="lammps data file", 
                        dest="datafile", type=str, required=True)

    parser.add_argument("--potentialfile", help="name of the potential file, without .yaml extension, default: output_potential", 
                        dest="potentialfile", nargs='?', type=str, default="output_potential")

    parser.add_argument("--element_list", help="list of elements", 
                        dest="element_list", type=str, required=True)

    parser.add_argument("--temp", help="temperature of the simulation, default 350",
                        dest="temp", type=float, default=350.0)
    
    parser.add_argument("--num_steps", help="number of steps of simulation, default 50000",
                        dest="stepnumber", type=int, default=50000)

    args_parse = parser.parse_args(args)

    datafile = args_parse.datafile
    potentialfile = args_parse.potentialfile
    element_list = args_parse.element_list
    temp = args_parse.temp
    stepnumber = args_parse.stepnumber

    calc_done_file = "CALC_DONE"
    calc_done = False
    if calc_done_file in  glob.glob(calc_done_file):
        print("The calculation was already done! Skipping the rest")
        sys.exit(0)

    generate_lammps_input(datafile, potentialfile, element_list, temp, stepnumber)

    local('sbatch runbatch-lammps')
        #See if this can be done better, right now CALC_DONE file is written
        #by the runbatch script
    local("while [ ! -f 'CALC_DONE' ]; do sleep 1; done")

    sys.exit(0)
    


def generate_lammps_input(datafile, potentialfile, element_list, temp, stepnumber):

    with open("INP-lammps", "r") as f:
        input_lammps_text = f.read()

    #Apperntly this is possible with .format
    input_lammps_text = input_lammps_text.replace("{{DATAFILE}}", str(datafile))
    input_lammps_text = input_lammps_text.replace("{{POTENTIALFILE}}", str(potentialfile))
    input_lammps_text = input_lammps_text.replace("{{ELEMENTLIST}}", element_list)
    input_lammps_text = input_lammps_text.replace("{{TEMP}}", str(temp))
    input_lammps_text = input_lammps_text.replace("{{NUM_STEPS}}", str(stepnumber))

    with open("INP-lammps", "w") as f:
        print(input_lammps_text, file=f)
    print("Input file is written into `INP-lammps`")
    


if __name__ == "__main__":
    main(sys.argv[1:])
