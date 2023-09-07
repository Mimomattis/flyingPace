import argparse
import glob
import os
import sys

import pandas as pd

from invoke import run as local #To run local commands like rsync function
from shutil import copyfile

def main(args):
    parser = argparse.ArgumentParser(prog="SvP", description="Script to start pacemaker within the active learning scheme")

    parser.add_argument("--dataset", help="data set file, pickle file with .pckl.gzip extension", 
                        dest="dataset", type=str, required=True)

    parser.add_argument("--pot", help="previous potential, if available, default: output_potential.yaml", 
                        dest="potential", nargs='?', type=str, default="output_potential.yaml" )

    parser.add_argument("--test_size", help="test set fraction or size, default 0.05", 
                        dest="test_size", type=float, default=0.05 )

    parser.add_argument("--num_func", help="number of functions per element, default 700",
                        dest="num_func", type=int, default=700 )
    
    parser.add_argument("--cutoff", help="cutoff in angstrom, default 7.0",
                        dest="cutoff", type=float, default=7.0 )
    
    parser.add_argument("--weighting", help="enter weighting scheme type - `uniform` or `energy`, default 'uniform' ",
                        dest="weighting", type=str, default="uniform" )
    
    parser.add_argument("--kappa", help="enter value for kappa, 'auto' or value between 0 and 1, default 0.3 ",
                        dest="kappa", default=0.3 )
    
    parser.add_argument("--max_iter", help="maximum number of iterations, default 2000 ",
                        dest="max_iter", type=int, default=2000 )

    args_parse = parser.parse_args(args)

    train_filename = args_parse.dataset
    prev_potfilename = args_parse.potential
    testset_size_inp = args_parse.test_size
    number_of_functions_per_element = args_parse.num_func
    cutoff = args_parse.cutoff
    weighting_inp = args_parse.weighting
    kappa = args_parse.kappa
    max_iter = args_parse.max_iter

    gpu_ace_activeset = '/home/titan/nfcc/nfcc010h/software/privat/conda/envs/ace/bin/pace_activeset'

    calc_done_file = "CALC_DONE"
    calc_done = False
    if calc_done_file in  glob.glob(calc_done_file):
        print("The calculation was already done! Skipping the rest")
        sys.exit(0)

    generate_pace_input(train_filename, testset_size_inp,
                        number_of_functions_per_element, cutoff, weighting_inp, kappa, max_iter)

    with open("runbatch-ace", "r") as f:
        runbatch_text = f.read()
    runbatch_text =runbatch_text.replace("{{PREV_POT}}", prev_potfilename)
    with open("runbatch-ace", "w") as f:
        print(runbatch_text, file=f)
    print("Runbatch file written in 'runbatch-ace'")

    local("sbatch runbatch-ace")
    #See if this can be done better, right now CALC_DONE file is written
    #by the runbatch script
    local("while [ ! -f 'CALC_DONE' ]; do sleep 1; done")
    local(gpu_ace_activeset + " -d fitting_data_info.pckl.gzip output_potential.yaml")

    sys.exit(0)
    


def generate_pace_input(train_filename, testset_size_inp,
                        number_of_functions_per_element, cutoff, weighting_inp, kappa, max_iter):

    # checking dataset
    df = pd.read_pickle(train_filename, compression="gzip")
    
    if 'ase_atoms' in df.columns:
        print("Determining available elements...")
        elements_set = set()
        df["ase_atoms"].map(lambda at: elements_set.update(at.get_chemical_symbols()));
        elements = sorted(elements_set)
        print("Found elements: ", elements)
    else:
        print("ERROR! No `ase_atoms` column found")
        sys.exit(1)

    print("Number of elements: ", len(elements))
    print("Elements: ", elements)

    # weighting scheme
    default_energy_based_weighting = """{ type: EnergyBasedWeightingPolicy, DElow: 1.0, 
                                    DEup: 10.0, DFup: 50.0, DE: 1.0, DF: 1.0, wlow: 0.75, 
                                    energy: convex_hull, reftype: all,seed: 42}"""
    weighting = None
    if weighting_inp not in ['uniform', 'energy']:
        print("ERROR! No valid weighting_scheme")
        sys.exit(1)
    if weighting_inp == "energy":
        weighting = default_energy_based_weighting
        print("Use EnergyBasedWeightingPolicy: ", weighting)
    else:
        weighting = None
        print("Use UniformWeightingPolicy")

    with open("input.yaml", "r") as f:
        input_yaml_text = f.read()

    input_yaml_text = input_yaml_text.replace("{{ELEMENTS}}", str(elements))
    input_yaml_text = input_yaml_text.replace("{{CUTOFF}}", str(cutoff))
    input_yaml_text = input_yaml_text.replace("{{DATAFILENAME}}", train_filename)
    input_yaml_text = input_yaml_text.replace("{{number_of_functions_per_element}}",
                                              "number_of_functions_per_element: {}".format(
                                                  number_of_functions_per_element))
    input_yaml_text = input_yaml_text.replace("{{KAPPA}}", str(kappa))
    input_yaml_text = input_yaml_text.replace("{{MAXITER}}", str(max_iter))

    if weighting:
        input_yaml_text = input_yaml_text.replace("{{WEIGHTING}}", "weighting: " + weighting)
    else:
        input_yaml_text = input_yaml_text.replace("{{WEIGHTING}}", "")

    if testset_size_inp > 0:
        input_yaml_text = input_yaml_text.replace("{{test_size}}", "test_size: {}".format(testset_size_inp))
    else:
        input_yaml_text = input_yaml_text.replace("{{test_size}}", "")

    with open("input.yaml", "w") as f:
        print(input_yaml_text, file=f)
    print("Input file is written into `input.yaml`")
    


if __name__ == "__main__":
    main(sys.argv[1:])
