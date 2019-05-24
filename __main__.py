
import argparse
from main import main

VERSION = 'ISA_MC_from_ids'

#SIDHYA change
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run the ISA version of the model.')

    #parser.add_argument('-n', '--n_sim', metavar='N_SIM', type=int, help='Number of MC loops.', default=10000)
    parser.add_argument('-g','--gamma', metavar='GAMMA', type=str, help='Gamma (float; if not specified, uses all of [1.0, 2.0, 3.0, 4.0])',default='ALL')

    args = parser.parse_args()
    gamma = args.gamma
    if gamma.upper() == "ALL":
        gamma = [1.0, 2.0, 3.0, 4.0]
    else:
        gamma = float(gamma)

    main(version=VERSION, gamma=gamma)
