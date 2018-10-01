
import argparse
from main import main

VERSION = 'ISA_MC'

#SIDHYA change
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run the ISA version of the model.')

    parser.add_argument('n_sim', metavar='N_SIM', type=int, help='Number of MC loops.')
    parser.add_argument('-g','--gamma', metavar='GAMMA', type=str, help='Gamma (float; if not specified, uses all of [1.4, 2.2, 3.5, 4.2, 6.7])',default='ALL')

    args = parser.parse_args()
    gamma = args.gamma
    if gamma.upper() == "ALL":
        gamma = [1.4, 2.2, 3.5, 4.2, 6.7]
    else:
        gamma = float(gamma)

    main(version=VERSION, n_sim=args.n_sim,gamma=gamma)
