
import argparse
from main import main

VERSION = 'DEBT'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run the DEBT version of the model.')
    parser.add_argument('gamma', metavar='GAMMA', type=float, help='Gamma')
    parser.add_argument('n_sim', metavar='N_SIM', type=int, help='Number of MC loops.')
    args = parser.parse_args()
    main(version=VERSION, gamma=args.gamma, n_sim=args.n_sim)
