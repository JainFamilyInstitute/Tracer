import argparse
from .main import main

VERSION = 'ISA'


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description='Run the ISA version of the model.')
    parser.add_argument('gamma_max', metavar='GAMMA_MAX', type=float, help='Maximum gamma for gamma loop.')
    parser.add_argument('gamma_step', metavar='GAMMA_STEP', type=float, help='Step increment for gamma loop.')
    args = parser.parse_args()
    main(version=VERSION, gamma_max=args.gamma_max, gamma_step=args.gamma_step)

