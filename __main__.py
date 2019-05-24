import os
import argparse
from main import main

VERSION = 'ISA_MC_from_ids'

#SIDHYA change
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run the ISA version of the model.')

    parser.add_argument('-i', '--id_file', metavar='ID_FN', type=str, help='ID file')
    parser.add_argument('-g','--gamma', metavar='GAMMA', type=str, help='Gamma (float; if not specified, uses all of [1.0, 2.0, 3.0, 4.0])',default='ALL')

    args = parser.parse_args()
    gamma = args.gamma
    id_fn = args.id_file
    if gamma.upper() == "ALL":
        gamma = [1.0, 2.0, 3.0, 4.0]
    else:
        gamma = float(gamma)

    try:
        assert os.path.isfile(id_fn)
    except:
        base_path = os.path.dirname(__file__)
        id_fn = os.path.join(base_path, 'data', id_fn)
        assert os.path.isfile(id_fn)
    main(version=VERSION, id_fn=id_fn, gamma=gamma)
