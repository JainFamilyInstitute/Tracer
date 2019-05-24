import os
import argparse
from main import main

VERSION = 'ISA_MC_from_ids'

#SIDHYA change
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run the ISA version of the model.')

    parser.add_argument('-i', '--id_file', metavar='ID_FN', type=str, help='ID file', default='Mixture_IDs.xls')
    parser.add_argument('-g','--gamma', metavar='GAMMA', type=str, help='Gamma (float; if not specified, uses all of [1.0, 2.0, 3.0, 4.0])',default='ALL')
    parser.add_argument('-d', '--alt_deg', metavar='ALT_DEG', type=int, help='Alt_deg to use (2: High School, 3: Some College, 4: College Grad', default=2)
    args = parser.parse_args()
    gamma = args.gamma
    id_fn = args.id_file
    alt_deg = int(args.alt_deg)
    if gamma.upper() == "ALL":
        gamma = [1.0, 2.0, 3.0, 4.0]
    else:
        gamma = float(gamma)

    if alt_deg not in [3, 4]:
        raise(ValueError("Alt_deg should be set to 3 or 4"))


    try:
        assert os.path.isfile(id_fn)
    except:
        base_path = os.path.dirname(__file__)
        id_fn = os.path.join(base_path, 'data', id_fn)
        assert os.path.isfile(id_fn)
    main(version=VERSION, id_fn=id_fn, alt_deg=alt_deg, gamma=gamma)
