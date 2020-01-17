import os
import argparse
from main import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run the model.')
    parser.add_argument('-i', '--id_file', metavar='ID_FN', type=str, help='ID file', default='Mixture_IDs.xls')
    parser.add_argument('-g','--gamma', metavar='GAMMA', type=str, help='gamma (float; if not specified, uses all of [1.0, 2.0, 3.0, 4.0])',default='ALL')
    # parser.add_argument('-d', '--alt_deg', metavar='ALT_DEG', type=int, help='Education level to use (2: High School, 3: Some College, 4: College Grad); if not specified, uses all.', default='ALL')
    alt_deg = [2,3,4]
    args = parser.parse_args()
        
    id_fn = str(args.id_file)
    
    gamma = args.gamma
    if gamma.upper() == "ALL":
        gamma = [1.0, 2.0, 3.0, 4.0]
    else:
        gamma = float(gamma)
    
    try:
        assert os.path.isfile(id_fn)
        id_fp = id_fn
    except:
        base_path = os.path.dirname(__file__)
        id_fp = os.path.join(base_path, 'data', id_fn)
        assert os.path.isfile(id_fp)

    main(id_fp=id_fp, alt_deg=alt_deg, gamma=gamma)
