import os
from main import main

if __name__ == "__main__":
    alt_degs = [2,3,4]
    id_fn = "Mixture_IDs.xls"
    gammas = [1.0, 2.0, 3.0, 4.0]
    
    try:
        assert os.path.isfile(id_fn)
        id_fp = id_fn
    except:
        base_path = os.path.dirname(__file__)
        id_fp = os.path.join(base_path, 'data', id_fn)
        assert os.path.isfile(id_fp)

    main(id_fp=id_fp, alt_degs=alt_degs, gammas=gammas)
