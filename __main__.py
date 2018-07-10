from DEBT_base.main import main as debt_main
from ISA_base.main import main as isa_main


if __name__ == "__main__":
    version = input("version? [debt/isa]: ").upper()
    if version == 'DEBT':
        print("Running debt version.")
        gamma = float(input("gamma: "))
        n_sim = int(input("n_sim: "))
        debt_main(version=version, gamma=gamma, n_sim=n_sim)
    elif version == 'ISA':
        print("Running ISA version.")
        gamma_max = float(input("gamma_max: "))
        gamma_step = float(input("gamma_step: "))
        isa_main(version=version, gamma_max=gamma_max, gamma_step=gamma_step)
    else:
        print(f"Version not recognized: {version}")
