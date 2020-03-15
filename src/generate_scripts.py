for k in [10,100]:
    for rho in [0,0.2,0.5]:
        for sig in [1,2,3]:
            for model in [1,2,3]:
                rho_a = str(rho)[-1]
                command = f"""#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=sched_mit_sloan_batch
#SBATCH --time=00-06:00
#SBATCH -o out/pso_out_{k}_{rho_a}_{sig}_{model}.txt
#SBATCH -e err/pso_err_{k}_{rho_a}_{sig}_{model}.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fsusan@mit.edu

module load python/3.6.3

python main.py {k} {rho} {sig} {model} > results/result_pso_{k}_{rho_a}_{sig}_{model}.txt"""
                with open(f"pso_{k}_{rho_a}_{sig}_{model}.sh", "w") as text_file:
                    text_file.write(command)


# for f in *.sh; do sbatch $f; done
# srun --pty --x11 --cpus-per-task=9 --mem=16G --constraint="centos7" --time=1-00:00 --partition=sched_mit_sloan_interactive julia PSO.jl 12 1 0 > results/result_pso_12_1_0_1.txt

