#!/bin/bash
#SBATCH -J halo_project
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --time=10:00:00
#SBATCH --mem=36G
#SBATCH --output=logs/outputs.out
#SBATCH --error=logs/errors.err
#SBATCH --mail-user=nvanalfen2@gmail.com
#SBATCH --mail-type=ALL

module load conda
conda activate diffsky

srun python3 project_lc_halos.py