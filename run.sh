#!/bin/bash -x
# create slurm input file for job submission
cat > runDES.slurm << EOF
#!/bin/bash

#SBATCH --mem=100000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:k80
#SBATCH --time=5:00:00
#SBATCH --account=dftclass

#SBATCH -e errfile

module purge
module load anaconda
module load singularity
module load tensorflow/1.12.0-py36


singularity-gpu run /scratch/pz4ee/tensorflow-1.12.0-py36.simg main.py

EOF

sbatch runDES.slurm
