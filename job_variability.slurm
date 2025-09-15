#!/bin/bash
#SBATCH --job-name=VariabilityOfLightCurve
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=03:00:00               
#SBATCH --partition=cpu-preempt       
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1             
#SBATCH --mem=64G                     
#SBATCH --constraint="scratch"
#SBATCH --account=bekt-delta-cpu      

# Activar entorno Conda
source /u/drojaspaternina/anaconda3/bin/activate
conda activate Entorno_aart           

echo "El trabajo empieza en $(hostname)"
srun python FastLightMovie.py             
