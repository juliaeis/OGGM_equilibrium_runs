#!/bin/bash
#
#SBATCH --job-name=01_eq
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --array=2506-26897
#SBATCH --time=3-00:00:00
#SBATCH --mail-user=jeis@uni-bremen.de
#SBATCH --mail-type=ALL


# Abort whenever a single step fails. Without this, bash will just continue on errors.

set -e

# activate the conda environement
eval "$(conda shell.bash hook)"
conda activate oggm_env

# On every node, when slurm starts a job, it will make sure the directory
# /work/username exists and is writable by the jobs user.
# We create a sub-directory there for this job to store its runtime data at.
export WORKDIR="/work/${SLURM_JOB_USER}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}/run_CMIP6_geod"
export PROJDIR="$HOME/OGGM_equilibrium_runs"
export OUTDIR="$PROJDIR/out/"

mkdir -p "$WORKDIR"
mkdir -p "$OUTDIR"

export S_WORKDIR
export OUTDIR

echo "Workdir for this run: $WORKDIR"

export REGION=01
export JOB_NR=$SLURM_ARRAY_TASK_ID

export OGGM_DOWNLOAD_CACHE="/home/data/download"
export OGGM_DOWNLOAD_CACHE_RO=1


# Run the actual job. The srun invocation starts it as individual step for slurm.
srun -n 1 -c "${SLURM_JOB_CPUS_PER_NODE}" python3 ../cluster.py
# Print a final message so you can actually see it being done in the output log.
echo "DONE"

# Once a slurm job is done, slurm will clean up the /work directory on that node from any leftovers $
# So copy any result data you need from there back to your home dir!
# $SLURM_SUBMIT_DIR points to the directory from where the job was initially commited.

echo "Start copying..."
# Copy any neccesary result data.
cp -r "${WORKDIR}" "${OUTDIR}"
echo "SLURM DONE"
