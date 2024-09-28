cfgs=( # for next three repeats of wide model with Spasm alone
"/slurm-storage/linbao/thesis_code_1/GRIT/configs/HomBias/RUN_THIS_EMILY/hom-wide.yaml"
)

seeds=(14 48 96)

for c in "${cfgs[@]}"; do
    for s in "${seeds[@]}"; do
        cat <<EOT > temp_submit.job
#!/bin/bash
#SBATCH -N 1      # nodes requested
#SBATCH --time=23:59:59
#SBATCH --gres=gpu:1
#SBATCH --qos=medium
#SBATCH --output=/slurm-storage/linbao/thesis_code_1/outputs/%x_%j.out   # standard output
#SBATCH --error=/slurm-storage/linbao/thesis_code_1/outputs/%x_%j.err    # standard error

echo "$c"
srun -u /slurm-storage/linbao/.conda/envs/GPS-envV5/bin/python "/slurm-storage/linbao/thesis_code_1/GRIT/main.py" --cfg "$c" --repeat 1 wandb.use True seed $s
EOT
        sleep 1s 
        sbatch temp_submit.job
        #cat temp_submit.job
    done
done

