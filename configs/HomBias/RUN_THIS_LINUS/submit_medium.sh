cfgs=( # for Spasm alone
"/slurm-storage/linbao/thesis_code_1/GRIT/configs/HomBias/RUN_THIS_LINUS/drop.yaml"
"/slurm-storage/linbao/thesis_code_1/GRIT/configs/HomBias/RUN_THIS_LINUS/homWide.yaml"
"/slurm-storage/linbao/thesis_code_1/GRIT/configs/HomBias/RUN_THIS_LINUS/long.yaml"
"/slurm-storage/linbao/thesis_code_1/GRIT/configs/HomBias/RUN_THIS_LINUS/small.yaml"
)

for c in "${cfgs[@]}"; do
    cat <<EOT > temp_submit.job
#!/bin/bash
#SBATCH -N 1      # nodes requested
#SBATCH --time=23:59:59
#SBATCH --gres=gpu:1
#SBATCH --qos=medium
#SBATCH --output=/slurm-storage/linbao/thesis_code_1/outputs/%x_%j.out   # standard output
#SBATCH --error=/slurm-storage/linbao/thesis_code_1/outputs/%x_%j.err    # standard error

echo "$c"
srun -u /slurm-storage/miniconda3/envs/HomGRIT/bin/python "/slurm-storage/linbao/thesis_code_1/GRIT/main.py" --cfg "$c" --repeat 1 wandb.use True
EOT


    sleep 1s 
    sbatch temp_submit.job
    #cat temp_submit.job

done

