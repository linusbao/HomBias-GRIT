cfgs=(
"/slurm-storage/linbao/thesis_code_1/GraphGPS/configs/GPS/ICLR/WLtree-cossinEnc/MLP/hcpe/big22.yaml"
"/slurm-storage/linbao/thesis_code_1/GraphGPS/configs/GPS/ICLR/WLtree-cossinEnc/MLP/hcpe/big24.yaml"
"/slurm-storage/linbao/thesis_code_1/GraphGPS/configs/GPS/ICLR/WLtree-cossinEnc/MLP/hcpe/small22.yaml"
"/slurm-storage/linbao/thesis_code_1/GraphGPS/configs/GPS/ICLR/WLtree-cossinEnc/MLP/hcpe/small24.yaml"
)

for c in "${cfgs[@]}"; do
    cat <<EOT > temp_submit.job
#!/bin/bash
#SBATCH -N 1      # nodes requested
#SBATCH --time=2:59:59
#SBATCH --gres=gpu:1
#SBATCH --qos=short
#SBATCH --output=/slurm-storage/linbao/thesis_code_1/outputs/%x_%j.out   # standard output
#SBATCH --error=/slurm-storage/linbao/thesis_code_1/outputs/%x_%j.err    # standard error

echo "$c"
srun -u /slurm-storage/linbao/.conda/envs/GPS-envV5/bin/python "/slurm-storage/linbao/thesis_code_1/GRIT/main.py" --cfg "$c" --repeat 1 wandb.use True
EOT
    sleep 1s 
    sbatch temp_submit.job
    #cat temp_submit.job
done

