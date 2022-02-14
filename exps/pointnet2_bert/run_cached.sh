work_path=$(dirname $0)
model_name='mln_v1.8_wp_cma128_down2x_finetune'
now=$(date +%s)

MASTER_PORT=8375 TOKENIZERS_PARALLELISM=false PYTHONPATH=./ python -u \
run.py --exp-config configs/dis_mlnv1_cached.yaml \
--run-type train \
LOG_DIR tmp/
#\
#2>&1 | tee $work_path/train.$now.log.out
