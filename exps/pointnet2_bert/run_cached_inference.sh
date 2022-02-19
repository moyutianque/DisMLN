work_path=$(dirname $0)
now=$(date +%s)

MASTER_PORT=8375 TOKENIZERS_PARALLELISM=false PYTHONPATH=./ python -u \
run.py --exp-config configs/dis_mlnv1_cached.yaml \
--run-type inference \
LOG_DIR tmp/ \
EVAL_PATH "tmp/PointNet_Transformer/version_1/checkpoints/epoch=1-step=9356.ckpt"
#\
#2>&1 | tee $work_path/train.$now.log.out
