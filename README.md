python3 ./object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=/home/tsamsiyu/Code/Projects/tensorflow_magistra/data/ssd_mobilenet_v1_traffic_signs.config \
    --train_dir=/home/tsamsiyu/Code/Projects/tensorflow_magistra/training

python object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --checkpoint_dir=${PATH_TO_TRAIN_DIR} \
    --eval_dir=${PATH_TO_EVAL_DIR}
