python object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path=gs://tensorflow-magistra-traindata/ssd_mobilenet_v1/optimized.config \
    --trained_checkpoint_prefix  gs://tensorflow-magistra-traindata/train_checkpoint2/model.ckpt-35950 \
    --output_directory gs://tensorflow-magistra-traindata/export/trs_model_large_35950