cloud ml-engine jobs submit training object_detection_`date +%s` \
    --job-dir=gs://${GCS_BUCKET}/${TRAIN_DIR} \
    --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz \
    --module-name object_detection.train \
    --region us-central1 \
    --config ${CLOUD_PATH} \
    -- \
    --train_dir=gs://${GCS_BUCKET}/${TRAIN_DIR} \
    --pipeline_config_path=gs://${GCS_BUCKET}/${PIPELINE_CONFIG_PATH}