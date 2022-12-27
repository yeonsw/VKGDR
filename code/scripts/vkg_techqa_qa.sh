DATA_DIR=$PROJECT_DIR/data/TechQA
OUTPUT_DIR=$PROJECT_DIR/outputs
CHECKPOINT_DIR=$PROJECT_DIR/checkpoints

python techqa_vkg_qa.py \
    --init_checkpoint $CHECKPOINT_DIR/techqa/techqa_rel_encoder_bert_large_uncased_last_checkpoint \
    --per_device_pred_batch_size 32 \
    --pred_file $OUTPUT_DIR/techqa/qas/preprocessed_dev_whole_Q_A.jsonl \
    --query_file $OUTPUT_DIR/techqa/qas/query_vecs.jsonl \
    --max_length 300 
