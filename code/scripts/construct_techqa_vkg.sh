DATA_DIR=$PROJECT_DIR/dataTechQA
OUTPUT_DIR=$PROJECT_DIR/outputs
CHECKPOINT_DIR=$PROJECT_DIR/checkpoints

python construct_vkg.py \
    --init_checkpoint $CHECKPOINT_DIR/techqa/techqa_rel_encoder_bert_large_uncased_last_checkpoint \
    --per_device_pred_batch_size 1024 \
    --vkg_path $OUTPUT_DIR/vkg/vkg \
    --corpus_file $OUTPUT_DIR/techqa/whole_corpus/dev_corpus_sentences_n_thresh.jsonl \
    --target_pair_file $OUTPUT_DIR/techqa/whole_corpus/dev_target_entity_pairs_n_thresh.jsonl \
    --max_length 128
