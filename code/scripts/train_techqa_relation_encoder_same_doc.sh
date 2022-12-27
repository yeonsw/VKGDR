OUTPUT_DIR=$PROJECT_DIR/outputs
CHECKPOINT_DIR=$PROJECT_DIR/checkpoints

python train_relation_encoder_same_doc.py \
    --train_file $OUTPUT_DIR/techqa/whole_corpus/whole_train_n_thresh.jsonl \
    --eval_file $OUTPUT_DIR/techqa/whole_corpus/whole_valid_n_thresh.jsonl \
    --target_entity_pair_file $OUTPUT_DIR/techqa/whole_corpus/whole_target_entity_pairs_n_thresh.jsonl \
    --max_length 128 \
    --n_hard_negs 2 \
    --init_checkpoint $CHECKPOINT_DIR/pretrained/rel_encoder_bert_large_uncased_pretrained_last_checkpoint \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --eval_steps 10000 \
    --n_epochs 2 \
    --learning_rate 2e-5 \
    --sampling_ratio 1.0 \
    --checkpoint_save_dir $CHECKPOINT_DIR/techqa/techqa_rel_encoder_bert_large_uncased
