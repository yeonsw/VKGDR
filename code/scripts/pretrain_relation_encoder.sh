OUTPUT_DIR=$PROJECT_DIR/outputs/pretraining_data
CHECKPOINT_DIR=$PROJECT_DIR/checkpoints

python train_relation_encoder.py \
    --train_file $OUTPUT_DIR/pretraining_data_train.jsonl \
    --eval_file $OUTPUT_DIR/pretraining_data_valid.jsonl \
    --target_entity_pair_file $OUTPUT_DIR/target_entity_pairs.jsonl \
    --max_length 128 \
    --n_hard_negs 2 \
    --init_checkpoint bert-large-uncased \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --eval_steps 5000 \
    --n_epochs 3 \
    --learning_rate 2e-5 \
    --sampling_ratio 1.0 \
    --checkpoint_save_dir $CHECKPOINT_DIR/pretrained/rel_encoder_bert_large_uncased_pretrained
