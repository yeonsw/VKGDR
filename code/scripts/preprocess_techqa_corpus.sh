DATA_DIR=$PROJECT_DIR/data
OUTPUT_DIR=$PROJECT_DIR/outputs

python preprocess_techqa_corpus.py \
    --building_predefined_entities_list \
    --get_stats \
    --preprocess_techqa \
    --split_train_valid \
    --techqa_corpus_file $DATA_DIR/TechQA/technote_corpus/corpus.jsonl \
    --techqa_corpus_preprocessed_file $OUTPUT_DIR/techqa/whole_corpus/whole_corpus_sentences_n_thresh.jsonl \
    --target_entity_pair_file $OUTPUT_DIR/techqa/whole_corpus/whole_target_entity_pairs_n_thresh.jsonl \
    --entity_file $OUTPUT_DIR/techqa/whole_corpus/whole_entities_n_thresh.jsonl \
    --eid2n_file $OUTPUT_DIR/techqa/whole_corpus/whole_eid2n_n_thresh.jsonl \
    --train_file $OUTPUT_DIR/techqa/whole_corpus/whole_train_n_thresh.jsonl \
    --valid_file $OUTPUT_DIR/techqa/whole_corpus/whole_valid_n_thresh.jsonl \
    --train_prop 0.98 \
    --use_title

