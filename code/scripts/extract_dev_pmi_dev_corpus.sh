OUTPUT_DIR=$PROJECT_DIR/outputs

python extract_dev_pmi_dev_corpus.py \
    --target_pair_file $OUTPUT_DIR/techqa/whole_corpus/whole_target_entity_pairs_n_thresh.jsonl \
    --corpus_file $OUTPUT_DIR/techqa/whole_corpus/training_dev_corpus_sentences_n_thresh.jsonl \
    --dev_query_file $OUTPUT_DIR/techqa/qas/preprocessed_dev_whole_Q_A.jsonl \
    --dev_corpus_file $OUTPUT_DIR/techqa/whole_corpus/dev_corpus_sentences_n_thresh.jsonl \
    --dev_pmi_file $OUTPUT_DIR/techqa/whole_corpus/dev_target_entity_pairs_n_thresh.jsonl
