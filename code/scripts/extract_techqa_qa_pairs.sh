DATA_DIR=$PROJECT_DIR/data/TechQA
OUTPUT_DIR=$PROJECT_DIR/outputs

python extract_techqa_qa_pairs.py \
    --preprocess_qa_pairs \
    --techqa_qa_file $DATA_DIR/training_and_dev/training_Q_A.json\
    --entity_file $OUTPUT_DIR/techqa/whole_corpus/whole_entities_n_thresh.jsonl \
    --q_doc_file $OUTPUT_DIR/techqa/qas/preprocessed_training_whole_Q_A.jsonl

python extract_techqa_qa_pairs.py \
    --preprocess_qa_pairs \
    --techqa_qa_file $DATA_DIR/training_and_dev/dev_Q_A.json \
    --entity_file $OUTPUT_DIR/techqa/whole_corpus/whole_entities_n_thresh.jsonl \
    --q_doc_file $OUTPUT_DIR/techqa/qas/preprocessed_dev_whole_Q_A.jsonl
