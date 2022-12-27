DATA_DIR=$WORKING_DIR/data/TechQA
OUTPUT_DIR=$WORKING_DIR/outputs
CHECKPOINT_DIR=$WORKING_DIR/checkpoints

python preprocess_techqa_corpus.py \
    --preprocess_techqa \
    --eid2n_file $OUTPUT_DIR/techqa/whole_corpus/whole_eid2n_n_thresh.jsonl \
    --techqa_corpus_file $DATA_DIR/training_and_dev/training_dev_technotes.json \
    --techqa_corpus_preprocessed_file $OUTPUT_DIR/techqa/whole_corpus/training_dev_corpus_sentences_n_thresh.jsonl \
    --entity_file $OUTPUT_DIR/techqa/whole_corpus/whole_entities_n_thresh.jsonl \
    --n_entity_thresh 0 \
    --use_title
