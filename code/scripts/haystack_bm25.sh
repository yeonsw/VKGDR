DATA_DIR=$PROJECT_DIR/data/TechQA
OUTPUT_DIR=$PROJECT_DIR/outputs

python haystack_bm25.py \
    --doc_inpf $DATA_DIR/technote_corpus/corpus.jsonl \
    --q_inpf $DATA_DIR/training_and_dev/dev_Q_A.json \
    --outf $OUTPUT_DIR/techqa/bm25_scores_dev.jsonl
