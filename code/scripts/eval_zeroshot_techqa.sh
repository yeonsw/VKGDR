DATA_DIR=$PROJECT_DIR/data/TechQA
OUTPUT_DIR=$PROJECT_DIR/outputs

python eval_zeroshot.py \
    --original_q_file $OUTPUT_DIR/techqa/qas/preprocessed_dev_whole_Q_A.jsonl \
    --bm25_file $OUTPUT_DIR/techqa/bm25_scores_dev.jsonl \
    --query_file $OUTPUT_DIR/techqa/qas/query_vecs.jsonl \
    --vkg_path $OUTPUT_DIR/vkg/vkg \
    --rel_lmb 1.0 \
    --bm25_lmb 1.0
