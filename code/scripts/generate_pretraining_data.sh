DATA_DIR=$PROJECT_DIR/data/wikipedia
OUTPUT_DIR=$PROJECT_DIR/outputs/pretraining_data

python generate_pretraining_data.py \
    --preprocess_wikipedia \
    --get_wikiid2wikidata \
    --get_entities \
    --build_pretraining_data \
    --filter_sentences_and_get_target_pairs \
    --split_train_valid \
    --wikipedia_dump_file $DATA_DIR/enwiki_whole_toy \
    --wikidata_dump_file $DATA_DIR/enwiki_props/enwiki-latest-page_props.sql \
    --wiki_prep_file $OUTPUT_DIR/enwiki_whole_preprocessed.jsonl \
    --wikiid2wikidataid_file $OUTPUT_DIR/wikiid2wikidataid.json \
    --wiki_entity_file $OUTPUT_DIR/wiki_entities.jsonl \
    --wikipedia_sentences_file $OUTPUT_DIR/pretraining_data.jsonl \
    --target_entity_pair_file $OUTPUT_DIR/target_entity_pairs.jsonl \
    --pretraining_data_train_file $OUTPUT_DIR/pretraining_data_train.jsonl \
    --pretraining_data_valid_file $OUTPUT_DIR/pretraining_data_valid.jsonl \
    --n_filter 5 \
    --entity_pair_n_thresh 800000 \
    --train_prop 0.97

