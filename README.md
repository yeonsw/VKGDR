# Virtual Knowledge Graph Construction for Zero-Shot Domain-Specific Document Retrieval

Yeon Seonwoo, Seunghyun Yoon, Franck Dernoncourt, Trung Bui, Alice Oh | COLING 2022 | [Paper](https://aclanthology.org/2022.coling-1.101/)

KAIST, Adobe Research

Official implementation of "Virtual Knowledge Graph Construction for Zero-Shot Domain-Specific Document Retrieval"

# Abstract

Domain-specific documents cover terminologies and specialized knowledge. This has been the main challenge of domain-specific document retrieval systems. Previous approaches propose domain-adaptation and transfer learning methods to alleviate this problem. However, these approaches still follow the same document representation method in previous approaches; a document is embedded into a single vector. In this study, we propose VKGDR. VKGDR represents a given corpus into a graph of entities and their relations (known as a virtual knowledge graph) and computes the relevance between queries and documents based on the graph representation. We conduct three experiments 1) domain-specific document retrieval, 2) comparison of our virtual knowledge graph construction method with previous approaches, and 3) ablation study on each component of our virtual knowledge graph. From the results, we see that unsupervised VKGDR outperforms baselines in a zero-shot setting and even outperforms fully-supervised bi-encoder. We also verify that our virtual knowledge graph construction method results in better retrieval performance than previous approaches.

# Getting started

## Setup

1. Set the project directory

```
export PROJECT_DIR=/PATH/TO/THIS/PROJECT/FOLDER
```

1. Download Wikipedia and TechQA

```bash
wget https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-processed.tar.bz2
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-page_props.sql.gz
tar -xvf enwiki-20171001-pages-meta-current-withlinks-processed.tar.bz2
mkdir $PROJECT_DIR/data/wikipedia/enwiki_whole
mv enwiki-20171001-pages-meta-current-withlinks-processed/* $PROJECT_DIR/data/KGQA/wikipedia/enwiki_whole/
gzip -d enwiki-latest-page_props.sql.gz
mv enwiki-latest-page_props.sql $PROJECT_DIR/data/wikipedia/enwiki_props/
```

2. Download [TechQA](https://arxiv.org/abs/1911.02984)

```
mv TechQA.tar.bz2 $PROJECT_DIR/data
cd $PROJECT_DIR/data
tar -xvf TechQA.tar.bz2
cd TechQA/technote_corpus/
bzip2 -d full_technote_collection.txt.bz2
```

## Preprocess Wikipedia and TechQA

1. Preprocess Wikipedia

```
cd $PROJECT_DIR/code
bash scripts/generate_pretraining_data.sh
```

This step takes a while

2. Preprocess TechQA

```
cd $PROJECT_DIR/code
bash scripts/preprocess_techqa_full_technote_collection.sh
bash scripts/preprocess_techqa_corpus.sh
bash scripts/preprocess_techqa_train_dev_corpus.sh
```

3. Preprocess the TechQA train and dev set

```
bash scripts/extract_techqa_qa_pairs.sh
```

## Pre-train the relation encoder on the Wikipedia and TechQA datasets

```
bash scripts/pretrain_relation_encoder.sh
bash scripts/train_techqa_relation_encoder_same_doc.sh
```

## Construct a corpus-VKG

```
bash scripts/extract_dev_pmi_dev_corpus.sh
bash scripts/construct_techqa_vkg.sh
```

## Construct a query-VKG

```
bash scripts/vkg_techqa_qa.sh
```

## Compute BM25 similarity

1. Start Elasticsearch

```
mv $PROJECT_DIR/code
mkdir elasticsearch
mv elasticsearch
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.16.2-linux-x86_64.tar.gz -q
tar -xzf elasticsearch-7.16.2-linux-x86_64.tar.gz
cd elasticsearch-7.16.2/bin
bash elasticsearch
```

2. Keep Elasticsearch alive and run haystack to comptue BM25 similarity. 

```
bash script/haystack_bm25.sh
```

Please see more details of haystack [here](https://github.com/deepset-ai/haystack)

## Evalute the query vectors

```
bash eval_zeroshot_techqa.sh
```

