import jsonlines
from tqdm import tqdm

def read_bm25_score_file(inpf):
    data = {}
    with jsonlines.open(inpf, "r") as reader:
        for r in tqdm(reader, desc="Reading the BM25 file"):
            data[r["qid"]] = {}
            for _ in r["bm25_results"]:
                data[r["qid"]][_["doc_id"]] = _["bm25_score"]
    return data
