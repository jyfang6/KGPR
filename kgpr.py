import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier.model import add_ranks
from pyterrier.measures import *

import os
import json
import pickle
import argparse
import numpy as np
from tqdm import trange

import torch
import transformers
from transformers import LukeTokenizer
transformers.logging.set_verbosity_error()

from model import RelationLukeCrossEncoder
from get_freebase_subgraph import get_query_passage_subgraph


def load_json_file(path):

    outputs = None
    if path.endswith(".json"):
        outputs = json.loads(open(path, "r", encoding="utf-8").read())
    elif path.endswith(".jsonl"):
        outputs = []
        with open(path, "r", encoding="utf-8") as fin:
            for line in fin:
                outputs.append(json.loads(line))

    return outputs

def get_id2entity(path):

    elresult = load_json_file(path)
    for item in elresult:
        item["id"] = str(item["id"])
    id2entity = dict()
    for item in elresult:
        id2entity[item["id"]] = item
    return id2entity

class KGPR(pt.Transformer):

    def __init__(self, checkpoint, id2query_entity, id2passage_entity, num_rels, rel_dim, rel_embed, rel2id, subgraph_path, batch_size=4):

        super().__init__()

        self.id2query_entity = id2query_entity
        self.id2passage_entity = id2passage_entity
        self.num_rels = num_rels
        self.rel_dim = rel_dim
        self.rel2id = rel2id
        self.subgraph_path = subgraph_path
    
        self.batch_size = batch_size
        self.device = torch.device("cuda:0")
        self.max_hop = 1
        self.tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large")
        self.model = RelationLukeCrossEncoder(
            {
                "model_name": "studio-ousia/luke-large",
                "use_cls": True,
                "num_rels": num_rels,
                "rel_dim": rel_dim,
                "rel_embed": rel_embed,
                "rel_only": False
            }
        )

        print("loading checkpoint from : {} ...".format(checkpoint))
        checkpoint = torch.load(checkpoint, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"], strict=False)

        self.model.to(self.device)
        self.model.eval()

    def transform(self, rank):

        total = len(rank)
        num_steps = (total-1) // self.batch_size + 1
        scores = []
        
        for i in trange(num_steps):
            batch_rank = rank.iloc[i*self.batch_size: (i+1)*self.batch_size]
            batch_scores = self.batch_score(batch_rank)
            scores.extend(batch_scores)

        rank = rank.drop(columns=['score', 'rank'], errors='ignore').assign(score=scores)
        rank = add_ranks(rank)
        rank.sort_values(["qid", "rank"], ascending=[True, True], inplace=True)

        return rank

    def batch_score(self, rank):

        features = self.get_input_feature(rank=rank)
        logits = self.model({k: v.to(self.device) for k, v in features.items()})
        scores = torch.nn.functional.log_softmax(logits, dim=-1)[:, -1].detach().cpu().numpy()
        scores = list(scores)
        return scores

    def get_input_feature(self, rank):

        query_list = rank["query"].to_list()
        passage_list = rank["text"].to_list()
        query_span_list, query_entity_list, passage_span_list, \
            passage_entity_list = self.get_text_span(rank)

        features = self.tokenizer(
            text = query_list,
            text_pair = passage_list,
            entity_spans = query_span_list,
            entity_spans_pair = passage_span_list,
            entities = query_entity_list,
            entities_pair = passage_entity_list,
            padding = True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        # get subgraph
        query_passage_pair = []
        for qid, docno in zip(rank["qid"], rank["docno"]):
            query_passage_pair.append((qid, docno))
        subgraph = get_query_passage_subgraph(self.subgraph_path, query_passage_pair, self.id2query_entity, self.id2passage_entity)
        
        batch_size = features["input_ids"].shape[0]
        max_entity_len = features["entity_position_ids"].shape[-1]
        if subgraph is not None:
            batch_path_list = []
            batch_path_position_list = []
            for i, (qid, pid) in enumerate(zip(rank["qid"].to_list(), rank["docno"].to_list())):
                if qid not in subgraph or pid not in subgraph[qid]:
                    batch_path_list.append([])
                    batch_path_position_list.append([])
                    continue

                path_list = []
                path_position_list = []
                query_passage_paths = subgraph[qid][pid]
                for path in query_passage_paths:
                    query_entity = self.id2query_entity[qid]
                    try:
                        head_entity_index= query_entity["entity"].index(path[0])
                    except:
                        print(f"{path[0]} not in entity linking results of qid: {qid}")
                        continue
                    
                    if head_entity_index >= features["entity_ids"].shape[1]:
                        continue
                    head_entity_id = features["entity_ids"][i, head_entity_index].item()
                
                    passage_entity = self.id2passage_entity[pid]
                    try:
                        tail_entity_index = passage_entity["entity"].index(path[-1]) + len(query_entity["entity"])
                    except:
                        print(f"{path[-1]} not in entity linking results of pid: {pid}")
                        continue
                    
                    if tail_entity_index >= features["entity_ids"].shape[1]:
                        continue
                    tail_entity_id = features["entity_ids"][i, tail_entity_index].item()
                    
                    rels = []
                    try:
                        for r in path[1: -1]:
                            rels.append(self.rel2id[r])
                    except:
                        continue

                    rels = rels + [len(self.rel2id)] * (self.max_hop-len(rels))
                    path = [head_entity_id, tail_entity_id] + rels # path format: [h, t, r1, padding]

                    positions = []
                    for pos_id in features["entity_position_ids"][i, head_entity_index]:
                        if pos_id < 0:
                            break
                        positions.append(pos_id.item())
                    for pos_id in features["entity_position_ids"][i, tail_entity_index]:
                        if pos_id < 0:
                            break
                        positions.append(pos_id.item())

                    path_list.append(path)
                    path_position_list.append(positions)

                batch_path_list.append(path_list)
                batch_path_position_list.append(path_position_list)

            max_num_paths = 0
            for path_list in batch_path_list:
                max_num_paths = max(max_num_paths, len(path_list))
            max_num_paths = min(32, max(1, max_num_paths)) # 控制max_num_paths的数量在1--12之间
            path_ids = torch.zeros((batch_size, max_num_paths, 2+self.max_hop), dtype=torch.long).fill_(len(self.rel2id))
            path_position_ids = torch.zeros((batch_size, max_num_paths, max_entity_len), dtype=torch.long).fill_(-1)
            path_attention_mask = torch.zeros((batch_size, max_num_paths), dtype=torch.bool)

            for i in range(batch_size):
                for j, path in enumerate(batch_path_list[i]):
                    if j >= max_num_paths:
                        break
                    path_ids[i, j] = torch.tensor(path, dtype=torch.long)
                for j, path_pos_id in enumerate(batch_path_position_list[i]):
                    if j >= max_num_paths:
                        break
                    max_path_len = min(max_entity_len, len(path_pos_id))
                    path_position_ids[i, j, :max_path_len] = torch.tensor(path_pos_id[:max_path_len], dtype=torch.long)
                path_attention_mask[i, :len(batch_path_list[i])] = True

            features["path_ids"] = path_ids
            features["path_position_ids"] = path_position_ids
            features["path_attention_mask"] = path_attention_mask

        return features

    def get_text_span(self, rank):

        query_span_list, query_entity_list = [], []
        if self.id2passage_entity is not None:
            passage_span_list, passage_entity_list = [], []
        else:
            passage_span_list, passage_entity_list = None, None

        for i in range(len(rank)):

            item = rank.iloc[i]
            query_id = item["qid"]
            query = item["query"]
            passage_id = item["docno"]
            passage = item["text"]

            span_entities = self.get_span_entities(query, self.id2query_entity[query_id])
            query_span_list.append([span for span, entity in span_entities])
            query_entity_list.append([entity for span, entity in span_entities])
            if self.id2passage_entity is not None:
                span_entities = self.get_span_entities(passage, self.id2passage_entity[passage_id])
                passage_span_list.append([span for span, entity in span_entities])
                passage_entity_list.append([entity for span, entity in span_entities])
        
        outputs = (query_span_list, query_entity_list, passage_span_list, passage_entity_list)
        
        return outputs
         
    def get_span_entities(self, text, entity_item):

        lower_text = text.lower()
        lower = False
        if len(lower_text) == len(text):
            text = lower_text
            lower = True

        span_entities = []
        for mention, entity in zip(entity_item["mention"], entity_item["entity_name"]):
            if lower:
                start_idx = text.find(mention.lower())
            else:
                start_idx = text.find(mention)

            if start_idx < 0 or start_idx + len(mention) > len(text):
                continue
            span_entities.append(((start_idx, start_idx + len(mention)), entity))

        return span_entities
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str, default="data/index", help="the folder of the pyterrier index files of msmarco")
    parser.add_argument("--query_entity_linking_file", type=str, default="data/entity_linking_results/query.jsonl", help="the path to the query entity linking files")
    parser.add_argument("--passage_entity_linking_file", type=str, default="data/entity_linking_results/passage.jsonl", help="the path to the passage entity linking files")
    parser.add_argument("--freebase", type=str, default="data/freebase", help="the folder of the freebase data")
    parser.add_argument("--checkpoint", type=str, default="data/checkpoint/kgpr.pt", help="checkpoint file")
    parser.add_argument("--eval_dataset", type=str, default="dl-hard", help="evaluation dataset: dl-hard, dl-2019, dl-2020")
    args = parser.parse_args()

    index = pt.IndexFactory.of(args.index)
    bm25 = pt.BatchRetrieve(index, wmodel="BM25")

    print("Loading entity linking results ... ")
    
    id2query_entity = get_id2entity(args.query_entity_linking_file)
    id2passage_entity = get_id2entity(args.passage_entity_linking_file)

    print("Loading relation embeddings...")
    with open(os.path.join(args.freebase, "rel2id.pickle"), "rb") as fin:
        rel2id = pickle.load(fin)
    relation_embed_path = os.path.join(args.freebase, "relation_embed_300d.npy")
    if relation_embed_path is not None:
        rel_embed = np.load(relation_embed_path)
        assert len(rel2id) == rel_embed.shape[0]
        rel_dim = rel_embed.shape[1]
    num_rels = len(rel2id)
    rel_dim = rel_dim
    rel_embed = rel_embed

    kgpr_scorer = KGPR(args.checkpoint, id2query_entity, id2passage_entity, num_rels, rel_dim, rel_embed, rel2id, args.freebase, batch_size=8)

    DATASET_MAP = {
        "dl-hard": "irds:msmarco-passage/trec-dl-hard",
        "dl-2019": "irds:msmarco-passage/trec-dl-2019/judged",
        "dl-2020": "irds:msmarco-passage/trec-dl-2020/judged"
    }

    dataset = pt.get_dataset(DATASET_MAP[args.eval_dataset])

    result = pt.Experiment(
        [
            bm25,
            bm25 >> pt.text.get_text(dataset, "text") >> kgpr_scorer,
        ],
        dataset.get_topics(),
        dataset.get_qrels(),
        eval_metrics=[nDCG@10],
        names=["BM25", "BM25 >> KGPR"]
    )

    print("Evaluation Results: ")
    print(result)

    
