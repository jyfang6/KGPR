
import os
from collections import defaultdict
from KG_api import KnowledgeGraph


kg = None

def get_query_passage_subgraph(output_dir, query_passage_pair, id2query_entities, id2passage_entities):

    triples_path = os.path.join(output_dir, "subgraph_1hop_triples.npy")
    ent_type_path = os.path.join(output_dir, "ent_type_ary.npy")
    ent2id_path = os.path.join(output_dir, "ent2id.pickle")
    rel2id_path = os.path.join(output_dir, "rel2id.pickle")
    
    global kg 
    if kg is None:
        print("Loading knowledge graph from {} ...".format(output_dir))
        kg = KnowledgeGraph("test", sparse_kg_path=(triples_path, ent_type_path), \
                            ent2id_path=ent2id_path, rel2id_path=rel2id_path)
    
    subgraph = {}
    for qid, docno in query_passage_pair:
        if qid not in subgraph:
            subgraph[qid] = defaultdict(list)
        for query_entity in id2query_entities[qid]["entity"]:
            for passage_entity in id2passage_entities[docno]["entity"]:
                path_list = kg.get_paths("sparse", query_entity, passage_entity, 1, None)
                if len(path_list) > 0:
                    for path in path_list:
                        subgraph[qid][docno].append([query_entity] + path + [passage_entity])
    
    return subgraph