from omegaconf import OmegaConf
from typing import List
from copy import deepcopy

import omegaconf


class Node:
    def __init__(self, name, parents: List[str,], idx, id, level):
        self.name = name
        self.parents = parents
        self.idx = idx
        self.level = level
        self.id = id

    def get_name(self):
        return self.name

    def __str__(self):
        return self.name
    # def to_parent


def search_child_name(ontology, parent_list: List[Node,]):
    child = deepcopy(ontology)
    for node in parent_list:
        child = child[node.name]
    return list(child.keys()) if child is not None else None


def visit_childs(node, queue, node_set, ontology, name2id):
    level = node.level
    parents = node.parents + [node]
    child_names = search_child_name(ontology, parents)
    if child_names is None:
        return queue, node_set
    for child_name in child_names:
        last_idx = node_set[-1].idx
        newNode = Node(child_name, parents, last_idx+1, name2id[child_name], level + 1)
        node_set.append(newNode)
        queue.append(newNode)
    return queue, node_set


def search_all_nodes(meta_path: str):
    ontology = OmegaConf.load(f'{meta_path}/ontology2.yaml')
    id2name, name2id = new_audioset_label_indices(meta_path)

    queue = [Node(key, [], idx, name2id[key], 0) for idx, key in enumerate(ontology.keys())]
    node_set = [Node(key, [], idx, name2id[key], 0) for idx, key in enumerate(ontology.keys())]
    while queue:
        node = queue.pop(0)
        queue, node_set = visit_childs(node, queue, node_set, ontology, name2id)
    return node_set


def new_audioset_label_indices(meta_root: str):
    id2name = {}
    name2id = {}
    with open(f'{meta_root}/new_class_labels_indices.csv', 'r', encoding='utf8') as f:
        for line_num, line in enumerate(f.readlines()):
            if line_num == 0:
                continue
            splited_line = line.strip().split(',')
            idx, id, name = splited_line[0], splited_line[1], ','.join(splited_line[2:]).replace('\"', '')
            idx, id, name = int(idx.strip()), id.strip(), name.strip()
            id2name[id] = name
            name2id[name] = id
    return id2name, name2id


def get_ontology_names(ontology: omegaconf.dictconfig.DictConfig):
    key_queue = [[key] for key in ontology.keys()]
    node_set = [key for key in ontology.keys()]

    while key_queue:
        parent_names = key_queue.pop(0)
        childs = search_child_name(ontology, parent_names)
        if childs is None:
            continue
        for child in childs:
            node_set.append(child)
            key_queue.append(parent_names + [child])
    return node_set


def mk_new_indices(meta_root):
    ontology = OmegaConf.load(f'{meta_root}/ontology2.yaml')
    node_names = get_ontology_names(ontology)
    name2idx_n_id = {}
    with open(f'{meta_root}/class_labels_indices.csv', 'r', encoding='utf8') as f:
        for line_num, line in enumerate(f.readlines()):
            if line_num == 0:
                continue
            splited_line = line.strip().split(',')
            idx, id, name = splited_line[0], splited_line[1], ','.join(splited_line[2:]).replace('\"', '')
            idx, id, name = int(idx.strip()), id.strip(), name.strip()
            name2idx_n_id[name] = (idx, id)

    names = list(name2idx_n_id.keys())
    idx = len(name2idx_n_id)

    for name in node_names:
        if name not in names:
            name2idx_n_id[name] = (idx, '')
            idx += 1
    with open(f'{meta_root}/new_class_labels_indices.csv', 'w', encoding='utf8') as f:
        f.write('index,mid,display_name\n')
        for name in name2idx_n_id.keys():
            idx, id = name2idx_n_id[name]
            f.write('{},{},"{}"\n'.format(idx, id, name))