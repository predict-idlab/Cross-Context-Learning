import hashlib
import json
import logging

import networkx as nx
import pandas as pd
from gensim.models.doc2vec import TaggedDocument

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


class WeisfeilerLehmanMachine:
    """
    Weisfeiler Lehman feature extractor class.
    """

    def __init__(self, graph, features, iterations):
        """
        Initialization method which executes feature extraction.
        :param graph: The Nx graph object.
        :param features: Feature hash table.
        :param iterations: Number of WL iterations.
        """
        self.avg_nebs, self.total_nebs = 0, 0
        self.compression_rate = 0
        self.iterations = iterations
        self.graph = graph
        self.features = features
        self.nodes = sorted(list(self.graph.nodes()))
        self.new_labels = {}
        self.extracted_features = [str(v) for k, v in features.items()]
        self.do_recursions()

    def do_a_recursion(self):
        """
        Perform a single WL recursion.
        :return new_features: The hash table with extracted WL features.
        """
        new_features = {}
        for node in self.nodes:
            nebs = sorted(list(self.graph.neighbors(node)))

            self.avg_nebs += len(list(nebs))
            self.total_nebs += 1

            # print("neighbours for node", node, nebs)
            degs = [self.features[neb] for neb in nebs]

            # list_features = [str(self.features[node])] + degs
            list_features = [str(self.features[node])] + list(set(sorted([str(deg) for deg in degs])))

            features = "_".join(list_features)
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            new_features[node] = hashing
            self.new_labels[hashing] = list_features
            # print(hashing, list_features)
        self.extracted_features = self.extracted_features + list(new_features.values())
        return new_features

    def do_recursions(self):
        """
        Perform a series of WL recursions.
        """
        for iteration in range(self.iterations):
            self.features = self.do_a_recursion()
        self.compression_rate = 1
        if self.total_nebs > 0 and self.avg_nebs > 0:
            self.compression_rate = 1 / (self.avg_nebs / self.total_nebs)


def dataset_reader(path):
    """
    Read the graph and features from a json file.
    :param path: The path to the graph json.
    :return graph: The graph object.
    :return features: Features hash table.
    :return name: Name of the graph.
    """
    print("reading from", path)
    name = path.strip(".json").split("/")[-1]
    data = json.load(open(path))
    graph = nx.from_edgelist(data["edges"], create_using=nx.DiGraph())

    if "features" in data.keys():
        features = data["features"]
    else:
        features = nx.degree(graph)

    features = {int(k): v for k, v, in features.items()}
    return graph, features, name


def feature_extractor(path, rounds, fixed_name=''):
    """
    Extract WL features from a graph.
    :param path: The path to the graph json.
    :param rounds: Number of WL iterations.
    :return doc: Document collection object.
    """
    graph, features, name = dataset_reader(path)
    machine = WeisfeilerLehmanMachine(graph, features, rounds)
    if not fixed_name == '':
        name = fixed_name
    doc = TaggedDocument(words=machine.extracted_features, tags=[name])
    return doc, machine.new_labels, machine.compression_rate


def save_embedding(output_path, model, files, dimensions):
    """
    Save the embedding.
    :param output_path: Path to the embedding csv.
    :param model: The embedding model object.
    :param files: The list of files.
    :param dimensions: The embedding dimension parameter.
    """
    out = []
    for f in files:
        identifier = f.split("/")[-1].strip(".json")
        out.append([int(identifier)] + list(model.docvecs["g_" + identifier]))

    out = pd.DataFrame(out, columns=["type"] + ["x_" + str(dimension) for dimension in range(dimensions)])
    out = out.sort_values(["type"])
    out.to_csv(output_path, index=None)