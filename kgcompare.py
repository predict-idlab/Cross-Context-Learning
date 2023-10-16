import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial


class KGCompare:
    """Class for Knowledge Graph comparison"""

    def __init__(self, embedder):
        """
        Initialisation
        :param embedder:
        """
        self.embedder = embedder

    def get_dist_matrix(self, distance_metric=spatial.distance.cosine):
        """
        Compute distances matrix based on the embedding in the KGEmbedder
        :param distance_metric:
        :return:
        """
        try:
            return self.dist_matrix
        except AttributeError:
            all_distances = []
            for embedi in self.embedder.embeddings:
                distances = []
                for embedj in self.embedder.embeddings:
                    distances.append(distance_metric(embedi, embedj))
                all_distances.append(np.array(distances))
            self.dist_matrix = np.array(all_distances)
            return self.dist_matrix

    def get_generic_similarity(self, graph_id, similarity_measure):
        """
        Get list with most similar graph ids, based on generic similarity measure, either min/max.
        :param graph_id:
        :param similarity_measure:
        :return:
        """
        matrix = self.get_dist_matrix()
        min_dist = similarity_measure(np.concatenate([matrix[graph_id][:graph_id], matrix[graph_id][graph_id + 1:]]))
        return [i for i, j in enumerate(matrix[graph_id]) if j == min_dist]

    def get_most_similar(self, graph_id):
        """
        Get list with most similar graph ids.
        :param graph_id:
        :return:
        """
        return self.get_generic_similarity(graph_id, min)

    def get_least_similar(self, graph_id):
        """
        Get list with least similar graph ids.
        :param graph_id:
        :return:
        """
        return self.get_generic_similarity(graph_id, max)

    def get_graph_overlap(self):
        """
        Get overlap between graphs stored by the embedder.
        :return:
        """
        try:
            return self.graph_overlap
        except AttributeError:
            self.graph_overlap = np.array(
                [np.array([po_match(G1, G2) / len(G1) for G1 in self.embedder.graphs]) for G2 in self.embedder.graphs])
            return self.graph_overlap


def po_match(G1, G2):
    counter = 0
    for s, p, o in G1:
        if (None, p, o) in G2:
            counter += 1
    return counter
