import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from natsort import natsorted
from sklearn.manifold import TSNE

from graph2vec import feature_extractor, save_embedding
from utils import converter


class KGEmbedder:
    """
    A knowledge graph embedder using graph2vec
    """
    def __init__(self, graphs, names, json_dir, relable=True, hyperparameters=None):
        self.hyperparameters = hyperparameters
        if hyperparameters is None:
            self.hyperparameters = {'vector_size': 128,
                                    'alpha': 0.025,
                                    'min_alpha': 0.00025,
                                    'min_count': 1, 'dm': 0,
                                    'epochs': 1,
                                    'alpha_decay': 0.0002,
                                    'wl_rounds': 4}
        self.graphs = graphs
        self.embeddings = []
        self.model = None
        self.documents = None
        self.all_new_labels = None
        self.names = names
        self.compression_rates = dict()

        if relable:
            self.label_dict, self.label_dict_reverse, _ = converter.convert_to_json_edge_relabling(graphs, json_dir)
            print(self.label_dict, self.label_dict_reverse)
        self.json_graphs = natsorted(glob.glob(os.path.join(json_dir, '*.json')))
        print("json graphs:", self.json_graphs)
        self.set_doc_vec_params({
                                    param: val
                                    for param, val
                                    in self.hyperparameters.items()
                                    if param not in ['alpha_decay', 'wl_rounds']
                                    })

    def set_doc_vec_params(self, params):
        """set other doc_vec params"""
        self.max_epochs = params['epochs']
        del params['epochs']
        self.doc_vec_params = params

    def train_embeddings(self, output_dest=None):
        """Starts embedding training"""
        print("\nFeature extraction started.\n")

        all_new_labels = {}
        document_collections = []
        counter = 0
        for i, n in enumerate(self.names):
            doc, new_labels, compression_rate = feature_extractor(self.json_graphs[i],
                                                                  self.hyperparameters['wl_rounds'],
                                                                  fixed_name=self.names[i])
            self.compression_rates[self.names[i]] = compression_rate
            document_collections.append(doc)
            all_new_labels.update(new_labels)
            counter += 1

        print("\nOptimization started.\n")

        model = Doc2Vec(**self.doc_vec_params)

        model.build_vocab(document_collections)

        for epoch in range(self.max_epochs):
            print('iteration {0}'.format(epoch))
            model.train(document_collections,
                        total_examples=model.corpus_count,
                        epochs=50)
            # decrease the learning rate
            model.alpha -= self.hyperparameters['alpha_decay']
            # fix the learning rate, no decay
            model.min_alpha = model.alpha
        if not output_dest is None:
            save_embedding(output_dest, model, self.json_graphs, self.doc_vec_params['vector_size'])
        self.model = model
        self.documents = document_collections
        self.all_new_labels = all_new_labels
        return model, document_collections, all_new_labels

    def embed(self):
        """Computes and returns embeddings"""
        model, document_collections, _ = self.train_embeddings()
        results = []
        for i in range(len(model.docvecs)):
            results.append(model.docvecs[i])
        self.embeddings = np.array(results)
        return self.embeddings

    def show_embeddings(self, cutoff=sys.maxsize):
        if len(self.embeddings) > 0:
            tsne = TSNE(random_state=42, perplexity=30, n_components=2)
            X_tsne = tsne.fit_transform(self.embeddings)

            plt.figure(figsize=(5, 5))
            cutoff = min(len(self.embeddings), cutoff)
            plt.scatter(X_tsne[:cutoff, 0], X_tsne[:cutoff, 1], s=10, alpha=0.5)
            if cutoff <= len(self.embeddings):
                plt.scatter(X_tsne[cutoff:, 0], X_tsne[cutoff:, 1], color='red', s=10, alpha=0.5)
            plt.show()
