import json
import os
import rdflib
from collections import defaultdict

from geopy import distance
from tqdm import tqdm

from incremental import *
from kg2vec import KGEmbedder
from kgcompare import KGCompare


def get_finished_nodes(directory, fold):
    """
    Get the nodes that are already associated with transfer results.
    :param directory: directory to go look for nodes
    :param fold: fold inside the directory
    :return:
    """
    worst_nodes, best_nodes, double_nodes = set(), set(), set()
    node_to_files = defaultdict(set)
    node_dir = os.path.join(directory, fold)
    for filename in os.listdir(node_dir):
        els = filename.split("_")
        if "best" in filename:
            best_node = els[-3] + "_" + els[-2]
            node_to_files[best_node].add(filename)
            if best_node in best_nodes:
                double_nodes.add(best_node)
            best_nodes.add(best_node)
        elif "worst" in filename:
            worst_node = els[-3] + "_" + els[-2]
            node_to_files[worst_node].add(filename)
            if worst_node in worst_nodes:
                double_nodes.add(worst_node)
            worst_nodes.add(worst_node)

    nodes = best_nodes.intersection(worst_nodes)
    print("incomplete nodes:", best_nodes.symmetric_difference(worst_nodes))
    print("double nodes:", double_nodes)
    for node in best_nodes.symmetric_difference(worst_nodes):
        for filename in node_to_files[node]:
            print("Remove incomplete node", filename)
            os.remove(os.path.join(node_dir, filename))
    for node in double_nodes:
        for filename in node_to_files[node]:
            print("Remove double node", filename)
            os.remove(os.path.join(node_dir, filename))
    return nodes


def get_geo_bounded_nodes(names, coordinates):
    """
    Get a set of nodes that are non-overlapping within a given radius.
    :param names: node names
    :param coordinates: node coordinates
    :return:
    """
    if os.path.exists("geobounded.json"):
        with open("geobounded.json", "r") as jsonfile:
            return json.load(jsonfile)
    geonames = [name for name in names if name in coordinates.keys()]
    # use radius of brussels to isolate nodes
    # successively add nodes and see if their radii intersect
    # according to wikipedia, Brussels has a surface area of 162.4km^2
    # for simplicity's sake, we assume this is a circular area
    # we derive the radius as r = root(A / pi)
    r = math.sqrt(162.4 / math.pi)
    chosen_so_far = list()
    for name in geonames:
        intersection = False
        for cname in chosen_so_far:
            ref_lat, ref_lon = coordinates[name]
            lat, lon = coordinates[cname]
            dist = distance.distance((ref_lat, ref_lon), (lat, lon)).km
            intersection = dist <= r
        if not intersection:
            chosen_so_far.append(name)
    print("Found", len(chosen_so_far), "non-intersecting nodes... among", len(geonames), "and", len(names), "original geonodes/nodes")
    with open("geobounded.json", "w") as jsonfile:
        json.dump(chosen_so_far, jsonfile)
    return chosen_so_far


def run_random_experiment(names, coordinates, incremental=False, geo_bounded=False, nr_of_folds=1):
    """
    Run a cross-context experiment without a selection criterion.
    :param names: node names
    :param coordinates: node coordinates
    :param incremental: not supported
    :param geo_bounded: whether we use geo-bounded nodes or not
    :param nr_of_folds: number of folds we want to use
    :return:
    """
    if geo_bounded:
        names = get_geo_bounded_nodes(names, coordinates)
    N = len(names) # population size
    for i in range(nr_of_folds):
        selection = random.sample(names, N)
        indices = [names.index(s) for s in selection]
        nodes = get_finished_nodes("random", "fold_" + str(i + 1))
        for j, name in enumerate(selection):
            print("\n\n### START NEW RANDOM COMPARISON ###\n\n")
            if name in nodes:
                print("skipping", name)
                continue

            rd_idx = names.index(name)
            while rd_idx == names.index(name):
                rd_idx = random.sample(indices, 1)[0]

            to_node = name
            from_node = names[rd_idx]
            print("Transferring from random context", from_node, "to", to_node)
            anoms, aucs_close, aucs_close_dest = transfer_from_to(from_node, to_node, best=True, incremental=incremental, repo="random/fold_" + str(i + 1) + "/")


def run_semantic_distance_experiment(names, files, comparator, coordinates, incremental=False, geo_bounded=False, nr_of_folds=1):
    """
    Run a cross-context experiment using a selection criterion based on the distance between context graphs.
    :param names: node names
    :param files: names of files containing rdf context graphs
    :param comparator: instance of KGCompare
    :param coordinates: node coordinates
    :param incremental: not supported
    :param geo_bounded: whether we use geo-bounded nodes or not
    :param nr_of_folds: number of folds we want to use
    :return:
    """

    if geo_bounded:
        new_names = get_geo_bounded_nodes(names, coordinates)
        indices = [i for i, name in enumerate(names) if name in new_names]
        new_files = [files[i] for i in indices]

        new_graphs = []
        for f in new_files:
            if os.path.isfile(f):
                graph = rdflib.Graph()
                ending = f.split('.')[-1]
                if ending == "ttl":
                    ending = "n3"
                graph.parse(f, format=ending)
                new_graphs.append(graph)

        json_dir = os.path.join("graphs", "json_relabeled")

        embedder = KGEmbedder(new_graphs, new_names, json_dir)
        embedder.embed()

        names = new_names
        files = new_files
        comparator = KGCompare(embedder)

    N = len(names) # population size
    for i in range(nr_of_folds):
        selection = random.sample(names, N)
        indices = [names.index(s) for s in selection]
        nodes = get_finished_nodes("semantic_distance", "fold_" + str(i + 1))
        for j, name in enumerate(selection):
            print("\n\n### START NEW SEMANTIC COMPARISON ###\n\n")
            if name in nodes:
                print("skipping", name)
                continue
            best_idx = comparator.get_most_similar(indices[j])[0]
            worst_idx = comparator.get_least_similar(indices[j])[0]

            to_node = name
            best_from_node = names[best_idx]
            worst_from_node = names[worst_idx]

            print("Transferring from nearest context", best_from_node, "to", to_node)
            anoms, aucs_close, aucs_close_dest = transfer_from_to(best_from_node, to_node, best=True, incremental=incremental, repo="semantic_distance/fold_" + str(i + 1) + "/")

            print("Transferring from furthest context", worst_from_node, "to", to_node)
            anoms, aucs_far, aucs_far_dest = transfer_from_to(worst_from_node, to_node, best=False, incremental=incremental, repo="semantic_distance/fold_" + str(i + 1) + "/")


def run_ts_distance_experiment(names, coordinates, raw_data_location, incremental=False, geo_bounded=False, nr_of_folds=1):
    """
    Run a cross-context experiment using a selection criterion based on the distance between time series.
    :param names: node names
    :param coordinates: node coordinates
    :param incremental: not supported
    :param geo_bounded: whether we use geo-bounded nodes or not
    :param nr_of_folds: number of folds we want to use
    :return:
    """

    if geo_bounded:
        names = get_geo_bounded_nodes(names, coordinates)

    def get_ts_distances(name):
        distances = np.zeros((len(names)))
        for i, other_name in tqdm(enumerate(names)):
            if name == other_name:
                distances[i] = -1
            else:
                locale1, node1 = name.split("_")[0], name.split("_")[1]
                locale2, node2 = other_name.split("_")[0], other_name.split("_")[1]
                series1 = FeatureExtractor(data_dir=raw_data_location, locale=locale1, nodes=[node1]).nodes[node1].fillna(0).drop(columns=['time']).values
                series2 = FeatureExtractor(data_dir=raw_data_location, locale=locale2, nodes=[node2]).nodes[node2].fillna(0).drop(columns=['time']).values

                # select the smallest first 40% (portion of the training set)
                # ind1 = (series1.shape[0] * 0.6 * 0.8) * 0.4  # take into account test and validation sets
                # ind2 = (series2.shape[0] * 0.6 * 0.8) * 0.4  # take into account test and validation sets
                # select the whole training set
                ind1 = (series1.shape[0] * 0.6 * 0.8)
                ind2 = (series2.shape[0] * 0.6 * 0.8)
                min_ind = int(min(ind1, ind2))
                series1 = series1[:min_ind]
                series2 = series2[:min_ind]

                def euclidean(list1, list2):
                    sum_of = 0
                    for x, y in zip(list1, list2):
                        ans = (x - y)**2
                        sum_of += ans
                    return sum_of**(1/2)

                distances[i] = euclidean(series1.reshape(-1, 1).squeeze(), series2.reshape(-1, 1).squeeze())
        # make sure that that the distance to self can never be selected as minimum or maximum
        distances[names.index(name)] = np.mean(distances)
        return distances

    N = len(names) # population size
    for i in range(nr_of_folds):
        selection = random.sample(names, N)
        indices = [names.index(s) for s in selection]
        nodes = get_finished_nodes("ts_distance", "fold_" + str(i + 1))
        for j, name in enumerate(selection):
            print("\n\n### START NEW TS COMPARISON ###\n\n")
            if name in nodes:
                print("skipping", name)
                continue
            ts_distances = get_ts_distances(name)
            best_idx = np.argmin(ts_distances)
            worst_idx = np.argmax(ts_distances)

            to_node = name
            best_from_node = names[best_idx]
            worst_from_node = names[worst_idx]

            print("Transferring from nearest context", best_from_node, "to", to_node)
            anoms, mae_close, mae_close_dest = transfer_from_to(best_from_node, to_node, best=True, incremental=incremental, repo="ts_distance/fold_" + str(i + 1) + "/")

            print("Transferring from furthest context", worst_from_node, "to", to_node)
            anoms, mae_far, mae_far_dest = transfer_from_to(worst_from_node, to_node, best=False, incremental=incremental, repo="ts_distance/fold_" + str(i + 1) + "/")


def run_geo_distance_experiment(names, coordinates, incremental=False, density=1.0, radius_based=False, nr_of_folds=1):
    """
    Run a cross-context experiment using a selection criterion based on the distance between geo-locations.
    :param names: node names
    :param coordinates: node coordinates
    :param incremental: not supported
    :param density: the percentage of nodes we wish to retain
    :param radius_based: whether or not we wish to make sure the nodes are geo-bounded
    :param nr_of_folds: number of folds we want to use
    :return:
    """

    percentage = str(int(density * 100))

    subnames = [name for name in names if name in coordinates.keys()]
    if not radius_based:
        subnames = random.sample(subnames, int(density * len(subnames)))
    else:
        subnames = get_geo_bounded_nodes(names, coordinates)

    def get_geo_distances(name):
        ref_lat, ref_lon = coordinates[name]
        distances = np.zeros((len(subnames)))
        for i, other_name in tqdm(enumerate(subnames)):
            if name == other_name:
                distances[i] = -1
            else:
                lat, lon = coordinates[other_name]
                distances[i] = distance.distance((ref_lat, ref_lon), (lat, lon)).meters
        # make sure that that the distance to self can never be selected as minimum or maximum
        distances[subnames.index(name)] = np.mean(distances)
        return distances

    N = len(subnames) # population size
    for i in range(nr_of_folds):
        # we sample from coordinates to ensure that each name can be localized
        selection = random.sample(subnames, N)
        indices = [subnames.index(s) for s in selection]
        nodes = get_finished_nodes("geo_distance_" + percentage, "fold_" + str(i + 1))
        for j, name in enumerate(selection):
            print("\n\n### START NEW GEO COMPARISON ###\n\n")
            if name in nodes:
                print("skipping", name)
                continue
            geo_distances = get_geo_distances(name)
            best_idx = np.argmin(geo_distances)
            worst_idx = np.argmax(geo_distances)

            to_node = name
            best_from_node = subnames[best_idx]
            worst_from_node = subnames[worst_idx]

            print("Transferring from nearest context", best_from_node, "to", to_node)
            anoms, mae_close, mae_close_dest = transfer_from_to(best_from_node, to_node, best=True, incremental=incremental, repo="geo_distance_" + percentage + "/fold_" + str(i + 1) + "/")

            print("Transferring from furthest context", worst_from_node, "to", to_node)
            anoms, mae_far, mae_far_dest = transfer_from_to(worst_from_node, to_node, best=False, incremental=incremental, repo="geo_distance_" + percentage + "/fold_" + str(i + 1) + "/")


def summary(scores=None, incremental=False, repo=""):
    """
    Get a summary of scores.
    :param scores: matrix of scores
    :param incremental: not supported
    :param repo: where the score may be located if not matrix is provided
    :return:
    """
    node_map = defaultdict(list)
    if scores is not None:
        # in case of an array of multiple arrays
        print("### MEAN OF SCORES ###")
        print(np.mean(scores, axis=0))
        print("### STD OF SCORES ###")
        print(np.std(scores, axis=0))
        return np.mean(scores, axis=0), np.std(scores, axis=0)
    else:
        best_scores = np.zeros((3, 3))
        worst_scores = np.zeros((3, 3))
        if incremental:
            best_scores = np.zeros((5, 3))
            worst_scores = np.zeros((5, 3))

        best_roc_aucs1, best_roc_aucs2, best_roc_aucs3 = list(), list(), list()
        worst_roc_aucs1, worst_roc_aucs2, worst_roc_aucs3 = list(), list(), list()
        best_avg_prec1, best_avg_prec2, best_avg_prec3 = list(), list(), list()
        worst_avg_prec1, worst_avg_prec2, worst_avg_prec3 = list(), list(), list()

        nr_best_files = 0
        nr_worst_files = 0
        for filename in os.listdir(repo):
            if "csv" in filename:
                if "best" in filename:
                    index = 0
                    with open(os.path.join(repo, filename), 'r') as csvfile:
                        r = csv.reader(csvfile, delimiter=',')
                        scs = list()
                        next(r)
                        for row in r:
                            if not np.isnan(float(row[1])) and \
                                    not np.isnan(float(row[2])) and \
                                    not np.isnan(float(row[3])):
                                scs.append((float(row[1]), float(row[2]), float(row[3])))
                                if index == 0:
                                    best_roc_aucs1.append(float(row[1]))
                                    best_avg_prec1.append(float(row[2]))
                                if index == 1:
                                    best_roc_aucs2.append(float(row[1]))
                                    best_avg_prec2.append(float(row[2]))
                                if index == 2:
                                    best_roc_aucs3.append(float(row[1]))
                                    best_avg_prec3.append(float(row[2]))
                                index += 1
                        index = 0
                        name = filename.split("_")
                        if len(scs) == 3:
                            for sc in scs:
                                best_scores[index, 0] += sc[0]
                                best_scores[index, 1] += sc[1]
                                best_scores[index, 2] += sc[2]
                                index += 1
                            node_map[name[-3] + "_" + name[-2]].append(("BEST", scs[0][0], scs[1][0], scs[2][0]))
                            nr_best_files += 1
                else:
                    index = 0
                    with open(os.path.join(repo, filename), 'r') as csvfile:
                        r = csv.reader(csvfile, delimiter=',')
                        scs = list()
                        next(r)
                        for row in r:
                            if not np.isnan(float(row[1])) and \
                                    not np.isnan(float(row[2])) and \
                                    not np.isnan(float(row[3])):
                                scs.append((float(row[1]), float(row[2]), float(row[3])))
                                if index == 0:
                                    worst_roc_aucs1.append(float(row[1]))
                                    worst_avg_prec1.append(float(row[2]))
                                if index == 1:
                                    worst_roc_aucs2.append(float(row[1]))
                                    worst_avg_prec2.append(float(row[2]))
                                if index == 2:
                                    worst_roc_aucs3.append(float(row[1]))
                                    worst_avg_prec3.append(float(row[2]))
                                index += 1
                        index = 0
                        name = filename.split("_")
                        if len(scs) == 3:
                            for sc in scs:
                                worst_scores[index, 0] += sc[0]
                                worst_scores[index, 1] += sc[1]
                                worst_scores[index, 2] += sc[2]
                                index += 1
                            node_map[name[-3] + "_" + name[-2]].append(("WORST", scs[0][0], scs[1][0], scs[2][0]))
                            nr_worst_files += 1

        best_scores /= nr_best_files
        worst_scores /= nr_worst_files

        print("nr of best results", nr_best_files)
        print("nr of worst results", nr_worst_files)

        print("### SCORES FOR BEST TRANSFERS ###")
        print(best_scores)
        print(np.std(best_roc_aucs1), np.std(best_avg_prec1))
        print(np.std(best_roc_aucs2), np.std(best_avg_prec2))
        print(np.std(best_roc_aucs3), np.std(best_avg_prec3))
        print("### SCORES FOR WORST TRANSFERS ###")
        print(worst_scores)
        print(np.std(worst_roc_aucs1), np.std(worst_avg_prec1))
        print(np.std(worst_roc_aucs2), np.std(worst_avg_prec2))
        print(np.std(worst_roc_aucs3), np.std(worst_avg_prec3))

        return best_scores, worst_scores
