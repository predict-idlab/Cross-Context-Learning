import logging
import os
import pickle
import random
import sys

import numpy as np
import rdflib
from scipy.sparse import csgraph
from tqdm import tqdm_notebook as tqdm

logger = logging.getLogger()
logger.setLevel("INFO")


def extract_graph(subject, graph, fraction=0.1):
    """
    Extract subgraph starting from a subject (in str).
    Does not extract any general TBOX information.
    :param subject:
    :param graph:
    :return:
    """
    total_size = len(graph) * fraction
    current_result = rdflib.Graph()
    new_triples = set([])
    explored_individuals = set([])
    current_triple = rdflib.term.URIRef(subject)
    new_triples.add(current_triple)
    while 0 < len(new_triples) < total_size:
        current_triple = new_triples.pop()
        for s, p, o in graph.triples((current_triple, None, None)):
            current_result.add((s, p, o))
            if o not in explored_individuals:
                new_triples.add(o)
            if len(new_triples) >= total_size:
                break
        explored_individuals.add(current_triple)
    return current_result


def split_rdf_manually(working_folder, rdf_filename, splitting_query, rdf_file_format='xml'):
    """
    Split RDF graph based on query.
    :param working_folder:
    :param rdf_filename:
    :param splitting_query:
    :param rdf_file_format:
    :return:
    """
    logging.info('Reading RDF file manually...')
    graph = rdflib.Graph()
    graph.parse(rdf_filename, format=rdf_file_format)
    # Query what subjects have the given type:
    logging.info('Querying subjects that have the specified type...')
    subjects = graph.query(splitting_query)
    # Write an RDF file per subject:
    logging.info('Writing RDF files per subject...')
    for subject in tqdm(subjects):
        current_output_graph = extract_graph(subject[0], graph)
        print('The extrated graph contains %s percent of the triples from the original graph' % (
            str(len(current_output_graph) / len(graph))))
        logging.info('Writing the graph to file...')
        prefixed_subject = graph.namespace_manager.normalizeUri(subject[0])
        prefixed_subject = prefixed_subject.replace(':', '')
        if '<' in prefixed_subject and '/' in prefixed_subject:
            prefixed_subject = prefixed_subject[prefixed_subject.rfind('/') + 1:-1]
        with open(os.path.join(working_folder, 'graphs_rdf', prefixed_subject.replace(':', '') + '_manually.xml'),
                  'wb') as current_output_file:
            current_output_graph.serialize(current_output_file)


def convert_to_gefsg(full_graph_dir, list_sub_graphs_dirs, output_dir, split_file=False):
    """
    Converts a set of graphs to vertices and edges with assigned IDs.
    Full graphs are necessary to extract the recurrent IDs from the subgraphs.
    :param full_graph_dir:
    :param list_sub_graphs_dirs:
    :param output_dir:
    :param split_file:
    :return:
    """
    # load the whole graph
    graph = rdflib.Graph()
    graph.parse(location=full_graph_dir)
    # first make the ids for all the node labels and edge ids over the whole dataset
    edges_dict = {}
    edges_dict_reverse = {}
    label_dict = {}
    label_dict_reverse = {}
    label_id = 0
    edge_id = 0
    for subj, pred, obj in graph:
        if subj not in label_dict:
            label_dict[subj] = label_id
            label_dict_reverse[label_id] = subj
            label_id += 1
        if obj not in label_dict:
            label_dict[obj] = label_id
            label_dict_reverse[label_id] = obj
            label_id += 1
        if pred not in edges_dict:
            edges_dict[pred] = edge_id
            edges_dict_reverse[edge_id] = pred
            edge_id += 1
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    f = open(output_dir + "/labels.txt", "w+")
    f.write('label ids\n')
    for i in range(0, len(label_dict)):
        f.write(str(i) + ' ' + label_dict_reverse[i] + '\n')
    f.write('edge ids\n')
    for i in range(0, len(edges_dict_reverse)):
        f.write(str(i) + ' ' + edges_dict_reverse[i] + '\n')
    # convert the subgraphs based on the extracted labels
    graphId = 0
    f.write('subgraph ids\n')
    output_file = open(output_dir + "/graphs.txt", "w+")
    for subgraph_dir in list_sub_graphs_dirs:
        subgaph = rdflib.Graph()
        subgaph.parse(location=subgraph_dir)
        output_graph_file = output_file
        if split_file:
            output_graph_file = open(output_dir + "/" + str(graphId) + ".graph", "w+")
        # write the graph id to the labels file
        f.write(str(graphId) + ' ' + subgraph_dir + '\n')
        output_graph_file.write('t # ' + str(graphId) + '\n')
        graphId += 1
        nodes_dict = {}  # mapping nodes to ids
        nodes_dict_reverse = {}
        node_id = 0
        edges = []
        for subj, pred, obj in subgaph:
            if subj not in nodes_dict:
                nodes_dict[subj] = node_id
                nodes_dict_reverse[node_id] = subj
                node_id += 1
            if obj not in nodes_dict:
                nodes_dict[obj] = node_id
                nodes_dict_reverse[node_id] = obj
                node_id += 1
            edges.append((nodes_dict[subj], nodes_dict[obj], edges_dict[pred]))

        for i in range(0, node_id):
            output_graph_file.write('v ' + str(i) + ' ' + str(label_dict[nodes_dict_reverse[i]]) + '\n')
        for i in range(0, len(edges)):
            output_graph_file.write('e %s %s %s\n' % (edges[i][0], edges[i][1], edges[i][2]))
        if split_file:
            output_graph_file.close()
    output_file.close()
    f.close()


def convert_to_json(full_graph_dir, list_sub_graphs_dirs, output_dir):
    """
    Convert a set of graphs to vertices and edges with assigned IDs to json format.
    Full graphs are necessary to extract the recurrent IDs from the subgraphs.
    :param full_graph_dir:
    :param list_sub_graphs_dirs:
    :param output_dir:
    :return:
    """
    # load the whole graph
    graph = rdflib.Graph()
    graph.parse(location=full_graph_dir)
    # first make the ids for all the node labels and edge ids over the whole dataset
    edges_dict = {}
    edges_dict_reverse = {}
    label_dict = {}
    label_dict_reverse = {}
    label_id = 0
    edge_id = 0
    for subj, pred, obj in graph:
        if subj not in label_dict:
            label_dict[subj] = label_id
            label_dict_reverse[label_id] = subj
            label_id += 1
        if obj not in label_dict:
            label_dict[obj] = label_id
            label_dict_reverse[label_id] = obj
            label_id += 1
        if pred not in edges_dict:
            edges_dict[pred] = edge_id
            edges_dict_reverse[edge_id] = pred
            edge_id += 1
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    f = open(output_dir + "/labels.txt", "w+")
    f.write('label ids\n')
    for i in range(0, len(label_dict)):
        f.write(str(i) + ' ' + label_dict_reverse[i] + '\n')
    f.write('edge ids\n')
    for i in range(0, len(edges_dict_reverse)):
        f.write(str(i) + ' ' + edges_dict_reverse[i] + '\n')
    # convert the subgraphs based on the extracted labels
    graphId = 0
    f.write('subgraph ids\n')
    for subgraph_dir in list_sub_graphs_dirs:
        # create new graph file with id as file name
        output_file = open(output_dir + "/" + str(graphId) + ".json", "w+")

        subgaph = rdflib.Graph()
        subgaph.parse(location=subgraph_dir)
        # write the graph id to the labels file
        f.write(str(graphId) + ' ' + subgraph_dir + '\n')

        graphId += 1
        nodes_dict = {}  # mapping nodes to ids
        nodes_dict_reverse = {}
        node_id = 0
        edges = []
        for subj, pred, obj in subgaph:
            if subj not in nodes_dict:
                nodes_dict[subj] = node_id
                nodes_dict_reverse[node_id] = subj
                node_id += 1
            if obj not in nodes_dict:
                nodes_dict[obj] = node_id
                nodes_dict_reverse[node_id] = obj
                node_id += 1
            edges.append((edges_dict[pred], nodes_dict[subj], nodes_dict[obj]))
        output_file.write('{')
        output_file.write('"edges": [')
        for i in range(0, len(edges)):
            output_file.write('[%s, %s]' % (edges[i][1], edges[i][2]))
            if i < len(edges) - 1:
                output_file.write(', ')
        output_file.write(']')

        output_file.write(', "features": {')
        for i in range(0, node_id):
            output_file.write('"%s": "%s" ' % (str(i), str(label_dict[nodes_dict_reverse[i]])))
            if i < node_id - 1:
                output_file.write(', ')
        output_file.write('}')
        output_file.write('}')
        output_file.close()
    f.close()


def convert_to_json_edge_relabling(list_sub_graphs, output_dir, label_dict=None, label_dict_reverse=None):
    """
    Relable the graph such that edge labels are used as nodes.
    :param list_sub_graphs:
    :param output_dir:
    :param label_dict:
    :param label_dict_reverse:
    :return:
    """

    if not label_dict or not label_dict_reverse:
        label_dict = {}
        label_dict_reverse = {}
        print('Convert to json with edge relabeling...')
        # first make the ids for all the node labels and edge ids over the whole dataset
        label_id = 0
        for sub_graph in list_sub_graphs:
            for subj, pred, obj in sorted(list(sub_graph)):
                if subj not in label_dict:
                    label_dict[subj] = label_id
                    label_dict_reverse[label_id] = subj
                    label_id += 1
                if obj not in label_dict:
                    label_dict[obj] = label_id
                    label_dict_reverse[label_id] = obj
                    label_id += 1
                if pred not in label_dict:
                    label_dict[pred] = label_id
                    label_dict_reverse[label_id] = pred
                    label_id += 1
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        else:
            for entry in os.listdir(output_dir):
                print("removing old json file before relabeling:", entry)
                os.remove(os.path.join(output_dir, entry))
        f = open(output_dir + "/labels.txt", "w+")
        f.write('label\tids\n')
        for i in range(0, len(label_dict)):
            if len(label_dict_reverse[i].split(' ')) > 1:
                # f.write(str(i) + '\t' + '"' + "\\n".join(label_dict_reverse[i].splitlines()) + '"' + '\n')
                f.write(
                    str(i) + '\t' + '"' + label_dict_reverse[i].replace('\n', '\\n').replace('\r', '\\r') + '"' + '\n')
            else:
                f.write(str(i) + '\t' + label_dict_reverse[i] + '\n')
        f.close()

    # convert the subgraphs based on the extracted labels and relable the edges
    names = []
    graphId = 0
    for sub_graph in list_sub_graphs:
        # create new graph file with id as file name
        output_file = open(output_dir + "/" + str(graphId) + ".json", "w+")

        names.append(str(graphId) + ".json")

        print("creating new json file:", names[-1])

        graphId += 1
        nodes_dict = {}  # mapping nodes to ids
        nodes_dict_reverse = {}
        node_id = 0
        edge_dict = {}  # mapping edges to ids
        edge_dict_reverse = {}
        edge_id = 0
        edges = []
        for subj, pred, obj in sorted(list(sub_graph)):
            if subj not in nodes_dict:
                nodes_dict[subj] = node_id
                nodes_dict_reverse[node_id] = subj
                node_id += 1
            if obj not in nodes_dict:
                nodes_dict[obj] = node_id
                nodes_dict_reverse[node_id] = obj
                node_id += 1
            if pred not in edge_dict:
                edge_dict[pred] = []
            edge_dict[pred].append(node_id)
            nodes_dict_reverse[node_id] = pred
            edges.append((nodes_dict[subj], node_id, nodes_dict[obj]))
            node_id += 1
        output_file.write('{')
        output_file.write('"edges": [')
        features = {}
        for i in range(0, len(edges)):
            output_file.write('[%s, %s], [%s, %s]' % (edges[i][0], edges[i][1], edges[i][1], edges[i][2]))
            if i < len(edges) - 1:
                output_file.write(', ')
        output_file.write(']')

        output_file.write(', "features": {')
        for i in range(0, node_id):
            output_file.write('"%s": "%s" ' % (str(i), str(label_dict[nodes_dict_reverse[i]])))
            if i < node_id - 1:
                output_file.write(', ')
        output_file.write('}')
        output_file.write('}')
        output_file.close()

    return label_dict, label_dict_reverse, names


def convert_to_dict(full_graph_dir, list_sub_graphs_dirs, label_dict={},
                    label_dict_reverse={}):
    """
    Relabel graph such that edge labels are used as nodes.
    :param full_graph_dir:
    :param list_sub_graphs_dirs:
    :param label_dict:
    :param label_dict_reverse:
    :return:
    """
    if not label_dict or not label_dict_reverse:
        print('Loading the graph')
        # load the whole graph
        graph = rdflib.Graph()
        graph.parse(location=full_graph_dir)
        # first make the ids for all the node labels and edge ids over the whole dataset
        label_id = 0
        for subj, pred, obj in graph:
            if subj not in label_dict:
                label_dict[subj] = label_id
                label_dict_reverse[label_id] = subj
                label_id += 1
            if obj not in label_dict:
                label_dict[obj] = label_id
                label_dict_reverse[label_id] = obj
                label_id += 1
            if pred not in label_dict:
                label_dict[pred] = label_id
                label_dict_reverse[label_id] = pred
                label_id += 1

    # convert the subgraphs based on the extracted labels and relable the edges
    graphId = -1
    graphs = {}
    for subgraph_dir in list_sub_graphs_dirs:
        print('loading graph ' + subgraph_dir)

        # create new graph file with id as file name
        subgaph = rdflib.Graph()
        subgaph.parse(location=subgraph_dir)

        graphId += 1
        nodes_dict = {}  # mapping nodes to ids
        nodes_dict_reverse = {}
        node_id = 0
        edge_dict = {}  # mapping nodes to ids
        edge_dict_reverse = {}
        edge_id = 0
        edges = []
        edge_node_dict = {}
        for subj, pred, obj in subgaph:
            if subj not in nodes_dict:
                nodes_dict[subj] = node_id
                nodes_dict_reverse[node_id] = subj
                node_id += 1
            if obj not in nodes_dict:
                nodes_dict[obj] = node_id
                nodes_dict_reverse[node_id] = obj
                node_id += 1
            if pred not in edge_dict:
                edge_dict[pred] = []
            edge_dict[pred].append(node_id)
            nodes_dict_reverse[node_id] = pred
            edges.append((nodes_dict[subj], node_id, nodes_dict[obj]))
            if not nodes_dict[subj] in edge_node_dict:
                edge_node_dict[nodes_dict[subj]] = []
            edge_node_dict[nodes_dict[subj]].append(nodes_dict[obj])
            if not nodes_dict[obj] in edge_node_dict:
                edge_node_dict[nodes_dict[obj]] = []
            edge_node_dict[nodes_dict[obj]].append(nodes_dict[subj])
            node_id += 1
        graphs[graphId] = {}
        for node in range(node_id):
            if node in edge_node_dict:
                graphs[graphId][node] = {'neighbors': list(set(edge_node_dict[node])),
                                         'label': (label_dict[nodes_dict_reverse[node]],)}
            else:
                graphs[graphId][node] = {'neighbors': [], 'label': (label_dict[nodes_dict_reverse[node]],)}

    return graphs, label_dict_reverse


def get_similarity(graph1, graph2):
    """
    Get similarity percentage between graphs.
    :param graph1:
    :param graph2:
    :return:
    """
    sim_counter = 0
    for triple in graph1:
        if triple in graph2:
            sim_counter += 1
    return sim_counter / len(graph1)


def get_similarity_alt(graph1, graph2):
    """
    Get similarity percentage between graphs.
    :param graph1:
    :param graph2:
    :return:
    """
    sim_counter = 0
    for triple in graph1:
        if triple in graph2:
            sim_counter += 1
    return sim_counter / len(graph2)


def explore(subject, graph, depth, explored, results):
    """
    Recursively traverse graph starting from a given subject.
    :param subject:
    :param graph:
    :param depth:
    :param explored:
    :param results:
    :return:
    """
    for s, p, o in graph.triples((None, None, subject)):
        if not depth in results:
            results[depth] = []
        results[depth].append((s, p, o))
        if (s, p, o) not in explored:
            explored.append((s, p, o))
            explore(s, graph, depth + 1, explored, results)
        for s2, p2, o2 in graph.triples((subject, None, None)):
            if (s2, p2, o2) not in explored:
                explored.append((s2, p2, o2))
                explore(o2, graph, depth + 1, explored, results)


def convert_to_from_gradually(graph1, graph2, output_dest, steps):
    """
    Convert graph1 gradually to graph2 and save the changing graph to the output_dest.
    Steps defines the number of intermediate graphs that will be saved.
    :param graph1:
    :param graph2:
    :param output_dest:
    :param steps:
    :return:
    """
    ignoreP = ['http://www.w3.org/2002/07/owl#disjointWith', 'http://www.w3.org/2000/01/rdf-schema#subClassOf']
    ignoreO = ['http://www.w3.org/2002/07/owl#Class']
    overlap = []
    for (s, p, o) in graph1:
        if (None, p, o) in graph2 and str(p) not in ignoreP and str(o) not in ignoreO:
            overlap.append((s, p, o))
    if len(overlap) == 0:
        print('No overlap in graphs')
        return False
    start_subj = overlap[0][2]
    results = {}
    if not os.path.exists(output_dest):
        os.mkdir(output_dest)
    print("Start subject:", start_subj)
    explore(start_subj, graph1, 0, [], results)
    results_start = results
    results = {}
    explore(start_subj, graph2, 0, [], results)
    results_test = results
    if steps > len(results_start):
        print('# steps is larger than the possible changes')
        steps = len(results_start) - 1
    print("# intermediate graphs:", steps)
    print("# possible changes:", results_start)
    # note that test graph is deeper
    trace = []
    total_remove, total_add = [], []
    for i in range(0, len(results_start) - 1):
        if i < len(results_test):
            start_index = len(results_start) - 1 - i
            remove_triples = results_start[start_index]

            print('removing %s triples' % (len(remove_triples)))
            add_triples = results_test[i]
            for remove_triple in remove_triples:
                graph1.remove(remove_triple)

            print('Adding %s triples' % (len(add_triples)))
            for add_triple in add_triples:
                graph1.add(add_triple)

            total_remove.extend(remove_triples)
            total_add.extend(add_triples)
            if i % steps == 0:
                print(output_dest + '/dg_' + str((i // 10) + 1) + '.xml')
                trace.append((total_remove, total_add))
                graph1.serialize(destination=output_dest + '/dg_' + str((i // 10) + 1) + '.xml', format='xml')
                print('sim new graph', get_similarity(graph1, graph2))
                total_remove.clear()
                total_add.clear()

    return trace


def convert_from_gradually(graph, output_dest, steps, variable=False):
    """
    Convert graph gradually to an empty graph.
    Steps defines the number of intermediate graphs that will be saved.
    :param graph:
    :param output_dest:
    :param steps:
    :param variable:
    :return:
    """

    def get_nodes(graph):
        subjects = set(graph.subjects())
        objects = set(graph.objects())
        return list(subjects.union(objects))

    remaining_nodes = get_nodes(graph)
    total_count = len(remaining_nodes)
    bulk = total_count // steps
    trace = []
    for step in range(steps):
        remaining_nodes = get_nodes(graph)
        print(len(remaining_nodes), len(graph))
        if len(remaining_nodes) > 0:
            remaining_indices = list(range(len(remaining_nodes)))
            if variable:
                bulk = random.randint(bulk // 2, bulk * 2)
            remove_indices = random.sample(remaining_indices, min(len(remaining_indices), bulk))
            remove_nodes = [triple for i, triple in enumerate(remaining_nodes) if i in remove_indices]
            for node in remove_nodes:
                remove_triples = set(graph.triples((node, None, None)))
                remove_triples = list(remove_triples.union(set(graph.triples((None, None, node)))))
                for triple in remove_triples:
                    graph.remove(triple)
            trace.append((step, remove_nodes, []))
            print("Number of connected components:", get_nr_connected_components(graph))
            graph.serialize(destination=output_dest + '/dg_' + str(step) + '.xml', format='xml')

    return trace


def convert_from_gradually_with_connectivity(graph, root, output_dest, steps, variable=False):
    """
    Convert graph gradually to an empty graph, taking into consideration connectivity.
    Steps defines the number of intermediate graphs that will be saved.
    :param root:
    :param graph:
    :param output_dest:
    :param steps:
    :param variable:
    :return:
    """

    def get_nodes(graph):
        subjects = set(graph.subjects())
        objects = set(graph.objects())
        return list(subjects.union(objects))

    def get_least_connected(nodes, graph):
        chosen, connectivity = None, sys.maxsize
        for node in nodes:
            outgoing = list(graph.triples((node, None, None)))
            incoming = list(graph.triples((None, None, node)))
            if len(outgoing) + len(incoming) < connectivity:
                connectivity = len(outgoing) + len(incoming)
                chosen = node
        return chosen

    # work backwards (i.e. expand rather than contract) to ensure connectivity
    start_graph, dest_graph = rdflib.Graph(), graph
    print("Number of connected components in final graph:", get_nr_connected_components(dest_graph))

    dest_nodes = get_nodes(dest_graph)
    total_count = len(dest_nodes)
    bulk = total_count // steps

    # get node with lowest connectivity
    root = get_least_connected(dest_nodes, dest_graph)
    print("Staring from node with lowest connectivity:", root)

    to_visit, visited, trace = [root], [], []
    for step in range(steps - 1, -1, -1):
        print("Step:", step)
        dest_nodes = get_nodes(dest_graph)
        start_nodes = get_nodes(start_graph)
        print(len(start_nodes), len(dest_nodes), len(start_graph), len(dest_graph))
        if len(start_nodes) < len(dest_nodes):
            if variable:
                bulk = random.randint(bulk // 2, bulk * 2)
            print("Number of nodes to be removed:", bulk)
            newly_visited = []
            while len(newly_visited) < bulk:
                node = get_least_connected(to_visit, dest_graph)
                outgoing = dest_graph.triples((node, None, None))
                incoming = dest_graph.triples((None, None, node))
                start_graph += outgoing
                start_graph += incoming
                visited.append(node)
                newly_visited.append(node)
                to_visit.remove(node)
                for _node in get_nodes(start_graph):
                    if _node not in to_visit \
                            and _node not in visited:
                        to_visit.append(_node)

            trace.append((step, newly_visited, []))
            print("Number of connected components:", get_nr_connected_components(start_graph))
            start_graph.serialize(destination=output_dest + '/dg_' + str(step) + '.xml', format='xml')
            print("sim new graph", get_similarity_alt(start_graph, dest_graph))

    trace.reverse()
    return trace


def dump_to_pickle(data, file_path):
    """
    Write a binary dump to a pickle file of arbitrary size.
    src: https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb
    :param data:
    :param setup:
    :param file_path:
    :return:
    """
    bytes_out = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join("temp", file_path), 'wb') as f_out:
        # 2**31 - 1 is the max nr. of bytes pickle can dump at a time
        for idx in range(0, len(bytes_out), 2 ** 31 - 1):
            f_out.write(bytes_out[idx:idx + 2 ** 31 - 1])
    return


def load_from_pickle(file_path):
    """
    Read from a pickle file of arbitrary size.
    src: https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb
    :param setup:
    :param file_path:
    :return:
    """
    bytes_in = bytearray(0)
    input_size = os.path.getsize(os.path.join("temp", file_path))
    with open(os.path.join("temp", file_path), 'rb') as f_in:
        for _ in range(0, input_size, 2 ** 31 - 1):
            bytes_in += f_in.read(2 ** 31 - 1)
    return pickle.loads(bytes_in)


def get_nr_connected_components(graph):
    """
    Partition the matrix representation of a graph into connected components.
    :param graph:
    :return:
    """
    entities = set(graph.subjects())
    entities = list(entities.union(set(graph.objects())))
    adjacency_matrix = np.zeros((len(entities), len(entities)))
    for i, entity in enumerate(entities):
        for triple in graph.triples((entity, None, None)):
            j = entities.index(triple[2])
            adjacency_matrix[i, j] = 1

    n_components, labels = csgraph.connected_components(adjacency_matrix, directed=True)
    return n_components
