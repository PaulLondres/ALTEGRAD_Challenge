import pandas as pd
import networkx as nx
#from networkx.algorithms import community
from utils import preprocess_dataset
import numpy as np
from sklearn.metrics import mean_absolute_error
import pickle

def compute_metrics_from_edge_list(edge_list, return_dict = False):
    # Créer le graphe
    G = nx.Graph()
    G.add_edges_from(edge_list)

    # Calculer les métriques
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    avg_degree = sum(dict(G.degree()).values()) / num_nodes if num_nodes > 0 else 0
    num_triangles = sum(nx.triangles(G).values()) // 3
    global_clustering = nx.transitivity(G)
    max_k_core = max(nx.core_number(G).values()) if num_nodes > 0 else 0

    # Détecter les communautés
    # communities = list(community.greedy_modularity_communities(G))
    # num_communities = len(communities)
    # Détecter les communautés avec une méthode exacte (partitionnement de Louvain)
    #partition = community.asyn_fluidc(G, k=max(2, num_nodes // 10))
    # num_communities = nx.number_connected_components(G)#len(list(partition))
    # connected_components = list(nx.connected_components(G))
    # num_communities = 0
    # for component in connected_components:
    #     subgraph = G.subgraph(component)
    #     partition = community.asyn_fluidc(subgraph, k=max(2, subgraph.number_of_nodes() // 10))
    #     num_communities += len(list(partition))
    partition = nx.community.louvain_communities(G)
    num_communities = len(partition)#len(set(partition.values()))

    if return_dict:
        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'avg_degree': avg_degree,
            'num_triangles': num_triangles,
            'global_clustering': global_clustering,
            'max_k_core': max_k_core,
            'num_communities': num_communities
        }
    else:
        return [num_nodes, num_edges,avg_degree, num_triangles, global_clustering, max_k_core, num_communities]

def compute_predicted_graph_metrics(csv_file_path):
    # Lire le fichier CSV
    df = pd.read_csv(csv_file_path, header=0)

    # Matrice pour stocker les métriques
    metrics = []
    graph_ids = []
    for _, row in df.iterrows():
        graph_id = row['graph_id']
        edge_list_str = row['edge_list']

        # Convertir la chaîne de caractères en liste d'arêtes
        if isinstance(edge_list_str, str):
            edge_list = eval(edge_list_str)
            if isinstance(edge_list[0], int):
                edge_list = tuple([edge_list]) #Manage single edge case
            # Calculer les métriques
            graph_metrics = compute_metrics_from_edge_list(edge_list)
            metrics.append(graph_metrics)
            graph_ids.append(graph_id)
        else:
            metrics.append(np.zeros(7))

    return np.array(metrics), graph_ids

def compute_reference_graph_metrics(dataset_type = "valid", n_max_nodes=50, spectral_emb_dim=10):
    dataset = preprocess_dataset(dataset_type, n_max_nodes, spectral_emb_dim)
    metrics = np.zeros((len(dataset), 7))
    graph_ids = []
    for idx, graph in enumerate(dataset):
        edge_list = [(i, j) for i, j in graph.edge_index.numpy().T]
        graph_id = graph.filename
        graph_ids.append(graph_id)
        graph_metrics = compute_metrics_from_edge_list(edge_list)
        metrics[idx] = graph_metrics
    return metrics, graph_ids

def MAE_normalized(ref, preds):
    stds = np.std(ref, axis=0)
    means = np.mean(ref, axis=0)
    centered_reduced_ref = (ref - np.repeat(means[np.newaxis, :], 1000, axis=0))/stds
    centered_reduced_preds = (preds - np.repeat(means[np.newaxis, :], 1000, axis=0)) / stds
    final_mae = np.mean(np.abs(centered_reduced_ref - centered_reduced_preds))
    final_mae_per_metric = np.mean(np.abs(centered_reduced_ref - centered_reduced_preds), axis=0)
    return final_mae, final_mae_per_metric

def compute_prediction_mae(csv_file_path, ref_npy_path="validation_dataset_metrics.npy"):
    metrics_predicted, graph_ids_pred = compute_predicted_graph_metrics(csv_file_path)
    metrics_reference = np.load(ref_npy_path)  # compute_reference_graph_metrics() #
    mae, mae_per_metric = MAE_normalized(ref=metrics_reference, preds=metrics_predicted)
    return mae, mae_per_metric

if __name__ == '__main__':
    csv_path_predicted = "validation_dataset_metrics.csv"
    metrics_predicted, graph_ids_pred = compute_predicted_graph_metrics(csv_path_predicted)
    metrics_reference =  np.load("validation_dataset_metrics.npy") # compute_reference_graph_metrics() #
    print("MAE : ", mean_absolute_error(metrics_reference, metrics_predicted))
    print("MAE with balanced metric impact : ", MAE_normalized(ref=metrics_reference, preds=metrics_predicted))
