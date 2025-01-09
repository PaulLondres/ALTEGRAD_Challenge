import pandas as pd
import networkx as nx
from networkx.algorithms import community
from utils import preprocess_dataset
import numpy as np
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
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
    connected_components = list(nx.connected_components(G))
    num_communities = 0
    for component in connected_components:
        subgraph = G.subgraph(component)
        partition = community.asyn_fluidc(subgraph, k=max(2, subgraph.number_of_nodes() // 10))
        num_communities += len(list(partition))

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
    for _, row in tqdm(df.iterrows()):
        graph_id = row['graph_id']
        edge_list_str = row['edge_list']

        # Convertir la chaîne de caractères en liste d'arêtes
        edge_list = eval(edge_list_str)

        # Calculer les métriques
        graph_metrics = compute_metrics_from_edge_list(edge_list)
        metrics.append(graph_metrics)
        graph_ids.append(graph_id)

        # Ajouter l'ID du graphe et les métriques à la matrice
        # metrics.append([
        #     graph_id,
        #     graph_metrics['num_nodes'],
        #     graph_metrics['num_edges'],
        #     graph_metrics['avg_degree'],
        #     graph_metrics['num_triangles'],
        #     graph_metrics['global_clustering'],
        #     graph_metrics['max_k_core'],
        #     graph_metrics['num_communities']
        # ])

    return np.array(metrics), graph_ids

def compute_reference_graph_metrics(dataset_type = "valid", n_max_nodes=50, spectral_emb_dim=10):
    dataset = preprocess_dataset(dataset_type, n_max_nodes, spectral_emb_dim)
    metrics = np.zeros((len(dataset), 7))
    graph_ids = []
    for idx, graph in tqdm(enumerate(dataset)):
        edge_list = [(i, j) for i, j in graph.edge_index.numpy().T]
        graph_id = graph.filename
        graph_ids.append(graph_id)
        graph_metrics = compute_metrics_from_edge_list(edge_list)
        metrics[idx] = graph_metrics
    return metrics, graph_ids

def MAE_normalized(ref, preds):
    mae_rescaled = 0
    for i in range(ref.shape[0]):
        for j in range(ref.shape[1]):
            if ref[i, j] > 0:
                mae_rescaled += np.abs(ref[i, j] - preds[i, j]) /ref[i, j]
            else:
                mae_rescaled += np.abs(ref[i, j] - preds[i, j])
    return mae_rescaled/ref.size

if __name__ == '__main__':
    csv_path_predicted = "validation_dataset_metrics.csv"
    metrics_predicted, graph_ids_pred = compute_predicted_graph_metrics(csv_path_predicted)
    metrics_reference =  np.load("validation_dataset_metrics.npy")#compute_reference_graph_metrics()
    print("MAE : ", mean_absolute_error(metrics_reference, metrics_predicted))
    print("MAE with balanced metric impact : ", MAE_normalized(ref=metrics_reference, preds=metrics_predicted))
