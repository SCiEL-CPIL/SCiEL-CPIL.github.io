import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes a heuristic matrix for Ant Colony Optimization on the Orienteering Problem.
    
    Structural Change: Uses a prize-weighted local centrality (cluster value) 
    to identify nodes that are hubs of high-value regions, rather than just 
    relying on individual prize and distance.
    
    Formula: eta[i, j] = (prize[j] * ClusterValue[j]) / (distance[i, j] + epsilon)
    Where ClusterValue[j] = sum(k != j) prize[k] / (distance[j, k] + epsilon)
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    epsilon = 1e-9
    
    # Compute cluster value for each node j:
    # ClusterValue[j] = sum over k != j of prize[k] / (distance[j, k] + epsilon)
    # This measures how much high-prize "mass" is near node j
    
    # Create a matrix of prize[k] / (distance[j, k] + epsilon) for all j, k
    # distance_matrix is (n, n), prizes is (n,)
    # We want for each j, sum over k of prizes[k] / (distance_matrix[j, k] + epsilon)
    
    # Avoid division by zero on diagonal
    dist_with_eps = distance_matrix + epsilon
    
    # For each j, compute sum_k prizes[k] / dist_with_eps[j, k]
    # But we should exclude k=j (self-distance)
    # Set diagonal to a large value so it contributes ~0
    
    dist_for_cluster = dist_with_eps.copy()
    np.fill_diagonal(dist_for_cluster, np.inf)
    
    # cluster_values[j] = sum_k prizes[k] / dist_for_cluster[j, k]
    # This is (n,) array
    cluster_values = np.sum(prizes[None, :] / dist_for_cluster, axis=1)
    
    # Now compute heuristic: eta[i, j] = (prize[j] * ClusterValue[j]) / (distance[i, j] + epsilon)
    # prize[j] * ClusterValue[j] is a node attractiveness score
    node_attractiveness = prizes * cluster_values
    
    # For each i, j: eta[i, j] = node_attractiveness[j] / (distance[i, j] + epsilon)
    heur = node_attractiveness[None, :] / dist_with_eps
    
    # Ensure no inf/nan
    heur = np.nan_to_num(heur, nan=1e-9, posinf=1e-9, neginf=1e-9)
    
    return heur
