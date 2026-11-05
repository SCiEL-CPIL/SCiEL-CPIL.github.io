import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 10.0  # 10s safety margin out of 120s
    
    # Convert to numpy array if not already
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency, dtype=np.int8)
        
    n = adjacency.shape[0]
    
    # Compute degrees
    degrees = adjacency.sum(axis=1)
    
    # Compute sum of neighbor degrees
    sum_neighbor_degrees = adjacency @ degrees
    
    # Compute average neighbor degree
    avg_neighbor_degrees = np.zeros(n, dtype=float)
    mask = degrees > 0
    avg_neighbor_degrees[mask] = sum_neighbor_degrees[mask] / degrees[mask]
    
    # Compute Clustering Coefficient
    # Number of triangles for each node i is (A^3)[i][i] / 2
    A_sq = adjacency @ adjacency
    A_cu = A_sq @ adjacency
    triangles = A_cu.diagonal() / 2.0
    
    # Clustering coefficient C_i = 2 * Triangles_i / (d_i * (d_i - 1))
    clustering = np.zeros(n, dtype=float)
    valid_clustering_mask = degrees > 1
    denom = degrees[valid_clustering_mask] * (degrees[valid_clustering_mask] - 1)
    clustering[valid_clustering_mask] = (2.0 * triangles[valid_clustering_mask]) / denom
    
    # Heuristic:
    # 1. Low degree is better (negative weight)
    # 2. Low clustering is better (positive weight on (1 - clustering))
    #    Vertices in sparse neighborhoods allow more independent set growth.
    # 3. High average neighbor degree is better (positive weight)
    #    Sacrificing hubs is better than sacrificing leaves.
    
    alpha = 1.0
    beta = 1.5
    gamma = 1.0
    
    scores = -alpha * degrees.astype(float) + beta * (1.0 - clustering) + gamma * avg_neighbor_degrees
    
    # Check time
    if time.time() > deadline:
        return -degrees.astype(float)
        
    return scores
