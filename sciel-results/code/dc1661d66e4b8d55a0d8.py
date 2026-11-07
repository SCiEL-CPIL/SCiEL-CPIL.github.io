import numpy as np

def _recover_coords(dist):
    n = dist.shape[0]
    sq_dist = dist ** 2
    row_means = sq_dist.mean(axis=1)
    col_means = sq_dist.mean(axis=0)
    grand_mean = sq_dist.mean()
    B = -0.5 * (sq_dist - row_means[:, np.newaxis] - col_means[np.newaxis, :] + grand_mean)
    
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    top2_eigs = eigenvalues[:2]
    top2_vecs = eigenvectors[:, :2]
    top2_eigs = np.maximum(top2_eigs, 0)
    coords = top2_vecs * np.sqrt(top2_eigs)
    return coords

def _bitonic_tsp_edges(coords):
    n = len(coords)
    if n <= 2:
        return set([(0, 1), (1, 0)]) if n == 2 else set()
    
    centroid = coords.mean(axis=0)
    angles = np.arctan2(coords[:, 1] - centroid[1], coords[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    sorted_coords = coords[sorted_indices]
    
    INF = float('inf')
    dp = [[INF] * n for _ in range(n)]
    pred = [[-1] * n for _ in range(n)]
    
    dp[0][1] = np.linalg.norm(sorted_coords[0] - sorted_coords[1])
    
    for j in range(2, n):
        for i in range(0, j):
            if i < j - 1:
                dp[i][j] = dp[i][j-1] + np.linalg.norm(sorted_coords[j-1] - sorted_coords[j])
                pred[i][j] = i
            elif i == j - 1:
                min_cost = INF
                best_k = -1
                for k in range(0, i):
                    cost = dp[k][i] + np.linalg.norm(sorted_coords[k] - sorted_coords[j])
                    if cost < min_cost:
                        min_cost = cost
                        best_k = k
                dp[i][j] = min_cost
                pred[i][j] = best_k
    
    edges = set()
    i, j = n-2, n-1
    while not (i == 0 and j == 1):
        if i > j:
            i, j = j, i
        if i < j - 1:
            edges.add((j-1, j))
            j = j - 1
        elif i == j - 1:
            k = pred[i][j]
            edges.add((k, j))
            i, j = k, i
    
    edges.add((0, 1))
    edges.add((n-2, n-1))
    
    orig_edges = set()
    for u, v in edges:
        ou = sorted_indices[u]
        ov = sorted_indices[v]
        orig_edges.add((ou, ov))
        orig_edges.add((ov, ou))
        
    return orig_edges

def heuristics(distance_matrix):
    dist = np.array(distance_matrix)
    n = dist.shape[0]
    
    coords = _recover_coords(dist)
    tour_edges = _bitonic_tsp_edges(coords)
    
    eps = 1e-9
    base_heur = 1.0 / (dist + eps)
    
    boost_factor = 20.0
    for u, v in tour_edges:
        base_heur[u][v] *= boost_factor
        base_heur[v][u] *= boost_factor
        
    base_heur[base_heur <= 0] = 1e-9
    
    return base_heur
