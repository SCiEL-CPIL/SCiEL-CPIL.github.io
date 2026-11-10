import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes a heuristic matrix for Ant Colony Optimization on the Orienteering Problem.
    
    Structural Change: Models the heuristic as a learned linear combination of
    graph-theoretic centrality features (prize-weighted eigenvector centrality,
    degree centrality, prize-weighted betweenness centrality) rather than just
    hand-crafted prize/distance ratios. The optimizer discovers global structural
    attractors that guide ants toward high-value hubs.
    
    Formula: eta[i, j] = (linear_combo_of_centralities[j]) / (distance[i, j] + epsilon)
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    epsilon = 1e-9
    
    # Compute graph-theoretic centrality features
    
    # 1. Prize-weighted eigenvector centrality
    # This captures the "influence" of each node in the prize-weighted graph
    # Use power iteration to find the dominant eigenvector of the prize-weighted adjacency matrix
    W = 1.0 / (distance_matrix + epsilon)
    np.fill_diagonal(W, 0.0)
    
    # Prize-weighted adjacency: A[i,j] = W[i,j] * (prizes[i] + prizes[j]) / 2
    prize_avg = (prizes[None, :] + prizes[:, None]) / 2.0
    A = W * prize_avg
    
    # Power iteration for eigenvector centrality
    v = np.ones(n) / n
    for _ in range(50):
        v_new = A @ v
        norm = np.linalg.norm(v_new)
        if norm < 1e-15:
            break
        v = v_new / norm
    
    # Ensure non-negative
    eigenvector_centrality = np.maximum(v, 0.0)
    
    # 2. Degree centrality (prize-weighted)
    # Sum of weights connected to each node, weighted by the prize of the neighbor
    degree_centrality = W @ prizes
    
    # 3. Prize-weighted betweenness centrality (approximate)
    # For each pair (i, k), count how often j lies on shortest paths, weighted by prizes
    # This is expensive, so use a simplified version: 
    # Betweenness[j] = sum_{i,k != j} prize[i] * prize[k] * (1 if j is "between" i and k)
    # Approximation: use the fact that in a metric space, j is between i and k if
    # d[i,k] ≈ d[i,j] + d[j,k]. We'll use a soft approximation.
    
    # Simplified betweenness: for each j, compute how many pairs (i,k) have j as a "hub"
    # Use: betweenness[j] = sum_{i,k != j} prize[i] * prize[k] * exp(-alpha * |d[i,k] - d[i,j] - d[j,k]|)
    # This is O(n^3) which is fine for n=50
    
    alpha = 10.0  # Sharpness of the "betweenness" kernel
    betweenness = np.zeros(n)
    
    for j in range(n):
        if j == 0:
            continue
        # For all pairs (i, k) where i, k != j and i != k
        mask_i = np.ones(n, dtype=bool)
        mask_i[j] = False
        mask_k = np.ones(n, dtype=bool)
        mask_k[j] = False
        
        # d[i,k] for all i,k
        d_ik = distance_matrix
        d_ij = distance_matrix[:, j]  # d[i,j] for all i
        d_jk = distance_matrix[j, :]  # d[j,k] for all k
        
        # |d[i,k] - d[i,j] - d[j,k]|
        diff = np.abs(d_ik - d_ij[:, None] - d_jk[None, :])
        
        # Weight by prizes
        prize_weight = prizes[None, :] * prizes[:, None]
        
        # Mask out i=j, k=j, i=k
        mask = mask_i[:, None] & mask_k[None, :]
        eye = np.eye(n, dtype=bool)
        mask = mask & ~eye
        
        # Soft betweenness kernel
        kernel = np.exp(-alpha * diff)
        kernel = kernel * mask
        
        betweenness[j] = np.sum(prize_weight * kernel)
    
    # Normalize features
    def normalize(x):
        x_min = np.min(x)
        x_max = np.max(x)
        if x_max - x_min < 1e-15:
            return np.ones_like(x) / n
        return (x - x_min) / (x_max - x_min)
    
    eig_norm = normalize(eigenvector_centrality)
    deg_norm = normalize(degree_centrality)
    bet_norm = normalize(betweenness)
    
    # Learn linear combination weights via local optimization
    # We want to find weights w1, w2, w3 such that the heuristic leads to good ACO performance
    # Use a simple grid search over a small set of weight combinations
    
    start_time = time.time()
    deadline = start_time + 250.0  # Leave 80s margin for safety
    
    # Precompute common terms
    prize_row = prizes[None, :]  # (1, n)
    d_ij = distance_matrix + epsilon  # (n, n)
    
    # Feature matrix: (3, n)
    features = np.vstack([eig_norm, deg_norm, bet_norm])  # (3, n)
    
    # Grid of weight combinations
    weight_grids = [
        (1.0, 0.5, 0.3),
        (0.8, 0.6, 0.4),
        (0.7, 0.7, 0.3),
        (0.6, 0.8, 0.4),
        (0.5, 0.7, 0.5),
        (0.9, 0.4, 0.3),
        (1.0, 0.3, 0.5),
        (0.8, 0.5, 0.5),
        (0.7, 0.6, 0.4),
        (0.6, 0.7, 0.5),
        (0.5, 0.8, 0.4),
        (0.4, 0.7, 0.5),
        (0.3, 0.8, 0.6),
        (0.5, 0.5, 0.7),
        (0.4, 0.6, 0.8),
        (0.3, 0.7, 0.9),
        (0.2, 0.6, 1.0),
        (0.1, 0.5, 1.0),
        (0.0, 0.4, 1.0),
        (0.0, 0.3, 1.0),
        (0.0, 0.2, 1.0),
        (0.0, 0.1, 1.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.5, 0.5, 0.0),
        (0.3, 0.7, 0.0),
        (0.2, 0.8, 0.0),
        (0.1, 0.9, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 0.5, 0.0),
        (0.8, 0.4, 0.0),
        (0.6, 0.3, 0.0),
        (0.4, 0.2, 0.0),
        (0.2, 0.1, 0.0),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 0.5),
        (0.8, 0.8, 0.4),
        (0.6, 0.6, 0.3),
        (0.5, 0.5, 0.2),
        (0.4, 0.4, 0.1),
        (1.0, 0.8, 0.6),
        (0.9, 0.7, 0.5),
        (0.8, 0.6, 0.4),
        (0.7, 0.5, 0.3),
        (0.6, 0.4, 0.2),
        (0.5, 0.3, 0.1),
        (0.4, 0.2, 0.0),
        (0.3, 0.1, 0.0),
        (0.2, 0.0, 0.0),
        (0.1, 0.0, 0.0),
    ]
    
    best_weights = (1.0, 0.5, 0.3)
    best_score = -1.0
    
    # Quick evaluation: use a simplified ACO simulation with fewer iterations
    def quick_evaluate(weights):
        w1, w2, w3 = weights
        # Compute combined centrality
        centrality = w1 * eig_norm + w2 * deg_norm + w3 * bet_norm
        
        # Build heuristic
        numerator = centrality[None, :]
        denominator = d_ij
        h = numerator / denominator
        
        # Ensure positive
        h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
        
        # Quick ACO simulation: 5 iterations, 5 ants
        rng = np.random.RandomState(123)
        num_ants = 5
        num_iters = 5
        pheromone = np.ones((n, n))
        dist_to_home = distance_matrix[:, 0].copy()
        
        global_best = 0.0
        
        for it in range(num_iters):
            iter_best = 0.0
            
            for ant in range(num_ants):
                current = 0
                visited = np.zeros(n, dtype=bool)
                visited[0] = True
                path_len = 0.0
                total_prize = 0.0
                
                while True:
                    remaining = max_len - path_len
                    feasible = (~visited) & ((distance_matrix[current, :] + dist_to_home) <= remaining)
                    feasible[0] = False
                    
                    if not np.any(feasible):
                        break
                    
                    candidates = np.where(feasible)[0]
                    weights_aco = pheromone[current, candidates] * h[current, candidates]
                    
                    total_w = weights_aco.sum()
                    if total_w <= 0:
                        break
                    
                    probs = weights_aco / total_w
                    idx = rng.choice(len(candidates), p=probs)
                    next_node = candidates[idx]
                    
                    path_len += distance_matrix[current, next_node]
                    total_prize += prizes[next_node]
                    visited[next_node] = True
                    current = next_node
                
                if total_prize > iter_best:
                    iter_best = total_prize
            
            pheromone *= 0.9
            pheromone += iter_best / 10.0
            
            if iter_best > global_best:
                global_best = iter_best
        
        return global_best
    
    # Evaluate weight combinations
    for weights in weight_grids:
        if time.time() > deadline:
            break
        
        score = quick_evaluate(weights)
        if score > best_score:
            best_score = score
            best_weights = weights
    
    # Use the best weights found
    w1, w2, w3 = best_weights
    centrality = w1 * eig_norm + w2 * deg_norm + w3 * bet_norm
    
    # Build final heuristic
    numerator = centrality[None, :]
    denominator = d_ij
    heur = numerator / denominator
    
    # Handle any numerical issues
    heur = np.nan_to_num(heur, nan=1e-9, posinf=1e-9, neginf=1e-9)
    
    # Ensure positive values
    heur[heur <= 0] = 1e-9
    
    return heur
