import numpy as np
import time

def _two_opt_improve(tour, dist):
    n = len(tour) - 1
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            if improved:
                break
            a, b = tour[i], tour[i+1]
            for j in range(i+2, n):
                if j == n - 1:
                    c, d = tour[j], tour[0]
                else:
                    c, d = tour[j], tour[j+1]
                delta = dist[a][c] + dist[b][d] - dist[a][b] - dist[c][d]
                if delta < -1e-10:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    improved = True
                    break
    return tour

def _nearest_neighbor_tour(dist, start=0):
    n = len(dist)
    visited = [False] * n
    tour = [start]
    visited[start] = True
    current = start
    for _ in range(n - 1):
        best_d = np.inf
        best_j = -1
        row = dist[current]
        for j in range(n):
            if not visited[j] and row[j] < best_d:
                best_d = row[j]
                best_j = j
        tour.append(best_j)
        visited[best_j] = True
        current = best_j
    return tour

def simulate_aco_for_heuristic(distance_matrix, heuristic, n_ants=20, n_iter=30, seed=42):
    n = distance_matrix.shape[0]
    np.random.seed(seed)
    
    h = heuristic.copy()
    h[~np.isfinite(h)] = 1e-9
    h[h <= 0] = 1e-9
    
    tau = np.ones((n, n))
    global_best_length = np.inf
    
    for it in range(n_iter):
        deposits = np.zeros((n, n))
        
        for ant in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            current = 0
            tour_length = 0.0
            tour = [0]
            
            for _ in range(n - 1):
                probs = tau[current, :] * h[current, :]
                probs[visited] = 0
                
                total = np.sum(probs)
                if total == 0:
                    unvisited_idx = np.where(~visited)[0]
                    next_node = np.random.choice(unvisited_idx)
                else:
                    probs = probs / total
                    next_node = np.random.choice(n, p=probs)
                
                tour_length += distance_matrix[current, next_node]
                tour.append(next_node)
                visited[next_node] = True
                current = next_node
                
            tour_length += distance_matrix[current, 0]
            tour.append(0)
            
            if tour_length < global_best_length:
                global_best_length = tour_length
            
            deposit_val = 1.0 / tour_length
            for k in range(len(tour) - 1):
                i = tour[k]
                j = tour[k+1]
                deposits[i, j] += deposit_val
        
        tau = tau * 0.9 + deposits
    
    return global_best_length

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = 325.0
    
    d = np.asarray(distance_matrix, dtype=float)
    if not np.all(np.isfinite(d)):
        d = np.nan_to_num(d, nan=1.0, posinf=1.0, neginf=1.0)
    
    n = d.shape[0]
    d_list = d.tolist()
    
    # Generate structural prior using 2-opt refined nearest neighbor tours
    nn_tour = _nearest_neighbor_tour(d_list, 0)
    _two_opt_improve(nn_tour, d_list)
    
    # Build edge frequency/quality matrix from the optimized tour
    edge_quality = np.zeros((n, n))
    tour_len = 0.0
    for k in range(len(nn_tour) - 1):
        i, j = nn_tour[k], nn_tour[k+1]
        edge_quality[i, j] = 1.0
        tour_len += d[i, j]
    
    # Combine inverse distance with structural prior
    # Use a blend: heuristic = (1/d)^alpha * (1 + beta * edge_quality)
    # Search over alpha and beta
    
    inv_dist = 1.0 / (d + 1e-9)
    
    best_avg_score = np.inf
    best_heuristic = inv_dist ** 2.0
    best_params = (2.0, 0.0)
    
    seeds = [0, 1, 2, 3, 4]
    
    # Search grid: alpha in [1, 5], beta in [0, 2]
    alphas = np.linspace(1.0, 5.0, 25)
    betas = np.linspace(0.0, 2.0, 11)
    
    for alpha in alphas:
        if time.time() > deadline:
            break
        inv_alpha = np.power(inv_dist, alpha)
        for beta in betas:
            if time.time() > deadline:
                break
            h_candidate = inv_alpha * (1.0 + beta * edge_quality)
            
            total_score = 0.0
            for seed in seeds:
                score = simulate_aco_for_heuristic(d, h_candidate, seed=seed)
                total_score += score
            
            avg_score = total_score / len(seeds)
            
            if avg_score < best_avg_score:
                best_avg_score = avg_score
                best_heuristic = h_candidate.copy()
                best_params = (alpha, beta)
    
    # Refine around best params if time remains
    remaining = deadline - time.time()
    if remaining > 30.0:
        best_alpha, best_beta = best_params
        local_alphas = np.linspace(max(0.5, best_alpha - 0.5), best_alpha + 0.5, 11)
        local_betas = np.linspace(max(0.0, best_beta - 0.5), best_beta + 0.5, 11)
        
        for alpha in local_alphas:
            if time.time() > deadline:
                break
            inv_alpha = np.power(inv_dist, alpha)
            for beta in local_betas:
                if time.time() > deadline:
                    break
                h_candidate = inv_alpha * (1.0 + beta * edge_quality)
                
                total_score = 0.0
                for seed in seeds:
                    score = simulate_aco_for_heuristic(d, h_candidate, seed=seed)
                    total_score += score
                
                avg_score = total_score / len(seeds)
                
                if avg_score < best_avg_score:
                    best_avg_score = avg_score
                    best_heuristic = h_candidate.copy()
                    best_params = (alpha, beta)
    
    return best_heuristic
