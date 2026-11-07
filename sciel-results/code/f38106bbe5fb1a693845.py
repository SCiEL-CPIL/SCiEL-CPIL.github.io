import numpy as np
import time

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
                # Vectorized probability calculation
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
    deadline = 300.0  # Safety margin
    
    # Candidate alphas for eta = (1/d)^alpha
    # We expect the optimum to be in [1, 10]
    alphas = np.linspace(1.0, 10.0, 10)
    
    best_alpha = 1.0
    best_avg_score = np.inf
    best_heuristic = None
    
    inv_dist = 1.0 / (distance_matrix + 1e-9)
    
    seeds = [0, 1, 2]
    
    for alpha in alphas:
        if time.time() > deadline:
            break
            
        h_candidate = np.power(inv_dist, alpha)
        
        total_score = 0.0
        for seed in seeds:
            score = simulate_aco_for_heuristic(distance_matrix, h_candidate, seed=seed)
            total_score += score
            
        avg_score = total_score / len(seeds)
        
        if avg_score < best_avg_score:
            best_avg_score = avg_score
            best_alpha = alpha
            best_heuristic = h_candidate
            
    if best_heuristic is None:
        best_heuristic = inv_dist
        
    return best_heuristic
