import numpy as np
import time
from scipy.spatial import cKDTree

def _fast_aco_simulate(distance_matrix, heuristic, n_ants=5, n_iterations=10, rho=0.9):
    """
    Fast internal ACO simulation to evaluate heuristic quality.
    Correctly implements directed pheromone updates as per contract.
    """
    n = distance_matrix.shape[0]
    tau = np.ones((n, n))
    best_tour_len = float('inf')
    
    for it in range(n_iterations):
        tour_lengths = []
        for ant in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            current = 0
            tour_len = 0.0
            tour_path = [0]
            
            for _ in range(n - 1):
                probs = tau[current, :] * heuristic[current, :]
                probs[visited] = 0.0
                
                total_prob = probs.sum()
                if total_prob < 1e-10:
                    unvisited_indices = np.where(~visited)[0]
                    next_node = unvisited_indices[np.random.randint(len(unvisited_indices))]
                else:
                    probs = probs / total_prob
                    next_node = np.random.choice(n, p=probs)
                
                tour_len += distance_matrix[current, next_node]
                visited[next_node] = True
                tour_path.append(next_node)
                current = next_node
            
            tour_len += distance_matrix[current, 0]
            tour_lengths.append(tour_len)
            
            deposit = 1.0 / tour_len
            for i in range(n):
                u = tour_path[i]
                v = tour_path[(i + 1) % n]
                tau[u, v] += deposit
            
        tau *= rho
        
        if tour_lengths:
            current_best = min(tour_lengths)
            if current_best < best_tour_len:
                best_tour_len = current_best
                
    return best_tour_len

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 280.0
    
    n = distance_matrix.shape[0]
    
    # Base heuristic: inverse distance
    safe_dist = distance_matrix.copy()
    np.fill_diagonal(safe_dist, 1.0)
    safe_dist[safe_dist < 1e-9] = 1e-9
    
    base_heur = 1.0 / safe_dist
    
    # Build neighborhood structure using k-nearest neighbors
    # Extract points from distance matrix (assuming Euclidean in 2D)
    # We'll use a simple approach: for each node, find its k nearest neighbors
    k = 10  # number of nearest neighbors
    nn_mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        dists = distance_matrix[i].copy()
        dists[i] = np.inf
        nn_indices = np.argsort(dists)[:k]
        nn_mask[i, nn_indices] = True
    
    # Create neighborhood-enhanced base heuristic
    # Boost edges that are in the k-NN graph
    nn_boost = np.where(nn_mask, 1.5, 1.0)
    base_heur_nn = base_heur * nn_boost
    
    # Search for optimal beta with neighborhood enhancement
    # Use a wider range and more points
    betas = np.linspace(0.5, 5.0, 20)
    best_beta = 2.0
    best_score = float('inf')
    best_heuristic = base_heur.copy()
    
    # Coarse search
    for beta in betas:
        if time.time() > deadline - 30.0:
            break
            
        # Try both with and without NN boost
        h1 = base_heur ** beta
        h1[h1 <= 0] = 1e-9
        score1 = _fast_aco_simulate(distance_matrix, h1, n_ants=5, n_iterations=10)
        
        if score1 < best_score:
            best_score = score1
            best_beta = beta
            best_heuristic = h1.copy()
        
        h2 = base_heur_nn ** beta
        h2[h2 <= 0] = 1e-9
        score2 = _fast_aco_simulate(distance_matrix, h2, n_ants=5, n_iterations=10)
        
        if score2 < best_score:
            best_score = score2
            best_beta = beta
            best_heuristic = h2.copy()
    
    # Fine-grained search around best_beta
    if time.time() < deadline - 40.0:
        lower = max(0.5, best_beta - 0.3)
        upper = min(5.0, best_beta + 0.3)
        fine_betas = np.linspace(lower, upper, 15)
        
        for beta in fine_betas:
            if time.time() > deadline - 20.0:
                break
            
            h1 = base_heur ** beta
            h1[h1 <= 0] = 1e-9
            score1 = _fast_aco_simulate(distance_matrix, h1, n_ants=5, n_iterations=10)
            
            if score1 < best_score:
                best_score = score1
                best_beta = beta
                best_heuristic = h1.copy()
            
            h2 = base_heur_nn ** beta
            h2[h2 <= 0] = 1e-9
            score2 = _fast_aco_simulate(distance_matrix, h2, n_ants=5, n_iterations=10)
            
            if score2 < best_score:
                best_score = score2
                best_beta = beta
                best_heuristic = h2.copy()
    
    # Final polish: try a few more iterations with the best heuristic
    if time.time() < deadline - 10.0:
        h_final = best_heuristic.copy()
        h_final[h_final <= 0] = 1e-9
        score_final = _fast_aco_simulate(distance_matrix, h_final, n_ants=10, n_iterations=20)
        
        # If this is significantly better, we might want to adjust slightly
        if score_final < best_score * 0.95:
            # The heuristic is working well, keep it
            pass
        else:
            # Try a slightly different beta
            for beta_adj in [best_beta * 0.9, best_beta * 1.1]:
                if time.time() > deadline - 5.0:
                    break
                h_adj = base_heur ** beta_adj
                h_adj[h_adj <= 0] = 1e-9
                score_adj = _fast_aco_simulate(distance_matrix, h_adj, n_ants=5, n_iterations=10)
                
                if score_adj < best_score:
                    best_score = score_adj
                    best_heuristic = h_adj.copy()
    
    # Final sanity checks
    best_heuristic[best_heuristic <= 0] = 1e-9
    best_heuristic[~np.isfinite(best_heuristic)] = 1e-9
    
    return best_heuristic
