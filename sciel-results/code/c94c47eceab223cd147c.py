import numpy as np
import time

def _fast_aco_simulate(distance_matrix, heuristic, n_ants=5, n_iterations=10, rho=0.9, use_soft_min=False, soft_min_temp=0.1):
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

def _aco_soft_min_score(distance_matrix, heuristic, n_ants=5, n_iterations=10, rho=0.9, temp=0.1):
    """
    ACO simulation with soft-min surrogate: instead of tracking only the best tour,
    we track a smooth lower bound on the minimum tour length across all ants.
    This provides a smoother optimization landscape for beta search.
    """
    n = distance_matrix.shape[0]
    tau = np.ones((n, n))
    best_soft_min = float('inf')
    
    for it in range(n_iterations):
        tour_lengths = np.zeros(n_ants)
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
            tour_lengths[ant] = tour_len
            
            deposit = 1.0 / tour_len
            for i in range(n):
                u = tour_path[i]
                v = tour_path[(i + 1) % n]
                tau[u, v] += deposit
        
        tau *= rho
        
        # Soft-min surrogate: -temp * log(sum(exp(-tour_lengths/temp)))
        # This is a smooth approximation of min(tour_lengths)
        shifted = tour_lengths / temp
        max_shifted = np.max(shifted)
        log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
        soft_min = temp * log_sum_exp
        
        # Track the best (lowest) soft-min value
        if soft_min < best_soft_min:
            best_soft_min = soft_min
        
        # Also track actual min for comparison
        actual_min = np.min(tour_lengths)
        if actual_min < best_soft_min:
            best_soft_min = actual_min
            
    return best_soft_min

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 290.0  # Leave 40s safety margin
    
    n = distance_matrix.shape[0]
    
    safe_dist = distance_matrix.copy()
    np.fill_diagonal(safe_dist, 1.0)
    safe_dist[safe_dist < 1e-9] = 1e-9
    
    base_heur = 1.0 / safe_dist
    
    # Search for optimal beta with wider range including lower values
    betas = np.linspace(0.3, 6.0, 25)
    best_beta = 2.0
    best_score = float('inf')
    
    for beta in betas:
        if time.time() > deadline - 20.0:
            break
            
        h = base_heur ** beta
        h[h <= 0] = 1e-9
        
        score = _fast_aco_simulate(distance_matrix, h, n_ants=5, n_iterations=10)
        
        if score < best_score:
            best_score = score
            best_beta = beta
            
    # Fine-grained search around best_beta
    if time.time() < deadline - 30.0:
        lower = max(0.3, best_beta - 0.5)
        upper = min(6.0, best_beta + 0.5)
        fine_betas = np.linspace(lower, upper, 15)
        
        for beta in fine_betas:
            if time.time() > deadline - 20.0:
                break
            h = base_heur ** beta
            h[h <= 0] = 1e-9
            score = _fast_aco_simulate(distance_matrix, h, n_ants=5, n_iterations=10)
            
            if score < best_score:
                best_score = score
                best_beta = beta
                
    # Try adding local neighborhood enhancement
    if time.time() < deadline - 35.0:
        # Create a k-nearest-neighbor based enhancement
        k = 5
        dist_flat = safe_dist.copy()
        np.fill_diagonal(dist_flat, np.inf)
        
        knn_mask = np.zeros((n, n), dtype=bool)
        for i in range(n):
            k_indices = np.argsort(dist_flat[i])[:k]
            knn_mask[i, k_indices] = True
        
        # Blend: boost edges to nearest neighbors
        knn_boost = np.where(knn_mask, 2.0, 1.0)
        
        # Test a few blending factors
        for blend in [0.3, 0.5, 0.7]:
            if time.time() > deadline - 20.0:
                break
            
            h = (base_heur ** best_beta) * knn_boost
            h = h / h.max() * base_heur.max()
            h[h <= 0] = 1e-9
            
            score = _fast_aco_simulate(distance_matrix, h, n_ants=5, n_iterations=10)
            
            if score < best_score:
                best_score = score
                # Use this enhanced version
                final_heuristic = h.copy()
                final_heuristic[final_heuristic <= 0] = 1e-9
                final_heuristic[~np.isfinite(final_heuristic)] = 1e-9
                return final_heuristic
                
    # Try soft-min guided search for better beta
    if time.time() < deadline - 40.0:
        # Use soft-min surrogate to find a smoother optimum
        soft_betas = np.linspace(max(0.5, best_beta - 1.0), min(5.0, best_beta + 1.0), 10)
        best_soft_beta = best_beta
        best_soft_score = float('inf')
        
        for beta in soft_betas:
            if time.time() > deadline - 25.0:
                break
            h = base_heur ** beta
            h[h <= 0] = 1e-9
            score = _aco_soft_min_score(distance_matrix, h, n_ants=5, n_iterations=8, temp=0.05)
            
            if score < best_soft_score:
                best_soft_score = score
                best_soft_beta = beta
        
        # Verify with hard-min simulation
        if best_soft_beta != best_beta:
            h_soft = base_heur ** best_soft_beta
            h_soft[h_soft <= 0] = 1e-9
            score_soft = _fast_aco_simulate(distance_matrix, h_soft, n_ants=5, n_iterations=10)
            
            if score_soft < best_score:
                best_score = score_soft
                best_beta = best_soft_beta
                
    # Final polish: run more iterations on the best candidate
    if time.time() < deadline - 15.0:
        h_final = base_heur ** best_beta
        h_final[h_final <= 0] = 1e-9
        
        # Try a few variants with slight perturbations
        candidates = []
        candidates.append((best_beta, h_final.copy()))
        
        # Slightly different betas
        for dbeta in [-0.2, 0.2]:
            new_beta = best_beta + dbeta
            if new_beta > 0.1:
                h_var = base_heur ** new_beta
                h_var[h_var <= 0] = 1e-9
                candidates.append((new_beta, h_var))
        
        # KNN-enhanced version
        k = 5
        dist_flat = safe_dist.copy()
        np.fill_diagonal(dist_flat, np.inf)
        knn_mask = np.zeros((n, n), dtype=bool)
        for i in range(n):
            k_indices = np.argsort(dist_flat[i])[:k]
            knn_mask[i, k_indices] = True
        knn_boost = np.where(knn_mask, 1.5, 1.0)
        h_knn = (base_heur ** best_beta) * knn_boost
        h_knn = h_knn / h_knn.max() * base_heur.max()
        h_knn[h_knn <= 0] = 1e-9
        candidates.append((best_beta, h_knn))
        
        best_candidate = None
        best_candidate_score = float('inf')
        
        for beta_c, h_c in candidates:
            if time.time() > deadline - 10.0:
                break
            score_c = _fast_aco_simulate(distance_matrix, h_c, n_ants=5, n_iterations=12)
            if score_c < best_candidate_score:
                best_candidate_score = score_c
                best_candidate = h_c.copy()
        
        if best_candidate is not None:
            best_candidate[best_candidate <= 0] = 1e-9
            best_candidate[~np.isfinite(best_candidate)] = 1e-9
            return best_candidate
    
    # Construct final heuristic
    final_heuristic = base_heur ** best_beta
    final_heuristic[final_heuristic <= 0] = 1e-9
    final_heuristic[~np.isfinite(final_heuristic)] = 1e-9
    
    return final_heuristic
