import numpy as np
import time

def _run_aco_fast(prizes, distance_matrix, max_len, heuristic, num_ants=20, num_iters=30, seed=0):
    n = len(prizes)
    rng = np.random.default_rng(seed)
    
    eta = heuristic.copy()
    eta[eta <= 0] = 1e-9
    
    tau = np.ones((n, n))
    
    d = distance_matrix
    d_j0 = d[0, :] 
    cost = d + d_j0[None, :]
    
    best_global = 0.0
    
    for _ in range(num_iters):
        best_iter = 0.0
        
        for _ in range(num_ants):
            cur = 0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            length = 0.0
            prize = 0.0
            
            while True:
                remaining = max_len - length
                valid_mask = (~visited) & (cost[cur, :] <= remaining)
                valid_nodes = np.where(valid_mask)[0]
                
                if len(valid_nodes) == 0:
                    break
                
                w = tau[cur, valid_nodes] * eta[cur, valid_nodes]
                sum_w = w.sum()
                if sum_w < 1e-10:
                    break
                p = w / sum_w
                
                next_node = rng.choice(valid_nodes, p=p)
                
                visited[next_node] = True
                length += d[cur, next_node]
                prize += prizes[next_node]
                cur = next_node
            
            if prize > best_iter:
                best_iter = prize
        
        if best_iter > best_global:
            best_global = best_iter
            
        tau = 0.9 * tau + (best_iter / 10.0)
        
    return best_global

def _run_aco_quick(prizes, distance_matrix, max_len, heuristic, num_ants=10, num_iters=15, seed=0):
    """Faster version for initial candidate screening"""
    n = len(prizes)
    rng = np.random.default_rng(seed)
    
    eta = heuristic.copy()
    eta[eta <= 0] = 1e-9
    
    tau = np.ones((n, n))
    
    d = distance_matrix
    d_j0 = d[0, :] 
    cost = d + d_j0[None, :]
    
    best_global = 0.0
    
    for _ in range(num_iters):
        best_iter = 0.0
        
        for _ in range(num_ants):
            cur = 0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            length = 0.0
            prize = 0.0
            
            while True:
                remaining = max_len - length
                valid_mask = (~visited) & (cost[cur, :] <= remaining)
                valid_nodes = np.where(valid_mask)[0]
                
                if len(valid_nodes) == 0:
                    break
                
                w = tau[cur, valid_nodes] * eta[cur, valid_nodes]
                sum_w = w.sum()
                if sum_w < 1e-10:
                    break
                p = w / sum_w
                
                next_node = rng.choice(valid_nodes, p=p)
                
                visited[next_node] = True
                length += d[cur, next_node]
                prize += prizes[next_node]
                cur = next_node
            
            if prize > best_iter:
                best_iter = prize
        
        if best_iter > best_global:
            best_global = best_iter
            
        tau = 0.9 * tau + (best_iter / 10.0)
        
    return best_global

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    # Hard deadline of 95s per function call to be safe within 300s total for 3 instances
    deadline = start_time + 95.0
    
    n = len(prizes)
    d = distance_matrix
    p = prizes
    eps = 1e-9
    
    d_j0 = d[0, :]
    
    # Generate a diverse set of candidates
    candidates = []
    
    # 1. Standard inverse distance
    candidates.append(p[None, :] / (d + eps))
    
    # 2. Return-aware (prize / (distance to node + distance back to start))
    candidates.append(p[None, :] / (d + d_j0[None, :] + eps))
    
    # 3. Prize powers with return-aware denominator
    for alpha in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
        p_pow = np.where(p > eps, p**alpha, 0)
        candidates.append(p_pow[None, :] / (d + d_j0[None, :] + eps))
        
    # 4. Distance powers with return-aware denominator
    for beta in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        denom = (d + d_j0[None, :] + eps)**beta
        candidates.append(p[None, :] / denom)
        
    # 5. Hybrid approaches
    candidates.append(p[None, :] / (d + eps) * (1 / (d_j0[None, :] + eps)))
    candidates.append(p[None, :] / (np.maximum(d, d_j0[None, :]) + eps))
    candidates.append(p[None, :] / (np.minimum(d, d_j0[None, :]) + eps))
    
    # 6. Normalized prize
    p_max = np.max(p)
    if p_max > 0:
        p_norm = p / p_max
        candidates.append(p_norm[None, :] / (d + d_j0[None, :] + eps))
        
    # 7. Logarithmic prize
    p_log = np.log(p + 1e-6)
    candidates.append(p_log[None, :] / (d + d_j0[None, :] + eps))
    
    # 8. Inverse distance only (no prize weighting)
    candidates.append(1.0 / (d + d_j0[None, :] + eps))
    
    # 9. Prize-squared / distance
    candidates.append((p**2)[None, :] / (d + d_j0[None, :] + eps))
    
    # 10. Prize / (distance * return_distance)
    candidates.append(p[None, :] / (d * d_j0[None, :] + eps))
    
    # 11. Prize / (distance + 2*return_distance)
    candidates.append(p[None, :] / (d + 2*d_j0[None, :] + eps))
    
    # 12. Prize / (2*distance + return_distance)
    candidates.append(p[None, :] / (2*d + d_j0[None, :] + eps))
    
    # 13. Prize / sqrt(distance + return_distance)
    candidates.append(p[None, :] / np.sqrt(d + d_j0[None, :] + eps))
    
    # 14. Prize / (distance + return_distance)^0.75
    candidates.append(p[None, :] / (d + d_j0[None, :] + eps)**0.75)
    
    # 15. Prize^0.5 / (distance + return_distance)
    candidates.append(np.sqrt(np.maximum(p, 0))[None, :] / (d + d_j0[None, :] + eps))
    
    # 16. Prize / (distance + return_distance + 0.1)
    candidates.append(p[None, :] / (d + d_j0[None, :] + 0.1))
    
    # 17. Prize / (distance + return_distance)^1.25
    candidates.append(p[None, :] / (d + d_j0[None, :] + eps)**1.25)
    
    # 18. Prize^1.25 / (distance + return_distance)
    candidates.append((p**1.25)[None, :] / (d + d_j0[None, :] + eps))
    
    # 19. Prize / (distance^0.5 + return_distance^0.5)
    candidates.append(p[None, :] / (np.sqrt(d + eps) + np.sqrt(d_j0[None, :] + eps)))
    
    # 20. Prize / (distance^0.75 + return_distance^0.75)
    candidates.append(p[None, :] / (np.power(d + eps, 0.75) + np.power(d_j0[None, :] + eps, 0.75)))
    
    # Validate and filter candidates
    valid_candidates = []
    for cand in candidates:
        if cand.shape == (n, n) and np.all(np.isfinite(cand)):
            valid_candidates.append(cand)
    
    if not valid_candidates:
        # Fallback to a simple heuristic
        return np.ones((n, n))
    
    # Phase 1: Quick screening with fewer ants/iterations
    quick_results = []
    for i, cand in enumerate(valid_candidates):
        if time.time() > deadline - 30:
            break
        score = _run_aco_quick(p, d, max_len, cand, num_ants=10, num_iters=15, seed=0)
        quick_results.append((score, i))
    
    # Sort by quick score and keep top candidates
    quick_results.sort(reverse=True)
    top_candidates = [valid_candidates[i] for _, i in quick_results[:8]]
    
    # Phase 2: Full evaluation of top candidates
    best_score = -1
    best_heuristic = top_candidates[0] if top_candidates else valid_candidates[0]
    
    for cand in top_candidates:
        if time.time() > deadline - 15:
            break
        score = _run_aco_fast(p, d, max_len, cand, num_ants=20, num_iters=30, seed=0)
        if score > best_score:
            best_score = score
            best_heuristic = cand
    
    # Phase 3: Fine-tune around the best candidate if time permits
    if time.time() < deadline - 10:
        # Try slight variations of the best heuristic
        best_base = best_heuristic.copy()
        best_base[best_base <= 0] = 1e-9
        
        # Scale variations
        for scale in [0.5, 0.75, 1.25, 1.5, 2.0]:
            if time.time() > deadline - 5:
                break
            cand = best_base * scale
            if cand.shape == (n, n) and np.all(np.isfinite(cand)):
                score = _run_aco_fast(p, d, max_len, cand, num_ants=20, num_iters=30, seed=0)
                if score > best_score:
                    best_score = score
                    best_heuristic = cand
        
        # Add small random perturbations
        rng = np.random.default_rng(42)
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            noise = rng.normal(0, 0.1, size=(n, n))
            cand = best_base + noise
            cand = np.maximum(cand, 1e-9)
            if cand.shape == (n, n) and np.all(np.isfinite(cand)):
                score = _run_aco_fast(p, d, max_len, cand, num_ants=20, num_iters=30, seed=0)
                if score > best_score:
                    best_score = score
                    best_heuristic = cand
    
    # Final cleanup
    best_heuristic = best_heuristic.copy()
    best_heuristic[best_heuristic <= 0] = 1e-9
    
    return best_heuristic
