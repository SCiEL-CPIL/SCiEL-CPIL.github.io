import numpy as np
import time

def _run_aco_fast(prizes, distance_matrix, max_len, heuristic, num_ants=20, num_iters=30, seed=0):
    """
    Fast ACO implementation for evaluation.
    """
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

def _run_greedy_proxy(prizes, distance_matrix, max_len, heuristic, num_runs=5):
    """
    Fast proxy: Simulates greedy ants with slight noise to estimate robustness.
    """
    n = len(prizes)
    d = distance_matrix
    d_j0 = d[0, :]
    cost = d + d_j0[None, :]
    
    best_prize = 0.0
    
    for _ in range(num_runs):
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
            
            # Add small noise to heuristic to simulate different ant choices
            h_vals = heuristic[cur, valid_nodes] * (1 + 0.05 * np.random.randn(len(valid_nodes)))
            h_vals[h_vals <= 0] = 1e-9
            
            # Pick max
            next_node = valid_nodes[np.argmax(h_vals)]
            
            visited[next_node] = True
            length += d[cur, next_node]
            prize += prizes[next_node]
            cur = next_node
        
        if prize > best_prize:
            best_prize = prize
            
    return best_prize

def _build_density_heuristic(prizes, distance_matrix, k, gamma):
    """
    Builds heuristic based on local density of value.
    H[i, j] = prize[j] / (local_density[j])^gamma
    local_density[j] = average distance from j to its k nearest neighbors (weighted by prize? or just distance?)
    The prompt says: "prize-weighted average distance to the top-k nearest neighbors"
    So density_j = sum_{m in top_k(j)} prize[m] * d[j, m] / sum_{m in top_k(j)} prize[m]
    If sum prize is 0, density is large (bad) or 0? 
    "rewards moves into high-value clusters where multiple high-prize nodes are geographically close"
    So low density (close) + high prize = good.
    If neighbors have high prize and are close, density is low.
    If neighbors are far, density is high.
    So 1/density^gamma rewards close clusters.
    """
    n = len(prizes)
    d = distance_matrix
    p = prizes
    
    # Compute distances
    # For each node j, find k nearest neighbors
    # Sort distances for each row
    sorted_d = np.sort(d, axis=1)
    
    # k nearest neighbors (excluding self? d[j,j]=0 usually. If k includes self, dist 0. 
    # Usually "neighbors" implies others. Let's take k nearest *other* nodes.
    # So take indices 1 to k.
    # If n is small, handle k >= n-1.
    k_eff = min(k, n - 1)
    
    # Get the distances to the k nearest neighbors
    # Note: sorted_d[:, 0] is 0 (self).
    # We want the average distance to the k nearest neighbors.
    # The prompt says "prize-weighted average distance".
    
    # We need the indices of the k nearest neighbors to get their prizes.
    # This is a bit expensive if done naively for each j, but n=50 is small.
    
    local_density = np.zeros(n)
    
    for j in range(n):
        # Get distances from j to all others
        dists = d[j, :].copy()
        dists[j] = np.inf # Exclude self
        
        # Get k nearest
        if k_eff > 0:
            # Indices of k smallest
            idx = np.argpartition(dists, k_eff)[:k_eff]
            neighbor_dists = dists[idx]
            neighbor_prizes = p[idx]
            
            # Prize-weighted average distance
            sum_p = neighbor_prizes.sum()
            if sum_p > 1e-9:
                local_density[j] = np.sum(neighbor_prizes * neighbor_dists) / sum_p
            else:
                # No prize in neighbors? Density should be high (unattractive) or 0?
                # If no neighbors have prize, it's not a "high-value cluster".
                # Set to a large value to make 1/density small.
                local_density[j] = 1.0 # Arbitrary, will be divided out if prize[j] is 0
        else:
            local_density[j] = 1.0
    
    # Heuristic H[i, j] = prize[j] / (local_density[j] + eps)^gamma
    # This is independent of i. So H is a column vector repeated.
    # This is a structural change: the heuristic depends only on the destination node's local environment.
    
    eps = 1e-9
    density_term = (local_density + eps) ** gamma
    
    # H[i, j] = p[j] / density_term[j]
    h_col = p / density_term
    
    # Broadcast to matrix
    heuristic = np.tile(h_col, (n, 1))
    
    # Ensure finite and positive
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic

def _evaluate_params(prizes, distance_matrix, max_len, k, gamma, seed=0):
    """
    Constructs heuristic from params and evaluates with Greedy Proxy (fast).
    """
    n = len(prizes)
    
    try:
        heuristic = _build_density_heuristic(prizes, distance_matrix, k, gamma)
        
        if heuristic.shape != (n, n):
            return 0.0
        if not np.all(np.isfinite(heuristic)):
            return 0.0
            
        # Use greedy proxy for speed
        score = _run_greedy_proxy(prizes, distance_matrix, max_len, heuristic, num_runs=10)
        return score
    except Exception:
        return 0.0

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    deadline = start_time + 290.0 # 40s safety margin
    
    n = len(prizes)
    d = distance_matrix
    p = prizes
    
    best_k = 5
    best_gamma = 1.0
    best_score = -1
    
    # Phase 1: Coarse Grid Search over k and gamma
    # k: 1 to 20
    # gamma: 0.5 to 3.0
    ks = np.arange(1, 21)
    gammas = np.linspace(0.5, 3.0, 10)
    
    for k in ks:
        for g in gammas:
            if time.time() > deadline - 60.0:
                break
            score = _evaluate_params(prizes, distance_matrix, max_len, k, g, seed=0)
            if score > best_score:
                best_score = score
                best_k = k
                best_gamma = g
        
    # Phase 2: Local Refinement
    # Refine k (integer) and gamma (continuous)
    
    current_k = int(best_k)
    current_g = best_gamma
    current_score = best_score
    
    # Check neighbors for k
    for delta_k in [-2, -1, 1, 2]:
        nk = current_k + delta_k
        if nk < 1 or nk > 40:
            continue
        score = _evaluate_params(prizes, distance_matrix, max_len, nk, current_g, seed=0)
        if score > current_score:
            current_score = score
            current_k = nk
    
    # Refine gamma with smaller steps
    for _ in range(5):
        improved = False
        for dg in [-0.2, -0.1, -0.05, 0.05, 0.1, 0.2]:
            ng = current_g + dg
            if ng < 0.1 or ng > 5.0:
                continue
            score = _evaluate_params(prizes, distance_matrix, max_len, current_k, ng, seed=0)
            if score > current_score:
                current_score = score
                current_g = ng
                improved = True
        
        if not improved:
            break
    
    # Final Evaluation with ACO to confirm? 
    # The greedy proxy might differ from ACO. 
    # Let's do a quick ACO check on the best candidate and a few neighbors to ensure robustness.
    # But time is tight. The proxy is usually a good indicator.
    # Let's just build the final heuristic.
    
    final_heuristic = _build_density_heuristic(prizes, distance_matrix, current_k, current_g)
    
    # Ensure valid output
    if final_heuristic.shape != (n, n) or not np.all(np.isfinite(final_heuristic)):
        final_heuristic = np.ones((n, n))
        final_heuristic[0, :] = 1e-9
        final_heuristic[:, 0] = 1e-9
    
    final_heuristic[final_heuristic <= 0] = 1e-9
    
    return final_heuristic
