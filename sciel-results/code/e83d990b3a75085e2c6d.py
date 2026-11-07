import numpy as np
import time
import random

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 300.0  # Safety margin from 330s limit

def _two_opt(tour, dist_matrix, deadline):
    """Improve a tour using 2-opt heuristic. Returns the improved tour."""
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            if time.time() > deadline:
                return tour
            for j in range(i + 1, n):
                # Check if reversing the segment between i and j improves the tour
                # Edges involved: (tour[i-1], tour[i]) and (tour[j], tour[j+1])
                # vs (tour[i-1], tour[j]) and (tour[i], tour[j+1])
                a, b = tour[i - 1], tour[i]
                c, d = tour[j], tour[(j + 1) % n]
                
                if j == i + 1:
                    continue
                    
                current_cost = dist_matrix[a, b] + dist_matrix[c, d]
                new_cost = dist_matrix[a, c] + dist_matrix[b, d]
                
                if new_cost < current_cost - 1e-10:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
    return tour

def _run_aco(D, eta, n_ants=20, n_iter=30, seed=42):
    """Run ACO with given heuristic eta, return best tour length."""
    n = D.shape[0]
    rng = np.random.RandomState(seed)
    
    tau = np.ones((n, n))
    best_len = float('inf')
    
    for it in range(n_iter):
        all_paths = []
        all_lens = []
        
        for ant in range(n_ants):
            current = 0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            path = [0]
            tour_len = 0.0
            
            for step in range(1, n):
                available = np.where(~visited)[0]
                
                probs = tau[current, available] * eta[current, available]
                total = probs.sum()
                
                if total <= 0:
                    next_node = available[0]
                else:
                    probs = probs / total
                    r = rng.random()
                    cumsum = np.cumsum(probs)
                    next_node = available[np.searchsorted(cumsum, r)]
                
                tour_len += D[current, next_node]
                current = next_node
                visited[current] = True
                path.append(current)
            
            tour_len += D[current, 0]
            path.append(0)
            
            if tour_len < best_len:
                best_len = tour_len
            
            all_paths.append(path)
            all_lens.append(tour_len)
        
        tau *= 0.9
        for ant_idx in range(n_ants):
            l = all_lens[ant_idx]
            if l <= 0:
                continue
            reward = 1.0 / l
            path = all_paths[ant_idx]
            for k in range(len(path) - 1):
                i = path[k]
                j = path[k + 1]
                tau[i, j] += reward
    
    return best_len

def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix (larger is more attractive) of the same shape.
    
    Strategy:
    1. Generate a good baseline tour using Nearest Neighbor + 2-opt.
    2. Construct a heuristic matrix that strongly favors edges in this baseline tour
       while still allowing exploration based on inverse distance.
    3. Optionally, perform a quick local search on a 'boost' parameter if time permits,
       though a fixed strong boost is often sufficient and faster.
    """
    if time.time() > _DEADLINE - 10.0:
        # Emergency fallback: simple inverse distance
        d = np.asarray(distance_matrix, dtype=np.float64)
        eta = 1.0 / (d + 1e-9) ** 2
        eta[eta <= 0] = 1e-9
        return eta

    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    
    # 1. Nearest Neighbor Tour
    visited = [False] * n
    tour = [0]
    visited[0] = True
    current = 0
    
    for _ in range(n - 1):
        if time.time() > _DEADLINE - 20.0:
            break
        min_dist = float('inf')
        next_node = -1
        for j in range(n):
            if not visited[j]:
                if d[current, j] < min_dist:
                    min_dist = d[current, j]
                    next_node = j
        if next_node == -1:
            break
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
        
    if len(tour) < n:
        # Fallback
        eta = 1.0 / (d + 1e-9) ** 2
        eta[eta <= 0] = 1e-9
        return eta

    # 2. 2-Opt Improvement
    # Allocate specific time for 2-opt, e.g., 10 seconds
    two_opt_deadline = time.time() + 10.0
    tour = _two_opt(tour, d, two_opt_deadline)
    
    # 3. Construct Heuristic Matrix
    # Base heuristic: inverse distance squared
    base_heuristic = 1.0 / (d + 1e-9) ** 2
    
    # Topological boost: edges in the 2-opt improved tour get a large multiplier
    # We want to guide the ACO towards this known-good structure.
    # Let's try a fixed boost factor first. If time permits, we could search for it.
    # A high boost ensures the ACO converges quickly to a tour similar to the 2-opt one.
    # If the boost is too high, it might get stuck. If too low, it ignores the hint.
    # 10.0 seems reasonable based on inspiration 1.
    
    boost = 10.0
    
    # Create final heuristic
    eta = base_heuristic.copy()
    
    # Apply boost to tour edges
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        eta[u, v] = max(eta[u, v], boost)
        # Note: Pheromone is directional, but heuristic is usually symmetric in simple ACO.
        # However, the task says "pheromone[i, j] ... in travel direction only".
        # The heuristic is just a matrix. Usually symmetric.
        # If we make it asymmetric, eta[u,v] != eta[v,u], it might help or hurt.
        # Standard ACO uses symmetric heuristic. Let's keep it symmetric for stability,
        # unless we have evidence otherwise. The inspiration 1 made it symmetric.
        eta[v, u] = max(eta[v, u], boost)

    # Optional: Quick search for best boost if time permits
    # The current program did a search for 'p'. Here we have a fixed structure.
    # We can search for the optimal 'boost' value.
    # Let's see if we have time.
    # 20 ants * 30 iter * 50 nodes is fast.
    # Let's evaluate a few boost values.
    
    current_time = time.time()
    time_left = _DEADLINE - 30.0 - current_time # Leave 30s safety margin
    
    if time_left > 5.0:
        best_boost = boost
        best_score = _run_aco(d, eta, seed=42)
        
        # Try a few other boost values
        candidates = [1.0, 5.0, 10.0, 50.0, 100.0]
        # Remove current best_boost if in list, or just check all
        for b in candidates:
            if time.time() > _DEADLINE - 30.0:
                break
            test_eta = base_heuristic.copy()
            for i in range(n):
                u = tour[i]
                v = tour[(i + 1) % n]
                test_eta[u, v] = max(test_eta[u, v], b)
                test_eta[v, u] = max(test_eta[v, u], b)
            
            score = _run_aco(d, test_eta, seed=42)
            if score < best_score:
                best_score = score
                best_boost = b
                eta = test_eta
    
    # Ensure non-positive become 1e-9
    eta[eta <= 0] = 1e-9
    
    return eta
