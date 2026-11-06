import numpy as np
import time

# Global variables to store optimized weights
OPTIMIZED_WEIGHTS = None
DEADLINE = time.time() + 140  # Safety margin for 150s limit

def _generate_test_instance(n=50, seed=42):
    rng = np.random.default_rng(seed)
    points = rng.random((n, 2))
    dm = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(points[i] - points[j])
            dm[i, j] = dist
            dm[j, i] = dist
    return dm

def _two_opt_improve(tour, dm):
    """
    Simple 2-opt improvement for the tour.
    tour: list of nodes [start, ..., end] (start==end for closed tour? No, usually path)
    Here we assume tour is a list of unique nodes visited in order.
    We want to minimize sum(d(tour[i], tour[i+1])) + d(tour[-1], tour[0]).
    """
    n = len(tour)
    if n <= 3:
        return tour
    
    # Convert to list for easier manipulation
    t = list(tour)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if j == i + 1:
                    continue
                # Calculate delta cost
                # Edges: (i-1, i), (i, i+1) ... (j-1, j), (j, j+1)
                # New edges: (i-1, j), (i, j+1)
                # Wait, standard 2-opt reverses segment [i..j].
                # Edges removed: (t[i-1], t[i]) and (t[j], t[j+1]) (if j < n-1) or (t[j], t[0])?
                # Since it's a closed tour, let's handle indices carefully.
                # Let's assume t is the open path, and we add the closing edge at the end for evaluation.
                # For 2-opt on open path with fixed start/end? No, TSP is closed.
                # The function select_next_node builds a path from 0 to 0.
                # The "tour" in _evaluate_tour_length is a sequence of nodes.
                
                # Let's just do a simple check on the closed loop.
                # Indices in closed loop: 0..n-1.
                # Edge (i, i+1) is t[i] to t[(i+1)%n].
                
                # Remove edges (t[i-1], t[i]) and (t[j], t[j+1])
                # Add edges (t[i-1], t[j]) and (t[i], t[j+1])
                # Note: if i=0, t[i-1] is t[n-1].
                
                idx_prev_i = (i - 1) % n
                idx_next_j = (j + 1) % n
                
                c_old = dm[t[idx_prev_i], t[i]] + dm[t[j], t[idx_next_j]]
                c_new = dm[t[idx_prev_i], t[j]] + dm[t[i], t[idx_next_j]]
                
                if c_new < c_old - 1e-9:
                    # Reverse segment t[i..j]
                    t[i:j+1] = reversed(t[i:j+1])
                    improved = True
    
    return t

def _evaluate_tour_length(weights, dm, k_dp=12):
    """
    Evaluate the tour length for a given distance matrix using the specified weights
    for the heuristic phase and DP for the tail.
    """
    n = dm.shape[0]
    unvisited = list(range(1, n))
    current = 0
    dest = 0
    
    total_cost = 0.0
    unvisited_set = set(unvisited)
    tour_nodes = [0]
    
    # Phase 1: Heuristic selection with 2-step lookahead
    while len(unvisited_set) > k_dp:
        unvisited_list = list(unvisited_set)
        if not unvisited_list:
            break
            
        best_score = float('inf')
        best_node = unvisited_list[0]
        
        # Precompute distances from current to all unvisited
        d_cur = dm[current, unvisited_list]
        
        # For each candidate, compute 2-step cost
        # To speed up, we can precompute min distance from each unvisited to other unvisited
        # But 2-step lookahead needs min_{v != u} (d(u,v) + d(v,dest))
        
        w1, w2 = weights # w1 for immediate, w2 for lookahead
        
        for idx, u in enumerate(unvisited_list):
            cost_immediate = d_cur[idx]
            
            # Lookahead: find best next node v from u
            # Exclude u from unvisited_list for the min calculation
            # d(u, v) for all v in unvisited_list, v != u
            d_u_to_others = dm[u, unvisited_list]
            # Set d_u_to_others[idx] to inf to exclude u
            d_u_to_others_copy = d_u_to_others.copy()
            d_u_to_others_copy[idx] = float('inf')
            
            min_d_u_v = np.min(d_u_to_others_copy)
            # Find argmin to get v
            idx_v = np.argmin(d_u_to_others_copy)
            v = unvisited_list[idx_v]
            
            # Cost of going u -> v -> dest
            cost_lookahead = min_d_u_v + dm[v, dest]
            
            score = w1 * cost_immediate + w2 * cost_lookahead
            
            if score < best_score:
                best_score = score
                best_node = u
        
        total_cost += dm[current, best_node]
        unvisited_set.remove(best_node)
        current = best_node
        tour_nodes.append(best_node)
        
    # Phase 2: DP for the rest
    if unvisited_set:
        unvisited_arr = np.array(list(unvisited_set))
        # Use Held-Karp for the tail
        # Re-implementing a fast DP here for evaluation
        m = len(unvisited_arr)
        if m <= 15:
            INF = float('inf')
            dp = np.full((1 << m, m), INF)
            parent = np.full((1 << m, m), -1, dtype=int)
            for i in range(m):
                dp[1 << i, i] = dm[current, unvisited_arr[i]]
                
            for mask in range(1 << m):
                for i in range(m):
                    if not (mask & (1 << i)):
                        continue
                    cost_i = dp[mask, i]
                    if cost_i == INF:
                        continue
                    for j in range(m):
                        if mask & (1 << j):
                            continue
                        new_mask = mask | (1 << j)
                        new_cost = cost_i + dm[unvisited_arr[i], unvisited_arr[j]]
                        if new_cost < dp[new_mask, j]:
                            dp[new_mask, j] = new_cost
                            parent[new_mask, j] = i
                            
            full_mask = (1 << m) - 1
            best_cost = INF
            best_last = -1
            for i in range(m):
                total = dp[full_mask, i] + dm[unvisited_arr[i], dest]
                if total < best_cost:
                    best_cost = total
                    best_last = i
                    
            # Backtrack to get the sequence
            path = []
            cur_mask = full_mask
            cur_i = best_last
            while cur_mask != 0:
                path.append(unvisited_arr[cur_i])
                if cur_mask & (cur_mask - 1) == 0:
                    break
                prev_i = parent[cur_mask, cur_i]
                cur_mask ^= (1 << cur_i)
                cur_i = prev_i
                
            path.reverse()
            for node in path:
                total_cost += dm[current, node]
                current = node
                tour_nodes.append(node)
        else:
            # Fallback greedy if DP too slow (shouldn't happen with k_dp=12)
            while unvisited_set:
                unv_list = list(unvisited_set)
                d = dm[current, unv_list]
                idx = np.argmin(d)
                next_node = unv_list[idx]
                total_cost += dm[current, next_node]
                unvisited_set.remove(next_node)
                current = next_node
                tour_nodes.append(next_node)
                
    total_cost += dm[current, dest]
    
    # Apply 2-opt to the tour_nodes to see if it helps
    # Note: 2-opt on 50 nodes is O(N^2 * iterations). Should be fast.
    improved_tour = _two_opt_improve(tour_nodes, dm)
    
    # Calculate length of improved tour
    imp_len = 0.0
    for i in range(len(improved_tour)):
        u = improved_tour[i]
        v = improved_tour[(i + 1) % len(improved_tour)]
        imp_len += dm[u, v]
        
    return imp_len

def _optimize_weights():
    global OPTIMIZED_WEIGHTS
    
    # Generate a few test instances
    instances = []
    for seed in range(10):
        instances.append(_generate_test_instance(50, seed))
        
    best_score = float('inf')
    best_weights = (1.0, 0.0) # Default: nearest neighbor
    
    rng = np.random.default_rng(0)
    
    # Search over w1, w2 where w1+w2=1
    # w1 in [0, 1], w2 = 1 - w1
    candidates = []
    for _ in range(100):
        w1 = rng.random()
        candidates.append((w1, 1.0 - w1))
        
    # Add structured points
    candidates.append((1.0, 0.0))
    candidates.append((0.0, 1.0))
    candidates.append((0.5, 0.5))
    candidates.append((0.8, 0.2))
    candidates.append((0.2, 0.8))
    
    for w in candidates:
        total_len = 0.0
        for dm in instances:
            total_len += _evaluate_tour_length(w, dm)
        avg_len = total_len / len(instances)
        if avg_len < best_score:
            best_score = avg_len
            best_weights = w
            
    OPTIMIZED_WEIGHTS = best_weights
    return best_weights

# Run optimization at load time, but with a time limit
if time.time() < 10:
    try:
        _optimize_weights()
    except Exception:
        OPTIMIZED_WEIGHTS = (1.0, 0.0)
else:
    OPTIMIZED_WEIGHTS = (1.0, 0.0)

def _held_karp_next(current_node, destination_node, unvisited_nodes, distance_matrix):
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(unv[0])
    
    INF = float('inf')
    parent = np.full((1 << n, n), -1, dtype=int)
    dp = np.full((1 << n, n), INF, dtype=float)
    for i in range(n):
        dp[1 << i, i] = distance_matrix[current_node, unv[i]]
        
    for mask in range(1 << n):
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            cost_i = dp[mask, i]
            if cost_i == INF:
                continue
            for j in range(n):
                if mask & (1 << j):
                    continue
                new_mask = mask | (1 << j)
                new_cost = cost_i + distance_matrix[unv[i], unv[j]]
                if new_cost < dp[new_mask, j]:
                    dp[new_mask, j] = new_cost
                    parent[new_mask, j] = i

    full_mask = (1 << n) - 1
    best_cost = INF
    best_last_i = -1
    for i in range(n):
        total = dp[full_mask, i] + distance_matrix[unv[i], destination_node]
        if total < best_cost:
            best_cost = total
            best_last_i = i
            
    # Backtrack to find first node
    cur_mask = full_mask
    cur_i = best_last_i
    first_i = cur_i
    while cur_mask != 0:
        if cur_mask & (cur_mask - 1) == 0:
            first_i = cur_i
            break
        prev_i = parent[cur_mask, cur_i]
        if prev_i == -1:
            break
        cur_i = prev_i
        cur_mask = cur_mask ^ (1 << cur_i)
        
    return int(unv[first_i])

def _select_heuristic(current_node, destination_node, unvisited_nodes, distance_matrix):
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(unv[0])
        
    w1, w2 = OPTIMIZED_WEIGHTS if OPTIMIZED_WEIGHTS else (1.0, 0.0)
    
    d_cur = distance_matrix[current_node, unv]
    
    best_score = float('inf')
    best_idx = 0
    
    # 2-step lookahead
    for idx, u in enumerate(unv):
        cost_immediate = d_cur[idx]
        
        d_u_to_others = distance_matrix[u, unv]
        d_u_to_others_copy = d_u_to_others.copy()
        d_u_to_others_copy[idx] = float('inf')
        
        min_d_u_v = np.min(d_u_to_others_copy)
        idx_v = np.argmin(d_u_to_others_copy)
        v = unv[idx_v]
        
        cost_lookahead = min_d_u_v + distance_matrix[v, destination_node]
        
        score = w1 * cost_immediate + w2 * cost_lookahead
        
        if score < best_score:
            best_score = score
            best_idx = idx
            
    return int(unv[best_idx])

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    if time.time() > DEADLINE:
        unv = np.asarray(unvisited_nodes)
        if unv.size == 0:
            return int(destination_node)
        d = distance_matrix[current_node, unv]
        return int(unv[np.argmin(d)])
        
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    
    if n <= 12:
        return _held_karp_next(current_node, destination_node, unv, distance_matrix)
    else:
        return _select_heuristic(current_node, destination_node, unv, distance_matrix)
