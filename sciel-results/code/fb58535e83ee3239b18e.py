import numpy as np
import time

_START_TIME = None

def _get_elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _deadline_guard(margin=5.0):
    """Return True if we're getting close to the 150s limit."""
    return _get_elapsed() > (150.0 - margin)

def _initial_tour_construction(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Construct a fast initial tour using a potential-field heuristic.
    Score for node j: 
    1. Distance from current (local attraction)
    2. Distance to destination (global pressure)
    3. Connectivity to remaining unvisited (avoid isolating nodes)
    """
    unv = np.asarray(unvisited_nodes)
    n_total = len(unvisited_nodes)
    
    if n_total == 0:
        return []
    
    # State: list of visited nodes in order
    tour = [current_node]
    remaining = set(unvisited_nodes)
    current = current_node
    
    # Precompute destination distances for all nodes to speed up scoring
    # We assume distance_matrix is symmetric and includes all nodes
    dest = destination_node
    
    while remaining:
        if _deadline_guard(margin=10.0):
            # Emergency exit: just append nearest neighbor to finish
            # This is a fallback to ensure we return something valid quickly
            rem_list = list(remaining)
            if not rem_list:
                break
            d = distance_matrix[current, rem_list]
            next_node = rem_list[np.argmin(d)]
            tour.append(next_node)
            remaining.remove(next_node)
            current = next_node
            continue
        
        rem_list = list(remaining)
        n_rem = len(rem_list)
        
        if n_rem == 1:
            tour.append(rem_list[0])
            break
            
        # Vectorized scoring
        # 1. Local distance from current
        d_local = distance_matrix[current, rem_list]
        
        # 2. Distance to destination (we want to avoid going too far from dest too early, 
        #    but also not get stuck far away. A soft penalty for being far from dest 
        #    helps guide the tour back.)
        d_dest = distance_matrix[rem_list, dest]
        
        # 3. Connectivity: Average distance from candidate to other remaining nodes.
        #    We prefer nodes that are "central" to the remaining cluster to avoid long tails.
        #    Approximation: Min distance to any other remaining node (nearest neighbor in remaining)
        #    This is O(N^2) per step if done naively. For N=50, 50*50*50 = 125k ops, which is fine.
        #    Let's compute min distance to other remaining nodes.
        
        # Optimization: We can precompute the min distance to other remaining nodes 
        # but since 'remaining' changes, we have to do it every step.
        # For N=50, this is fast enough.
        
        min_d_to_other = np.zeros(n_rem)
        for i, node in enumerate(rem_list):
            # Distances to all other remaining nodes
            # Exclude self
            other_indices = [j for j in range(n_rem) if j != i]
            if not other_indices:
                min_d_to_other[i] = 0
            else:
                min_d_to_other[i] = np.min(distance_matrix[node, rem_list[other_indices]])
        
        # Normalize scores to [0, 1] to combine them
        def normalize(x):
            mn, mx = x.min(), x.max()
            if mx - mn < 1e-9:
                return np.zeros_like(x)
            return (x - mn) / (mx - mn)
        
        s_local = normalize(d_local)
        s_dest = normalize(d_dest)
        s_conn = normalize(min_d_to_other)
        
        # Weighted combination:
        # 0.5 * Local + 0.3 * Destination + 0.2 * Connectivity
        # Tuning weights is part of the "heuristic" but we keep it structural.
        score = 0.5 * s_local + 0.3 * s_dest + 0.2 * s_conn
        
        next_idx = np.argmin(score)
        next_node = rem_list[next_idx]
        
        tour.append(next_node)
        remaining.remove(next_node)
        current = next_node
        
    # Close the tour back to destination? 
    # The task says "picks the next city while a traveling-salesman tour is built step by step".
    # The interface returns the *next* node. 
    # Wait, the interface is `select_next_node`. It is called sequentially.
    # My strategy of building the *whole* tour at once is incompatible with the sequential interface 
    # UNLESS I simulate the sequential calls but use the global view.
    
    # CRITICAL REALIZATION: 
    # The function `select_next_node` is called N times.
    # If I build the whole tour in the first call, I must store it and return nodes one by one.
    # But `unvisited_nodes` changes in each call.
    # I can cache the full tour generated in the first call (or first few calls) and just pop from it.
    
    return tour

# Global cache for the planned tour
_planned_tour_cache = {}
_tour_lock = False

def _compute_tour_length(tour, distance_matrix):
    """Compute total length of a closed tour."""
    if len(tour) < 2:
        return 0.0
    n = len(tour)
    total = 0.0
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        total += distance_matrix[u, v]
    return total

def _two_opt(tour, distance_matrix, time_limit=10.0):
    """
    Perform 2-Opt improvement on the tour.
    tour: list of nodes (excluding the starting node? No, usually includes all).
    Here tour is the sequence of nodes to visit. 
    The tour is closed: start -> tour[0] -> ... -> tour[-1] -> dest -> start?
    Actually, the problem is a path from current to destination visiting all unvisited.
    So the "tour" is a path: current -> n1 -> n2 -> ... -> nk -> destination.
    Length = d(current, n1) + d(n1, n2) + ... + d(nk, destination).
    
    2-Opt on a path:
    We can reverse a segment of the path.
    If we reverse segment [i, j], the edges (i-1, i) and (j, j+1) are replaced by (i-1, j) and (i, j+1).
    (Indices in the sequence including start and end).
    """
    # Extend tour with start and end for easier indexing
    # sequence = [current, n1, ..., nk, destination]
    # But we don't have 'current' and 'destination' in the 'tour' list passed to this helper 
    # unless we pass them. 
    # Let's assume 'tour' passed here is the full sequence [start, n1, ..., nk, end].
    
    n = len(tour)
    if n < 4:
        return tour
    
    best_tour = tour
    best_len = _compute_tour_length(best_tour, distance_matrix)
    
    # We only improve the internal structure. Start and End are fixed.
    # i ranges from 1 to n-3 (index of first node of segment)
    # j ranges from i+1 to n-2 (index of last node of segment)
    
    # To speed up, we can check improvement delta.
    # Delta = d(i-1, j) + d(i, j+1) - d(i-1, i) - d(j, j+1)
    
    improved = True
    while improved:
        improved = False
        # Check time
        if time.time() > time_limit:
            break
            
        for i in range(1, n - 2):
            if time.time() > time_limit:
                break
            for j in range(i + 1, n - 1):
                # Calculate gain
                # Old edges: (i-1, i) and (j, j+1)
                # New edges: (i-1, j) and (i, j+1)
                d_old = distance_matrix[tour[i-1], tour[i]] + distance_matrix[tour[j], tour[j+1]]
                d_new = distance_matrix[tour[i-1], tour[j]] + distance_matrix[tour[i], tour[j+1]]
                
                if d_new < d_old - 1e-9:
                    # Perform reversal
                    best_tour[i:j+1] = reversed(best_tour[i:j+1])
                    improved = True
                    # We could continue or restart, but continuing is fine for now
                    # Update best_len if we track it, but we recompute at end or track delta
                    best_len += d_new - d_old
    
    return best_tour

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _planned_tour_cache, _tour_lock
    
    unv = np.asarray(unvisited_nodes)
    n_unvisited = len(unvisited_nodes)
    
    # If we are at the end, return destination
    if n_unvisited == 0:
        return int(destination_node)
    
    # If we have a cached plan for this specific state (current, dest, set of unvisited), use it
    # Key: (current_node, destination_node, tuple(sorted(unvisited_nodes)))
    # Sorting is O(N log N) but N is small.
    key = (int(current_node), int(destination_node), tuple(sorted(unvisited_nodes)))
    
    if key in _planned_tour_cache:
        planned = _planned_tour_cache[key]
        # Return the next node in the plan
        next_node = planned[0]
        # Update cache: remove the first node
        _planned_tour_cache[key] = planned[1:]
        return int(next_node)
    
    # If no plan, generate one
    # We need to generate a plan for the *current* state.
    # The plan should be a list of nodes to visit, starting from the one we return now.
    
    # 1. Construct initial tour
    # The initial tour construction uses the current node and destination.
    # It returns a list of nodes [n1, n2, ..., nk].
    
    # We need to ensure the "tour" for 2-Opt includes the start and end.
    # Sequence: [current_node, n1, n2, ..., nk, destination_node]
    
    initial_nodes = _initial_tour_construction(current_node, destination_node, unvisited_nodes, distance_matrix)
    
    # Build full sequence for 2-Opt
    full_seq = [int(current_node)] + [int(x) for x in initial_nodes] + [int(destination_node)]
    
    # 2. Refine with 2-Opt
    # Allocate time for 2-Opt. 
    # We have 150s total. 
    # We are in the first call (or a new branch). 
    # Let's give 2-Opt a generous budget, say 50s, but check global deadline.
    # Since this is called sequentially, we only do this heavy work once per unique state.
    # However, the state changes every call (unvisited shrinks).
    # So we will do 2-Opt on the remaining sub-problem every time?
    # That would be O(N * 2-Opt). 
    # 2-Opt on N=50 is fast. 
    # But doing it 50 times (once per step) might be too slow if each takes 10s.
    # Better strategy: Only do heavy 2-Opt on the *first* call (full problem) or when N is large.
    # For subsequent calls, the "plan" from the previous step's 2-Opt is already optimized 
    # for the remaining nodes? 
    # Actually, if we 2-Opt the full tour [S, n1...nk, D], and then remove n1, 
    # the remaining path [n2...nk, D] starting from n1 (which is now current) is NOT necessarily optimal 
    # for the new current node. 
    # However, it is a very good starting point.
    
    # Let's use a time budget strategy:
    # If N > 20, use 2-Opt with a time limit proportional to N.
    # Otherwise, use exact DP (Held-Karp) which is fast for small N.
    
    if n_unvisited <= 15:
        # Use Held-Karp for exact solution
        # Re-implementing HK here or calling a helper
        next_node = _held_karp_next(current_node, destination_node, unvisited_nodes, distance_matrix)
        # We don't cache HK results in the same way, but we can just return it.
        # Actually, for consistency, we can just return the HK result directly.
        return int(next_node)
    
    # For N > 15, use 2-Opt
    # Time budget for 2-Opt: 
    # We have ~150s total. 
    # We are at step 0 (or similar). 
    # Let's allow up to 10 seconds for 2-Opt on this sub-problem.
    # Since we do this for every step, 50 steps * 10s = 500s > 150s. 
    # So we must reduce the time per step.
    # Total budget 150s. 50 steps. 3s per step average.
    # Let's set time_limit = 2.0 seconds for 2-Opt in each step.
    
    time_limit = 2.0
    refined_seq = _two_opt(full_seq, distance_matrix, time_limit=time_limit)
    
    # The refined sequence is [current, n1', n2', ..., nk', destination]
    # The next node to visit is refined_seq[1]
    # The remaining plan is refined_seq[1:-1]
    
    plan = refined_seq[1:-1]
    
    # Cache the plan
    _planned_tour_cache[key] = plan
    
    return int(plan[0])

def _held_karp_next(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Finds the optimal next node for the last k nodes using Held-Karp DP.
    Returns the first node in the optimal sequence.
    """
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    
    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(unv[0])
    
    # Map unvisited nodes to local indices 0..n-1
    full_mask = (1 << n) - 1
    
    INF = float('inf')
    dp = np.full((1 << n, n), INF, dtype=float)
    parent = np.full((1 << n, n), -1, dtype=int)
    
    # Base cases: start at current_node, go to unv[i]
    for i in range(n):
        dp[1 << i, i] = distance_matrix[current_node, unv[i]]
        
    # Fill DP table
    for mask in range(1 << n):
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            current_cost = dp[mask, i]
            if current_cost == INF:
                continue
            
            for j in range(n):
                if mask & (1 << j):
                    continue
                
                new_mask = mask | (1 << j)
                step_cost = distance_matrix[unv[i], unv[j]]
                new_cost = current_cost + step_cost
                
                if new_cost < dp[new_mask, j]:
                    dp[new_mask, j] = new_cost
                    parent[new_mask, j] = i

    # Find the best ending node
    best_cost = INF
    best_last_i = -1
    for i in range(n):
        total_cost = dp[full_mask, i] + distance_matrix[unv[i], destination_node]
        if total_cost < best_cost:
            best_cost = total_cost
            best_last_i = i
            
    if best_last_i == -1:
        return int(unv[0])
        
    # Backtrack to find the first node
    current_mask = full_mask
    current_i = best_last_i
    first_i = -1
    
    while current_mask != 0:
        if current_mask & (current_mask - 1) == 0:
            first_i = current_i
            break
        
        prev_i = parent[current_mask, current_i]
        if prev_i == -1:
            break
        current_mask = current_mask ^ (1 << current_i)
        current_i = prev_i
        
    if first_i == -1:
        first_i = 0
        
    return int(unv[first_i])
