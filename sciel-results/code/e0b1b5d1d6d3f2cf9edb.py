import numpy as np
import time

# Global state for deadline management
_START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin for 150s limit

def _compute_mst_cost(nodes, distance_matrix):
    """
    Compute the cost of the Minimum Spanning Tree for a set of nodes.
    This provides a lower bound on the cost to visit all nodes in the set.
    Uses Prim's algorithm with a simple priority queue (O(n^2) which is fine for n<=50).
    """
    if len(nodes) <= 1:
        return 0.0
    
    # Map nodes to local indices for faster access if needed, 
    # but direct indexing into distance_matrix is fine.
    # Prim's algorithm
    in_mst = np.zeros(len(nodes), dtype=bool)
    min_dist = np.full(len(nodes), np.inf)
    
    # Start from the first node in the set
    min_dist[0] = 0.0
    mst_cost = 0.0
    
    for _ in range(len(nodes)):
        # Find the unvisited node with the minimum distance to the MST
        # Since n is small, linear scan is fast
        min_val = np.inf
        min_idx = -1
        for i in range(len(nodes)):
            if not in_mst[i] and min_dist[i] < min_val:
                min_val = min_dist[i]
                min_idx = i
        
        if min_idx == -1:
            break
            
        in_mst[min_idx] = True
        mst_cost += min_val
        
        # Update distances for remaining nodes
        # dist from nodes[min_idx] to all other nodes
        current_node = nodes[min_idx]
        for j in range(len(nodes)):
            if not in_mst[j]:
                other_node = nodes[j]
                d = distance_matrix[current_node, other_node]
                if d < min_dist[j]:
                    min_dist[j] = d
                    
    return mst_cost

def _select_insertion_heuristic(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Heuristic based on insertion cost estimation.
    For each candidate node, we estimate the cost of adding it to the current path
    and the cost of completing the tour from there using an MST approximation.
    
    Score = (dist(prev, candidate) + dist(candidate, next_approx) - dist(prev, next_approx)) 
            + alpha * MST_cost(remaining_unvisited)
            
    Since we are picking ONE next node, we compare candidates based on:
    1. Immediate edge cost: dist(current, candidate)
    2. Future cost estimation: MST of the rest of the unvisited nodes + connection to destination.
    
    Actually, a simpler and more robust global metric is:
    Estimate total tour length if we pick candidate c:
    L(c) = dist(current, c) + dist(c, ...rest...) + ... + dist(last, dest)
    
    Approximation:
    L(c) ≈ dist(current, c) + MST_cost(unvisited \ {c}) + dist(c, destination) * 0.5 ?
    No, that's not quite right.
    
    Let's use the following logic:
    When we pick c, the remaining nodes are R = unvisited \ {c}.
    The cost to complete the tour from c visiting R and returning to dest is roughly:
    MST_cost(R) + min(dist(c, R), dist(c, dest)) ... this is getting complex.
    
    A standard heuristic for TSP construction (like Nearest Neighbor) is local.
    A better global heuristic is "Minimum Incremental Cost".
    
    Let's try:
    Score(c) = dist(current, c) + 0.5 * (MST_cost(unvisited) - min_edge_to_c_in_MST?)
    
    Actually, let's stick to a simpler global proxy:
    1. Calculate MST cost of ALL unvisited nodes.
    2. For each candidate c, calculate the "penalty" of removing c from the MST set?
       Or simply: Estimate the cost to visit the rest of the nodes from c.
       
    Let's use:
    Score(c) = dist(current, c) + dist(c, destination_node) * 0.1 + MST_cost(unvisited \ {c})
    
    This encourages picking nodes that are close to current, but also penalizes leaving
    a "scattered" set of remaining nodes (high MST cost).
    """
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    
    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(unv[0])
    
    # Precompute MST cost for the full set of unvisited nodes
    # This is O(n^2) ~ 2500 ops.
    # We need MST for n-1 subsets? That would be O(n^3) ~ 125k ops.
    # For n=50, this is very fast.
    
    # Optimization: 
    # Instead of full MST for each subset, we can use a cheaper metric.
    # But let's try the MST metric first as it captures global structure.
    
    # To speed up, we can approximate the change in MST cost.
    # Or just calculate it. 50 * (50^2) = 125,000 operations. In Python, this might take ~10-50ms.
    # 50 calls * 50ms = 2.5s. Acceptable within 140s.
    
    best_score = np.inf
    best_idx = 0
    
    # d_cur: distances from current to all unvisited
    d_cur = distance_matrix[current_node, unv]
    
    # d_dest: distances from unvisited to destination
    d_dest = distance_matrix[unv, destination_node]
    
    for i in range(n):
        candidate = unv[i]
        
        # Immediate cost
        cost_start = d_cur[i]
        
        # Remaining nodes
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        remaining = unv[mask]
        
        # Cost to complete from candidate to destination visiting remaining
        # Approximation: MST(remaining) + min connection from candidate to (remaining U dest)
        # Actually, the tour must end at destination.
        # So we need to visit remaining and end at dest.
        # Lower bound: MST(remaining U {dest})? No, dest is fixed end.
        # Let's use: MST(remaining) + min(dist(candidate, remaining), dist(candidate, dest))
        # This is a rough lower bound on the completion cost.
        
        if len(remaining) == 0:
            completion_cost = d_dest[i]
        else:
            mst_cost = _compute_mst_cost(remaining, distance_matrix)
            # Connect candidate to the best node in remaining or destination
            min_conn = min(np.min(distance_matrix[candidate, remaining]), distance_matrix[candidate, destination_node])
            completion_cost = mst_cost + min_conn
            
        total_est = cost_start + completion_cost
        
        if total_est < best_score:
            best_score = total_est
            best_idx = i
            
    return int(unv[best_idx])

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

    best_cost = INF
    best_last_i = -1
    for i in range(n):
        total_cost = dp[full_mask, i] + distance_matrix[unv[i], destination_node]
        if total_cost < best_cost:
            best_cost = total_cost
            best_last_i = i
            
    if best_last_i == -1:
        return int(unv[0])
        
    current_mask = full_mask
    current_i = best_last_i
    first_i = -1
    
    path = []
    while current_mask != 0:
        path.append(current_i)
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

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    
    # Check time limit
    elapsed = time.time() - _START_TIME
    if elapsed > TIME_LIMIT:
        # Fallback to nearest neighbor to ensure we finish
        if n == 0:
            return int(destination_node)
        if n == 1:
            return int(unv[0])
        d_cur = distance_matrix[current_node, unv]
        return int(unv[np.argmin(d_cur)])

    # Use Exact DP for small remaining sets
    if n <= 15:
        return _held_karp_next(current_node, destination_node, unv, distance_matrix)
    else:
        # Use Global Insertion/MST Heuristic
        return _select_insertion_heuristic(current_node, destination_node, unv, distance_matrix)
