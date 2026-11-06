"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # 5s safety margin before the 120s kill

def _time_exceeded():
    return time.time() > _DEADLINE

def _get_route_cost(route, dm, depot):
    """Calculate the total cost of a route including return to depot."""
    if len(route) <= 1:
        return 0.0
    # route is [start, n1, n2, ...]
    # Cost = sum(dm[route[i], route[i+1]]) + dm[route[-1], depot]
    # Using numpy for speed
    if len(route) == 2:
        return dm[route[0], route[1]] + dm[route[1], depot]
    
    # Vectorized distance sum
    # dm[route[:-1], route[1:]] gives an array of distances between consecutive nodes
    dists = dm[route[:-1], route[1:]]
    total = np.sum(dists) + dm[route[-1], depot]
    return total

def _vectorized_2opt(route, dm, depot):
    """
    Apply 2-opt moves using vectorization where possible.
    route: list of nodes [start, n1, n2, ...]
    Returns optimized route.
    """
    n = len(route)
    if n < 4: # 2-opt needs at least 4 nodes (start, a, b, end) to reverse [a, b]
        return route
    
    improved = True
    max_iter = 20
    iter_count = 0
    
    while improved and iter_count < max_iter and not _time_exceeded():
        improved = False
        iter_count += 1
        
        # We can vectorize the calculation of the gain for all (i, j) pairs.
        # However, applying them is sequential. 
        # Let's compute all potential gains first to find the best one, or just iterate.
        # Given n <= 50, n^2 = 2500 checks. This is fast.
        
        # Precompute matrix of distances for the route nodes
        # R = route
        # For 2-opt, we reverse segment route[i+1 : j+1]
        # This changes edges:
        # 1. (route[i], route[i+1]) and (route[j], route[j+1] or depot)
        # 2. (route[i], route[j]) and (route[i+1], route[j+1] or depot)
        
        # Let's stick to a slightly optimized loop that checks all pairs
        # but computes gains efficiently.
        
        best_gain = 1e-9
        best_i = -1
        best_j = -1
        
        # To be faster, we can just loop i and j
        # route indices: 0 is start. We don't change index 0.
        # i ranges from 0 to n-3 (since we need i+1 and i+2)
        # j ranges from i+2 to n-1
        
        for i in range(n - 2):
            node_i = route[i]
            node_i1 = route[i+1]
            
            # Vectorized check for all j > i+1
            # j indices in route: i+2 to n-1
            if i + 2 >= n:
                continue
                
            js = np.arange(i + 2, n)
            nodes_j = route[js]
            
            # Edge 1 to remove: (node_i, node_i1) is constant for fixed i
            # Edge 2 to remove: (node_j, next_j)
            # next_j is route[j+1] if j < n-1 else depot
            
            next_js = np.where(js < n - 1, route[js + 1], depot)
            
            old_costs = dm[node_i, node_i1] + dm[nodes_j, next_js]
            
            # New edges:
            # (node_i, node_j) and (node_i1, next_j)
            new_costs = dm[node_i, nodes_j] + dm[node_i1, next_js]
            
            gains = old_costs - new_costs
            
            max_gain = np.max(gains)
            if max_gain > best_gain:
                best_gain = max_gain
                best_j_idx_in_js = np.argmax(gains)
                best_j = js[best_j_idx_in_js]
                
        if best_i != -1 or best_j != -1: # Actually we need to track best_i too, let's fix logic
             # The above loop overwrites best_i. We need to keep the i that gave the best global gain.
             # Let's re-do this properly: find global best (i, j)
             pass
        
        # Simpler: Just find the single best move in the whole matrix
        # Since n is small, we can just do the double loop but with numpy inside for j
        # But the above loop structure was slightly flawed for tracking global best.
        
        # Let's restart the inner logic cleanly
        best_gain = 1e-9
        best_i = -1
        best_j = -1
        
        for i in range(n - 2):
            node_i = route[i]
            node_i1 = route[i+1]
            
            js = np.arange(i + 2, n)
            if len(js) == 0: continue
            nodes_j = route[js]
            next_js = np.where(js < n - 1, route[js + 1], depot)
            
            old_costs = dm[node_i, node_i1] + dm[nodes_j, next_js]
            new_costs = dm[node_i, nodes_j] + dm[node_i1, next_js]
            gains = old_costs - new_costs
            
            local_max_gain = np.max(gains)
            if local_max_gain > best_gain:
                best_gain = local_max_gain
                best_i = i
                best_j = js[np.argmax(gains)]
                
        if best_i != -1:
            # Apply move: reverse route[best_i+1 : best_j+1]
            # route is a list. 
            seg = route[best_i+1 : best_j+1]
            seg.reverse()
            route[best_i+1 : best_j+1] = seg
            improved = True
            # After applying one move, we restart the search (standard 2-opt behavior)
            # This is safe and simple.
            
    return route

def _vectorized_oropt(route, dm, depot):
    """
    Apply Or-opt (single node relocation) moves.
    """
    n = len(route)
    if n < 4:
        return route
        
    improved = True
    max_iter = 10
    iter_count = 0
    
    while improved and iter_count < max_iter and not _time_exceeded():
        improved = False
        iter_count += 1
        
        best_gain = 1e-9
        best_k = -1
        best_m = -1
        
        # k is the node to move (index 1 to n-1, we don't move start)
        for k in range(1, n):
            node_k = route[k]
            
            # Neighbors of k in current route
            L = route[k-1]
            R = route[k+1] if k < n-1 else depot
            
            # Cost to remove k: dm[L, node_k] + dm[node_k, R]
            # Cost to connect L and R: dm[L, R]
            # Net change from removal: dm[L, R] - (dm[L, node_k] + dm[node_k, R])
            removal_gain = dm[L, R] - (dm[L, node_k] + dm[node_k, R])
            
            if removal_gain <= 0:
                # If removing doesn't help locally, moving it probably won't help much 
                # unless it breaks a very bad edge elsewhere. 
                # But Or-opt gain = removal_gain + insertion_gain.
                # Let's check all insertion points.
                pass
            
            # Try inserting node_k at all positions m
            # m is the index in the route *after* removing k? 
            # Easier: iterate target position m in current route, skipping k and k-1?
            # Let's define m as the index in the *original* route where we insert.
            # If we insert at m, it goes between route[m-1] and route[m].
            # If m == k or m == k+1, it's effectively a no-op or swap (handled by 2-opt or no change).
            
            # We want to find the best m != k, m != k+1 (since k-1 is L, k is node, k+1 is R)
            # Actually, inserting at k-1 puts it before L. Inserting at k puts it back.
            
            # Let's just evaluate all m in 0..n-1 (conceptual slots)
            # Slot 0: before route[0] (not allowed, start is fixed)
            # Slot m: between route[m-1] and route[m] for m=1..n-1
            # Slot n: after route[n-1] (before depot)
            
            # We need to be careful: if m > k, the indices shift in the "removed" view.
            # Let's just simulate the cost for each potential insertion position in the *current* list,
            # ignoring the fact that k is there, and calculating the delta.
            
            # For each potential insertion index m (0 to n):
            # If m == 0: Not allowed (start fixed)
            # If m > 0: Insert between route[m-1] and route[m] (if m < n) or route[n-1] and depot (if m == n)
            
            # We must skip positions that result in no change or are invalid.
            # Positions:
            # m = k: Insert between L and node_k -> No change (node_k is already there)
            # m = k+1: Insert between node_k and R -> No change
            
            # Let's iterate m from 1 to n
            for m in range(1, n + 1):
                if m == k or m == k + 1:
                    continue
                
                # Determine neighbors for insertion
                if m < n:
                    L_ins = route[m-1]
                    R_ins = route[m]
                else:
                    L_ins = route[n-1]
                    R_ins = depot
                
                # If we are inserting at m, and m > k, then L_ins and R_ins are shifted?
                # No, we are calculating the cost *if* we moved it there.
                # The cost of insertion is dm[L_ins, node_k] + dm[node_k, R_ins] - dm[L_ins, R_ins]
                # BUT, if m > k, L_ins and R_ins in the *original* array include node_k?
                # No, L_ins and R_ins are defined by their indices in the original array.
                # If we move node_k to position m, the relative order of other nodes stays same.
                # However, if m > k, the node at index m in the original array is actually 
                # the node at index m-1 in the "removed" array.
                # My L_ins/R_ins logic uses the original array indices.
                # If m > k, then L_ins = route[m-1] is correct?
                # Example: Route [0, 1, 2, 3]. Move 1 (idx 1) to idx 3 (between 2 and 3).
                # m=3. L_ins = route[2]=2, R_ins = route[3]=3.
                # This is correct.
                # Example: Move 1 (idx 1) to idx 2 (between 1 and 2). This is m=k+1, skipped.
                # Example: Move 1 (idx 1) to idx 0. Not allowed.
                
                # One edge case: if m == k, we skip.
                # If m == k-1, we insert before L.
                
                ins_gain = dm[L_ins, node_k] + dm[node_k, R_ins] - dm[L_ins, R_ins]
                total_gain = removal_gain + ins_gain
                
                if total_gain > best_gain:
                    best_gain = total_gain
                    best_k = k
                    best_m = m
        
        if best_k != -1:
            # Apply move
            node_to_move = route.pop(best_k)
            # If best_m > best_k, the index best_m in the *new* list is best_m - 1
            # We want to insert at position best_m in the *original* indexing scheme?
            # No, we want it to be at position best_m in the *new* list?
            # My m was defined as "insert between route[m-1] and route[m]".
            # In the new list (after pop), the indices are shifted if m > k.
            # If we pop k, and we want to insert at original position m:
            # If m > k, the new index is m - 1.
            # If m < k, the new index is m.
            
            insert_idx = best_m - 1 if best_m > best_k else best_m
            
            # Bounds check
            if insert_idx < 0: insert_idx = 0
            if insert_idx > len(route): insert_idx = len(route)
            
            route.insert(insert_idx, node_to_move)
            improved = True
            
    return route

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic: Vectorized Local Search on Multiple Candidates.
    
    1. Generate a few diverse initial route candidates (NN, Regret, Random).
    2. Optimize each using Vectorized 2-Opt and Or-Opt.
    3. Return the first node of the best optimized route.
    """
    if len(unvisited_nodes) == 0:
        return depot

    # Filter by capacity
    feasible = [i for i in unvisited_nodes if demands[i] <= rest_capacity]
    if len(feasible) == 0:
        return depot
    if len(feasible) == 1:
        return int(feasible[0])

    # Time guard: if we are running out of time, fall back to simple nearest
    if _time_exceeded():
        idx = np.argmin(distance_matrix[current_node, feasible])
        return int(feasible[idx])

    dm = distance_matrix
    feas_arr = np.array(feasible)
    
    # 1. Generate Candidates
    
    candidates = []
    
    # Candidate 1: Nearest Neighbor
    route_nn = [current_node]
    remaining_cap = rest_capacity
    unvisited = set(feasible)
    last = current_node
    while unvisited:
        if _time_exceeded():
            break
        u_arr = np.array(list(unvisited))
        if len(u_arr) == 0: break
        dists = dm[last, u_arr]
        valid_mask = demands[u_arr] <= remaining_cap
        if not np.any(valid_mask):
            break
        valid_u = u_arr[valid_mask]
        valid_dists = dists[valid_mask]
        min_idx = np.argmin(valid_dists)
        next_node = valid_u[min_idx]
        route_nn.append(next_node)
        remaining_cap -= demands[next_node]
        unvisited.remove(next_node)
        last = next_node
    if len(route_nn) > 1:
        candidates.append(route_nn)

    # Candidate 2: Regret-2 (Choose node with highest regret: diff between 1st and 2nd best insertion cost)
    # Approximation: Just pick farthest from current? Or random?
    # Let's try a "Savings" like approach or just Random for diversity.
    # Random candidate
    route_rand = [current_node]
    remaining_cap = rest_capacity
    unvisited = set(feasible)
    last = current_node
    while unvisited:
        if _time_exceeded():
            break
        u_arr = np.array(list(unvisited))
        if len(u_arr) == 0: break
        valid_mask = demands[u_arr] <= remaining_cap
        if not np.any(valid_mask):
            break
        valid_u = u_arr[valid_mask]
        # Pick random
        idx = np.random.randint(0, len(valid_u))
        next_node = valid_u[idx]
        route_rand.append(next_node)
        remaining_cap -= demands[next_node]
        unvisited.remove(next_node)
        last = next_node
    if len(route_rand) > 1:
        candidates.append(route_rand)
        
    # Candidate 3: Farthest Neighbor (sometimes helps with clustering)
    route_far = [current_node]
    remaining_cap = rest_capacity
    unvisited = set(feasible)
    last = current_node
    while unvisited:
        if _time_exceeded():
            break
        u_arr = np.array(list(unvisited))
        if len(u_arr) == 0: break
        dists = dm[last, u_arr]
        valid_mask = demands[u_arr] <= remaining_cap
        if not np.any(valid_mask):
            break
        valid_u = u_arr[valid_mask]
        valid_dists = dists[valid_mask]
        max_idx = np.argmax(valid_dists)
        next_node = valid_u[max_idx]
        route_far.append(next_node)
        remaining_cap -= demands[next_node]
        unvisited.remove(next_node)
        last = next_node
    if len(route_far) > 1:
        candidates.append(route_far)

    if not candidates:
        # Fallback to NN if all failed (should not happen)
        idx = np.argmin(dm[current_node, feas_arr])
        return int(feas_arr[idx])

    # 2. Optimize Candidates
    
    best_route = None
    best_cost = float('inf')
    
    for route in candidates:
        if _time_exceeded():
            # If we don't have a best route yet, use the first one
            if best_route is None:
                best_route = route
            break
            
        # Optimize
        opt_route = _vectorized_2opt(route[:], dm, depot)
        opt_route = _vectorized_oropt(opt_route, dm, depot)
        
        cost = _get_route_cost(opt_route, dm, depot)
        
        if cost < best_cost:
            best_cost = cost
            best_route = opt_route
            
    if best_route is None or len(best_route) <= 1:
        # Fallback
        idx = np.argmin(dm[current_node, feas_arr])
        return int(feas_arr[idx])
        
    return int(best_route[1])

# EVOLVE-BLOCK-END
