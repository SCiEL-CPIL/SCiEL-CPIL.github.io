import time
import numpy as np

_DEADLINE_FRACTION = 0.85
_WALL_LIMIT = 330.0
_start = time.time()


def _deadline_left():
    return _WALL_LIMIT * _DEADLINE_FRACTION - (time.time() - _start)


def _two_opt(d, order):
    """
    Improve a tour using the 2-opt heuristic.
    Returns the improved order and its length.
    """
    n = len(order)
    if n < 4:
        return order, float(sum(d[order[i], order[(i + 1) % n]] for i in range(n)))
    
    def tour_len(o):
        return float(np.sum(d[o, np.roll(o, -1)]))

    best = list(order)
    best_len = tour_len(best)
    
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Edges to remove: (i-1, i) and (j, j+1)
                # Edges to add: (i-1, j) and (i, j+1)
                a = best[i - 1]
                b = best[i]
                c = best[j]
                e = best[(j + 1) % n]
                
                delta = (d[a, c] + d[b, e]) - (d[a, b] + d[c, e])
                
                if delta < -1e-10:
                    best[i:j + 1] = best[i:j + 1][::-1]
                    best_len += delta
                    improved = True
                    
    return best, best_len


def _three_opt(d, order):
    """
    Improve a tour using 3-opt (reversing and moving segments or swapping edges).
    A simple implementation: try all pairs of edges to swap in a way that preserves connectivity, 
    or specifically the "reversal" variant which is 2-opt, and the "exchange" variant.
    Actually, standard 3-opt involves removing 3 edges and reinserting. 
    Given time constraints, a simplified 3-opt that checks specific edge rearrangements is sufficient.
    We will implement a version that tries to swap two non-adjacent segments, which is effectively 
    a generalized 2-opt but we'll stick to a robust 2-opt + a simple Or-opt (move node) for safety and speed.
    """
    # For n=50, 2-opt is often enough. Or-opt (moving 1 node) is fast and effective.
    def tour_len(o):
        return float(np.sum(d[o, np.roll(o, -1)]))
        
    best = list(order)
    best_len = tour_len(best)
    n = len(best)
    
    improved = True
    while improved:
        improved = False
        # Or-opt: Move node i to position j
        for i in range(n):
            # Don't move if i is 0 and j is 1? Indices in list
            # Node at best[i]
            u = best[i]
            # Remove u
            # Neighbors: prev = best[(i-1)%n], next = best[(i+1)%n]
            prev_node = best[(i - 1) % n]
            next_node = best[(i + 1) % n]
            
            # Cost to remove: d[prev, u] + d[u, next] - d[prev, next]
            removal_cost = d[prev_node, u] + d[u, next_node] - d[prev_node, next_node]
            
            if removal_cost < -1e-10:
                # It's beneficial to remove u
                # Try inserting u at every other position
                # We need to simulate the tour without u
                # To keep it simple and fast, we can just try inserting u into the current list 
                # at various positions, but that's O(n^3).
                # For n=50, O(n^3) = 125,000 ops, which is fine.
                
                # Let's do a more direct search:
                # Create a copy without u
                temp_order = best[:]
                temp_order.pop(i)
                # Now insert u at every position k in temp_order
                best_in_local = temp_order
                best_len_local = best_len + removal_cost # This is not exactly right, length changes with insertion
                # Actually, let's just calculate full length for simplicity in the loop, 
                # or optimize delta.
                
                # Let's try all insertion positions
                for k in range(len(temp_order)):
                    # Insert u at index k
                    # Neighbors in temp_order: prev_k = temp_order[(k-1)%len], next_k = temp_order[k%len]
                    prev_k = temp_order[(k - 1) % len(temp_order)]
                    next_k = temp_order[k % len(temp_order)]
                    
                    # If k == 0, prev is last element of temp_order
                    # If k == len(temp_order)-1, next is first element? No, cyclic.
                    
                    new_len = best_len_local + d[prev_k, u] + d[u, next_k] - d[prev_k, next_k]
                    
                    if new_len < best_len_local:
                        best_len_local = new_len
                        best_in_local = temp_order[:]
                        best_in_local.insert(k, u)
                        
                if best_len_local < best_len:
                    best = best_in_local
                    best_len = best_len_local
                    improved = True
                    
    return best, best_len


def _nearest_neighbor_tour(d, start=0):
    n = d.shape[0]
    visited = np.zeros(n, dtype=bool)
    order = [start]
    visited[start] = True
    cur = start
    for _ in range(n - 1):
        dist = d[cur].copy()
        dist[visited] = np.inf
        nxt = int(np.argmin(dist))
        order.append(nxt)
        visited[nxt] = True
        cur = nxt
    return order


def _find_best_tour(d):
    n = d.shape[0]
    if n <= 2:
        return list(range(n))
        
    best_order = None
    best_len = np.inf
    
    # Try multiple starts - use all nodes for better coverage with time budget
    starts = list(range(n))
    
    for s in starts:
        if _deadline_left() < 3.0:
            break
            
        # 1. NN
        tour = _nearest_neighbor_tour(d, start=s)
        
        # 2. 2-Opt
        tour, l = _two_opt(d, tour)
        
        # 3. 3-Opt (Or-opt implementation above)
        tour, l = _three_opt(d, tour)
        
        if l < best_len:
            best_len = l
            best_order = tour
            
    if best_order is None:
        best_order = _nearest_neighbor_tour(d, start=0)
        
    return best_order


def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix for the ACO solver.
    Uses a smooth inverse-distance base (power 3) combined with topological boosts
    from a near-optimal tour and k-NN graph to guide ACO ants toward short, 
    non-crossing tours.
    """
    d = np.asarray(distance_matrix, dtype=float)
    n = d.shape[0]
    
    if n <= 1:
        return d.copy()
    
    # Base heuristic: smooth inverse distance with power 3 to aggressively
    # suppress long-range/crossing edges
    # Avoid division by zero on diagonal
    d_safe = d.copy()
    np.fill_diagonal(d_safe, 1.0)  # temporary to avoid div by zero
    base_h = 1.0 / (d_safe ** 3)
    np.fill_diagonal(base_h, 0.0)  # diagonal should be 0 (will become 1e-9 later)
    
    # Initialize H with the base heuristic
    H = base_h.copy()
    
    if n > 2 and _deadline_left() > 5.0:
        # Find a high-quality tour
        tour = _find_best_tour(d)
        
        # Boost tour edges significantly
        tour_boost = 5000.0
        
        for i in range(n):
            u = tour[i]
            v = tour[(i + 1) % n]
            # Set both directions for flexibility
            H[u, v] = max(H[u, v], tour_boost)
            H[v, u] = max(H[v, u], tour_boost)
        
        # Add k-NN graph boosts to further suppress crossing edges
        # For each node, boost edges to its k nearest neighbors
        k = 5
        knn_boost = 100.0
        
        for i in range(n):
            # Get distances from node i to all others (excluding self)
            dists = d[i].copy()
            dists[i] = np.inf
            # Get k nearest neighbors
            nn_indices = np.argsort(dists)[:k]
            for j in nn_indices:
                j = int(j)
                H[i, j] = max(H[i, j], knn_boost)
                H[j, i] = max(H[j, i], knn_boost)
    
    # Ensure all values are positive and finite per contract
    # Non-finite or non-positive values become 1e-9
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    return H
