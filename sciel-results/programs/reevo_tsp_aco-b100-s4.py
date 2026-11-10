import time
import numpy as np

_DEADLINE_FRACTION = 0.95
_WALL_LIMIT = 330.0
_start = time.time()


def _deadline_left():
    return _WALL_LIMIT * _DEADLINE_FRACTION - (time.time() - _start)


def _two_opt(d, order):
    """
    Improve a tour using the 2-opt heuristic.
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


def _or_opt(d, order):
    """
    Improve a tour using Or-opt (moving 1 node).
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
        for i in range(n):
            u = best[i]
            prev_node = best[(i - 1) % n]
            next_node = best[(i + 1) % n]
            
            # Cost to remove u from position i
            # Remove edges (prev, u) and (u, next), add (prev, next)
            removal_cost = d[prev_node, u] + d[u, next_node] - d[prev_node, next_node]
            
            # If removing u doesn't increase length (or decreases it), try reinserting
            # Actually, we should always try reinserting if there's a potential gain,
            # but checking removal_cost < 0 is a good heuristic filter for speed, 
            # though not strictly necessary for correctness if we compare final lengths.
            # Let's just try all insertions for robustness, as n=50 is small.
            
            # Create a copy without u
            temp_order = best[:]
            temp_order.pop(i)
            # Length of temp_order is n-1
            # Current best_len includes u. 
            # Let's calculate the length of the tour without u first?
            # No, simpler: iterate over insertion points k in temp_order.
            
            best_in_local = None
            best_len_local = np.inf
            
            for k in range(len(temp_order)):
                # Insert u at index k in temp_order
                # Neighbors in temp_order: prev_k, next_k
                prev_k = temp_order[(k - 1) % len(temp_order)]
                next_k = temp_order[k % len(temp_order)]
                
                # Cost change:
                # Remove edges from temp_order context? No, temp_order is the base.
                # We are inserting u between prev_k and next_k in the sequence defined by temp_order.
                # Original edges in this part of temp_order: (prev_k, next_k)
                # New edges: (prev_k, u) and (u, next_k)
                
                # We need the length of the tour formed by temp_order with u inserted at k.
                # Let's compute delta relative to the tour formed by temp_order alone?
                # It's safer to compute the full delta relative to the original best_len.
                
                # Original best_len has edges:
                # ... (prev_node, u), (u, next_node) ...
                # And the rest of the tour.
                
                # New tour has:
                # ... (prev_k, u), (u, next_k) ...
                # And the rest of the tour (which is temp_order).
                
                # The "rest of the tour" in the new configuration is the same set of edges as in the original,
                # EXCEPT for the edges around u and the edges that were broken in temp_order to insert u.
                
                # Let's use a simpler delta calculation:
                # 1. Remove u from original tour:
                #    Remove (prev_node, u) and (u, next_node). Add (prev_node, next_node).
                #    Len_without_u = best_len - d[prev_node, u] - d[u, next_node] + d[prev_node, next_node]
                
                # 2. Insert u at position k in the tour formed by the remaining nodes (temp_order).
                #    The tour formed by temp_order has length Len_without_u?
                #    NO. temp_order is just the list of nodes. The edges in temp_order are the same as in the original tour,
                #    EXCEPT the connection between prev_node and next_node is now a direct edge in the "skeleton".
                #    Wait, if we remove u, the edge (prev_node, next_node) is formed.
                #    So the "skeleton" tour length is Len_without_u.
                
                #    Now we break the edge (prev_k, next_k) in the skeleton and insert u.
                #    Delta_insert = d[prev_k, u] + d[u, next_k] - d[prev_k, next_k]
                
                #    Total new length = Len_without_u + Delta_insert
                
                len_without_u = best_len - d[prev_node, u] - d[u, next_node] + d[prev_node, next_node]
                delta_insert = d[prev_k, u] + d[u, next_k] - d[prev_k, next_k]
                new_len = len_without_u + delta_insert
                
                if new_len < best_len_local:
                    best_len_local = new_len
                    best_in_local = temp_order[:]
                    best_in_local.insert(k, u)
                    
            if best_in_local is not None and best_len_local < best_len - 1e-10:
                best = best_in_local
                best_len = best_len_local
                improved = True

    return best, best_len

def _swap_opt(d, order):
    """
    Improve a tour by swapping two non-adjacent nodes.
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
        for i in range(n):
            for j in range(i + 2, n):
                # Skip if adjacent (j == i+1) or wrap-around adjacent (j == n-1 and i == 0)
                if j == i + 1:
                    continue
                if i == 0 and j == n - 1:
                    continue
                
                # Nodes to swap
                u = best[i]
                v = best[j]
                
                # Neighbors
                # For u: prev_u, next_u
                # For v: prev_v, next_v
                prev_u = best[(i - 1) % n]
                next_u = best[(i + 1) % n]
                prev_v = best[(j - 1) % n]
                next_v = best[(j + 1) % n]
                
                # Edges to remove:
                # (prev_u, u), (u, next_u)
                # (prev_v, v), (v, next_v)
                
                # Edges to add (after swap):
                # u takes v's place: (prev_v, u), (u, next_v)
                # v takes u's place: (prev_u, v), (v, next_u)
                
                delta = (d[prev_v, u] + d[u, next_v] + d[prev_u, v] + d[v, next_u]) - \
                        (d[prev_u, u] + d[u, next_u] + d[prev_v, v] + d[v, next_v])
                
                if delta < -1e-10:
                    best[i], best[j] = best[j], best[i]
                    best_len += delta
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
    
    # Try all starts for better coverage
    starts = list(range(n))
    
    for s in starts:
        if _deadline_left() < 10.0:
            break
            
        # 1. NN
        tour = _nearest_neighbor_tour(d, start=s)
        
        # 2. 2-Opt
        tour, l = _two_opt(d, tour)
        
        # 3. Or-opt
        tour, l = _or_opt(d, tour)
        
        # 4. Swap-opt
        tour, l = _swap_opt(d, tour)
        
        # 5. Final 2-Opt polish
        tour, l = _two_opt(d, tour)
        
        if l < best_len:
            best_len = l
            best_order = tour
            
    if best_order is None:
        best_order = _nearest_neighbor_tour(d, start=0)
        
    return best_order


def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix for the ACO solver.
    Uses a high-power inverse distance base (power 5) to aggressively suppress
    long-range/crossing edges, combined with strong boosts from a high-quality
    reference tour found via NN + 2-Opt + Or-opt + Swap-opt.
    """
    d = np.asarray(distance_matrix, dtype=float)
    n = d.shape[0]
    
    if n <= 1:
        return d.copy()
    
    # Base heuristic: 1 / distance^5
    # This creates a very strong preference for short edges.
    # For d~0.1, H~100,000. For d~0.5, H~32. For d~1.0, H~1.
    d_safe = np.where(d > 1e-10, d, 1e10)
    H = 1.0 / (d_safe ** 5)
    
    # Ensure diagonal is small
    np.fill_diagonal(H, 1e-9)
    
    # Boost edges from a good tour
    if n > 2 and _deadline_left() > 15.0:
        tour = _find_best_tour(d)
        
        # Boost value: Set to a value significantly higher than the max base heuristic
        # Max base heuristic is ~1e5. Set boost to 1e7 to dominate.
        boost_val = 1e7
        
        for i in range(n):
            u = tour[i]
            v = tour[(i + 1) % n]
            H[u, v] = boost_val
            H[v, u] = boost_val
            
    # Sanitize per contract
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    return H
