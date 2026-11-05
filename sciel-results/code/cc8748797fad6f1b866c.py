import random
import time

def construct():
    n = 31
    target_edges = 91
    
    start_time = time.time()
    time_limit = 115  # Safety margin for 120s limit
    
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_c4_count = float('inf')
    best_edges = None
    best_neighbors = None
    
    attempts = 0
    while time.time() - start_time < time_limit:
        attempts += 1
        random.shuffle(all_pairs)
        
        neighbors = [0] * n
        edge_count = 0
        current_edges = []
        
        for u, v in all_pairs:
            common = (neighbors[u] & neighbors[v]).bit_count()
            if common <= 1:
                neighbors[u] |= (1 << v)
                neighbors[v] |= (1 << u)
                edge_count += 1
                current_edges.append((u, v))
                if edge_count >= 90:
                    break
        
        if edge_count < 90:
            continue
        
        # Find the best 91st edge to add
        edge_set = set(current_edges)
        min_cost = float('inf')
        best_add = None
        
        for u, v in all_pairs:
            if (u, v) in edge_set:
                continue
            common = (neighbors[u] & neighbors[v]).bit_count()
            cost = common * (common - 1) // 2
            if cost < min_cost:
                min_cost = cost
                best_add = (u, v)
        
        if best_add is None:
            continue
        
        # Build the 91-edge graph
        final_edges = current_edges + [best_add]
        final_neighbors = neighbors[:]
        u, v = best_add
        final_neighbors[u] |= (1 << v)
        final_neighbors[v] |= (1 << u)
        
        # Count C4s in this 91-edge graph
        def count_c4s(nbrs):
            total = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if (nbrs[i] >> j) & 1:
                        common = (nbrs[i] & nbrs[j]).bit_count()
                        total += common * (common - 1) // 2
            return total // 4
        
        current_c4s = count_c4s(final_neighbors)
        
        # Local search: try edge swaps to reduce C4 count
        improved = True
        local_time_limit = time_limit - (time.time() - start_time) - 3
        local_start = time.time()
        
        while improved and time.time() - local_start < local_time_limit:
            improved = False
            for i in range(len(final_edges)):
                if time.time() - local_start > local_time_limit:
                    break
                u, v = final_edges[i]
                # Remove edge (u,v)
                saved_nu = final_neighbors[u]
                saved_nv = final_neighbors[v]
                final_neighbors[u] = saved_nu & ~(1 << v)
                final_neighbors[v] = saved_nv & ~(1 << u)
                edge_set.discard((u, v))
                
                # Try adding a different edge
                best_alt_cost = current_c4s
                best_alt_pair = None
                for a, b in all_pairs:
                    if (a, b) in edge_set:
                        continue
                    common = (final_neighbors[a] & final_neighbors[b]).bit_count()
                    cost = common * (common - 1) // 2
                    if cost < best_alt_cost:
                        best_alt_cost = cost
                        best_alt_pair = (a, b)
                
                # Restore
                final_neighbors[u] = saved_nu
                final_neighbors[v] = saved_nv
                edge_set.add((u, v))
                
                if best_alt_pair is not None and best_alt_cost < current_c4s:
                    a, b = best_alt_pair
                    final_edges[i] = (a, b)
                    edge_set.add((a, b))
                    final_neighbors[a] |= (1 << b)
                    final_neighbors[b] |= (1 << a)
                    current_c4s = best_alt_cost
                    improved = True
                    break
        
        if current_c4s < best_c4_count:
            best_c4_count = current_c4s
            best_edges = final_edges[:]
            best_neighbors = final_neighbors[:]
            
            if best_c4_count == 1:
                break
    
    if best_edges is None:
        # Fallback: return first 91 pairs
        return [[i, j] for i, j in all_pairs[:91]]
    
    return [[u, v] for u, v in best_edges]

if hasattr(construct, '_best_edges'):
    del construct._best_edges
