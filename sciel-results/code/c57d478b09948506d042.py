import random
import time
import math

def construct():
    n = 31
    target_edges = 91

    start_time = time.time()
    time_limit = 115  # Safety margin for 120s limit

    # Deterministic construction: polarity graph of PG(2,5)
    # Vertices: 25 affine points (x,y) in F5^2, plus 5 line-at-infinity points L_m, plus 1 point at infinity P
    # Edges:
    #   - (x1,y1)-(x2,y2) if (x2-x1)^2 + (y2-y1)^2 == 0 in F5  (isotropic direction)
    #   - L_m connected to all (x,y) with y == m*x (5*5=25 edges)
    #   - P connected to all L_m (5 edges)
    # Total: 25 + 25 + 5 = 55 edges... wait, let me recount.
    # Actually for q=5, the polarity graph has (q+1)^2 = 36 vertices and q(q+1)^2/2 = 45 edges? 
    # No, that's not right either. Let me use a different approach.
    
    # Better: use the standard C4-free construction from finite geometry.
    # For n = q^2 + q + 1 = 31 with q=5, use the incidence graph approach.
    # Actually, let's just use a greedy random search which is simpler and proven to work.
    
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_neighbors = [0] * n
    best_edge_count = 0
    best_edges = []
    
    attempts = 0
    while time.time() - start_time < time_limit:
        attempts += 1
        # Randomize order
        random.shuffle(all_pairs)
        
        neighbors = [0] * n
        edge_count = 0
        current_edges = []
        
        for u, v in all_pairs:
            # Check if adding (u, v) creates a 4-cycle
            common = (neighbors[u] & neighbors[v]).bit_count()
            if common <= 1:
                # Add edge
                neighbors[u] |= (1 << v)
                neighbors[v] |= (1 << u)
                edge_count += 1
                current_edges.append((u, v))
                
                if edge_count >= 90:
                    break
        
        if edge_count > best_edge_count:
            best_edge_count = edge_count
            best_neighbors = neighbors[:]
            best_edges = current_edges[:]
            
        if best_edge_count == 90:
            break

    if best_edge_count < 90:
        # Fallback: use whatever we have
        final_edges = best_edges[:]
    else:
        final_edges = best_edges[:]

    # Now we have a C4-free core with 90 edges. We need to add 1 more edge.
    # The 91st edge will create some number of 4-cycles.
    # We want to choose the edge that creates the FEWEST 4-cycles.
    
    current_neighbors = best_neighbors[:]
    current_edge_set = set(final_edges)
    
    # Find the edge with minimum C4 creation cost
    best_cost = float('inf')
    best_pair = None
    
    for u, v in all_pairs:
        if (u, v) in current_edge_set:
            continue
        common = (current_neighbors[u] & current_neighbors[v]).bit_count()
        cost = common * (common - 1) // 2
        if cost < best_cost:
            best_cost = cost
            best_pair = (u, v)
    
    if best_pair is not None:
        u, v = best_pair
        final_edges.append((u, v))
        current_edge_set.add((u, v))
        current_neighbors[u] |= (1 << v)
        current_neighbors[v] |= (1 << u)
    
    # Local search: try to swap edges to reduce total C4 count
    # Compute initial C4 count
    def count_c4s(edges, nbrs):
        total = 0
        for u, v in edges:
            common = (nbrs[u] & nbrs[v]).bit_count()
            total += common * (common - 1) // 2
        # Each C4 is counted 4 times (once per edge), so divide by 4
        return total // 4
    
    current_c4s = count_c4s(final_edges, current_neighbors)
    
    # Try local search: remove an edge from the 90-edge core and add a different edge
    # to see if we can reduce total C4s
    if time.time() - start_time < time_limit - 5:
        improved = True
        while improved and time.time() - start_time < time_limit - 5:
            improved = False
            for i in range(len(final_edges)):
                u, v = final_edges[i]
                # Remove this edge
                saved_nu = current_neighbors[u]
                saved_nv = current_neighbors[v]
                current_neighbors[u] = saved_nu & ~(1 << v)
                current_neighbors[v] = saved_nv & ~(1 << u)
                current_edge_set.discard((u, v))
                
                # Try adding a different edge
                best_alt_cost = current_c4s
                best_alt_pair = None
                for a, b in all_pairs:
                    if (a, b) in current_edge_set:
                        continue
                    common = (current_neighbors[a] & current_neighbors[b]).bit_count()
                    cost = common * (common - 1) // 2
                    if cost < best_alt_cost:
                        best_alt_cost = cost
                        best_alt_pair = (a, b)
                
                # Restore
                current_neighbors[u] = saved_nu
                current_neighbors[v] = saved_nv
                current_edge_set.add((u, v))
                
                if best_alt_pair is not None and best_alt_cost < current_c4s:
                    # Accept the swap
                    a, b = best_alt_pair
                    final_edges[i] = (a, b)
                    current_edge_set.add((a, b))
                    current_neighbors[a] |= (1 << b)
                    current_neighbors[b] |= (1 << a)
                    current_c4s = best_alt_cost
                    improved = True
                    break
    
    return [[u, v] for u, v in final_edges]

if hasattr(construct, '_best_edges'):
    del construct._best_edges
