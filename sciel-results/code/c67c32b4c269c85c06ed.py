import random
import time

def construct():
    n = 31
    target_edges = 91
    
    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_score = float('inf')
    best_edges = None
    
    start_time = time.time()
    time_limit = 115  # Safety margin for 120s limit
    
    attempts = 0
    max_attempts = 10000  # Cap attempts to ensure we finish
    
    while time.time() - start_time < time_limit and attempts < max_attempts:
        attempts += 1
        random.shuffle(all_pairs)
        
        neighbors = [0] * n
        edge_count = 0
        current_edges = []
        edge_set = set()
        
        # Build a maximal C4-free subgraph (greedy)
        for u, v in all_pairs:
            common = (neighbors[u] & neighbors[v]).bit_count()
            if common <= 1:
                neighbors[u] |= (1 << v)
                neighbors[v] |= (1 << u)
                edge_count += 1
                current_edges.append((u, v))
                edge_set.add((u, v))
                
                if edge_count == 90:
                    break
        
        if edge_count < 90:
            continue
            
        # Evaluate the cost of adding the 91st edge
        min_cost = float('inf')
        best_add_edge = None
        
        for u, v in all_pairs:
            if (u, v) in edge_set:
                continue
            common = (neighbors[u] & neighbors[v]).bit_count()
            cost = common * (common - 1) // 2
            if cost < min_cost:
                min_cost = cost
                best_add_edge = (u, v)
                
        if min_cost < best_score:
            best_score = min_cost
            best_edges = current_edges[:] + [best_add_edge]
            
            if best_score == 1:
                # Theoretical minimum for 91 edges (if 0 is impossible)
                # We can break early if we are confident 1 is the min
                # But let's run a few more to be sure? No, 1 is likely the min.
                break

    if best_edges is None:
        # Fallback: just return first 91 pairs (bad score, but valid)
        return [[i, j] for i, j in all_pairs[:91]]
        
    return [[u, v] for u, v in best_edges]

# Clear static state
if hasattr(construct, '_best_edges'):
    del construct._best_edges
