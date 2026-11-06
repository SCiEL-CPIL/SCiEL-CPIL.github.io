import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    n = 16
    target_edges = 65
    
    # Precompute all possible edges
    all_edges = list(combinations(range(n), 2))
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    num_possible_edges = len(all_edges)
    
    # Map edge index to (u, v)
    edge_u = [e[0] for e in all_edges]
    edge_v = [e[1] for e in all_edges]
    
    # Lookup table for bit count (0 to 2^16 - 1)
    # 65536 entries, fast to build
    bit_count = [0] * (1 << n)
    for i in range(1, 1 << n):
        bit_count[i] = bit_count[i >> 1] + (i & 1)
        
    # Also precompute index of lowest set bit? 
    # lsb.bit_length() - 1 is fast enough.
    
    def build_adj_mask(edge_set):
        masks = [0] * n
        for idx in edge_set:
            u = edge_u[idx]
            v = edge_v[idx]
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
        return masks
    
    def calc_score(masks):
        """
        Calculate max degree sum over triangles.
        masks: list of ints (bitmasks)
        """
        deg = [bit_count[m] for m in masks]
        max_s = 0
        # Iterate over all possible edges (u, v)
        # If (u, v) is in graph, check common neighbors
        for u in range(n):
            mask_u = masks[u]
            # Only consider v > u to avoid double counting? 
            # No, triangle (u,v,w) is counted once if we fix u < v < w.
            # But here we iterate edges. An edge (u,v) is part of triangles (u,v,w).
            # We can just iterate all edges in the graph.
            # To avoid iterating all 120 possible edges, we only iterate existing ones.
            # But we don't have a list of existing edges easily from masks without converting back.
            # Let's iterate all pairs u < v.
            for v in range(u + 1, n):
                if (mask_u >> v) & 1:
                    # Edge (u, v) exists
                    common = mask_u & masks[v]
                    # We only care about w > v to ensure u < v < w
                    common &= ~((1 << (v + 1)) - 1)
                    
                    # Iterate set bits in common
                    c = common
                    while c:
                        lsb = c & (-c)
                        w = lsb.bit_length() - 1
                        s = deg[u] + deg[v] + deg[w]
                        if s > max_s:
                            max_s = s
                        c ^= lsb
        return max_s

    # Initial configuration: K_{8,8} + one internal edge (0, 1)
    init_edge_set = set()
    for u in range(8):
        for v in range(8, 16):
            init_edge_set.add(edge_to_idx[(u, v)])
    init_edge_set.add(edge_to_idx[(0, 1)])
    
    current_edge_set = set(init_edge_set)
    current_masks = build_adj_mask(current_edge_set)
    current_score = calc_score(current_masks)
    
    best_score = current_score
    best_edge_set = set(current_edge_set)
    
    rng = random.Random(42)
    
    # Simulated Annealing Parameters
    T_start = 2.0
    T_end = 0.01
    cooling_rate = 0.999
    
    # We need a list of edges in the graph to pick one to remove
    # And a list of edges NOT in the graph to pick one to add
    
    def get_edge_lists(edge_set):
        in_graph = [i for i in range(num_possible_edges) if i in edge_set]
        not_in_graph = [i for i in range(num_possible_edges) if i not in edge_set]
        return in_graph, not_in_graph
    
    in_graph, not_in_graph = get_edge_lists(current_edge_set)
    
    max_iters = 500000
    T = T_start
    
    for i in range(max_iters):
        if time.time() > deadline - 5:
            break
            
        # Pick random edge to remove
        if not in_graph:
            break
        remove_idx = in_graph[rng.randrange(len(in_graph))]
        
        # Pick random edge to add
        if not not_in_graph:
            break
        add_idx = not_in_graph[rng.randrange(len(not_in_graph))]
        
        # Apply move
        u_rem = edge_u[remove_idx]
        v_rem = edge_v[remove_idx]
        u_add = edge_u[add_idx]
        v_add = edge_v[add_idx]
        
        # Update masks
        new_masks = current_masks[:]
        new_masks[u_rem] ^= (1 << v_rem)
        new_masks[v_rem] ^= (1 << u_rem)
        new_masks[u_add] |= (1 << v_add)
        new_masks[v_add] |= (1 << u_add)
        
        # Calculate new score
        new_score = calc_score(new_masks)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta < 0 or (T > T_end and rng.random() < np.exp(-delta / T)):
            # Accept
            current_edge_set = set(current_edge_set)
            current_edge_set.discard(remove_idx)
            current_edge_set.add(add_idx)
            
            # Update lists
            in_graph.remove(remove_idx)
            in_graph.append(add_idx)
            not_in_graph.remove(add_idx)
            not_in_graph.append(remove_idx)
            
            current_masks = new_masks
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_edge_set = set(current_edge_set)
        
        T *= cooling_rate
    
    # Final Polish: Local Search
    # Try all single swaps that improve the score
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        in_graph, not_in_graph = get_edge_lists(best_edge_set)
        current_edge_set = set(best_edge_set)
        current_masks = build_adj_mask(current_edge_set)
        current_score = best_score
        
        # To speed up, we can break early if we find a big improvement?
        # No, just find ANY improvement and restart the scan.
        for remove_idx in in_graph:
            if time.time() > deadline - 2:
                break
            u_rem = edge_u[remove_idx]
            v_rem = edge_v[remove_idx]
            
            # Try adding each non-existent edge
            for add_idx in not_in_graph:
                u_add = edge_u[add_idx]
                v_add = edge_v[add_idx]
                
                new_masks = current_masks[:]
                new_masks[u_rem] ^= (1 << v_rem)
                new_masks[v_rem] ^= (1 << u_rem)
                new_masks[u_add] |= (1 << v_add)
                new_masks[v_add] |= (1 << u_add)
                
                new_score = calc_score(new_masks)
                
                if new_score < current_score:
                    # Apply
                    current_edge_set.discard(remove_idx)
                    current_edge_set.add(add_idx)
                    current_masks = new_masks
                    current_score = new_score
                    best_score = new_score
                    best_edge_set = set(current_edge_set)
                    improved = True
                    break # Restart scan with new best
            if improved:
                break
        
    edges_out = [[all_edges[e][0], all_edges[e][1]] for e in best_edge_set]
    return {'n': n, 'edges': edges_out}
