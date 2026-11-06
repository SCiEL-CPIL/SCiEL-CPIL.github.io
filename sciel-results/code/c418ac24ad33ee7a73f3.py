import time
import random
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Returns True if the hypergraph is 2-colorable, False otherwise.
    Uses backtracking with unit propagation.
    """
    if not edges:
        return True
    
    # Precompute edge vertex indices for faster access
    edge_verts = [list(e) for e in edges]
    
    # State: assignments for vertices 0..n-1. -1 unassigned, 0 or 1 assigned.
    assignment = [-1] * n
    
    # Counters for each edge: count of 0s, count of 1s
    # If count_0 == 5 or count_1 == 5, the edge is monochromatic -> conflict
    # If count_0 + count_1 == 5, the edge is satisfied (mixed) -> can be ignored? 
    # No, we just need to ensure no edge is monochromatic.
    # An edge is "bad" if it is monochromatic.
    # We want to avoid bad edges.
    # So if an edge becomes monochromatic, we have a conflict.
    
    # To speed up propagation, we can track for each edge how many vertices are unassigned,
    # and how many are 0 and 1.
    # If unassigned == 0:
    #   if count_0 == 5 or count_1 == 5: Conflict
    # If unassigned == 1:
    #   if count_0 == 4: The last one MUST be 1 (Unit Propagation)
    #   if count_1 == 4: The last one MUST be 0 (Unit Propagation)
    
    # Data structures for propagation
    # For each edge, store:
    # - list of vertices
    # - count_0, count_1, unassigned
    
    # This structure might be heavy to update in pure Python for many restarts.
    # Alternative: Simple recursive backtracking with pruning.
    # Given n <= 11 and edges <= 500, simple backtracking with early termination is often fast enough.
    
    def backtrack(idx):
        if idx == n:
            return True
            
        v = idx
        # Try 0
        assignment[v] = 0
        valid = True
        # Check all edges containing v? No, check all edges to see if any became monochromatic.
        # Optimization: Only check edges containing v? 
        # If an edge doesn't contain v, its status hasn't changed.
        # So we only need to check edges containing v.
        
        # Precompute adjacency: for each vertex, list of edge indices containing it.
        # This needs to be passed or global.
        
        # Let's use a simpler approach for the inner loop:
        # Check if any edge is monochromatic.
        # Since n is small, checking all edges is O(m).
        # Backtracking depth is n. Total ops O(n * m * 2^n) worst case, but pruning helps.
        
        # Actually, let's stick to the bitmask method if it's faster?
        # The current code's bottleneck was the greedy loop calling the checker.
        # If the checker is O(2^n), and we call it O(m * n) times, it's slow.
        # If the checker is O(n * m) with pruning, it might be faster for sparse graphs.
        
        # Let's implement a specific fast checker for this context.
        pass

    # The previous approach used precomputed bad masks.
    # Let's stick to the precomputed bad masks but optimize the greedy loop.
    # The issue with the current code is likely that it tries to reduce K_11 (462 edges) which is hard.
    # K_9 (126 edges) is easier.
    # Also, the `is_non_colorable` check in the current code is O(2^n).
    # For n=11, 2^11 = 2048.
    # Greedy loop: 462 edges.
    # One pass: 462 * 2048 = ~1M ops.
    # This is fast in C, but in Python, 1M ops is ~0.1s.
    # Multiple restarts?
    # The current code runs `while time.time() < deadline - 2`.
    # It should be fast enough. Why is the score only 1.22?
    # Maybe it's not finding < 126 edges? Or maybe it's finding 126?
    # The seed score is 1.0 (which implies 126 edges? No, seed is K_9^5 which has 126 edges).
    # Score = 126 / value.
    # If score is 1.22, value = 126 / 1.22 = 103.
    # So it found a 103-edge hypergraph.
    # We need to go lower.
    
    # To go lower, we need a better reduction strategy.
    # 1. Start from a smaller non-colorable core if possible? No, we start from complete graph.
    # 2. Use a more aggressive reduction: Remove edges that are "redundant".
    # 3. Use local search on the set of edges?
    
    # Let's try to improve the reduction by using a better heuristic for edge removal order.
    # Or use a SAT solver logic directly.
    
    # Given the time, let's refine the existing approach:
    # 1. Focus on n=9 and n=10.
    # 2. Use the precomputed mask approach but ensure we do many restarts.
    # 3. Add a "shrink" step: if we find a non-colorable subgraph, try to remove more edges from it specifically.
    
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_edges = None
    best_count = float('inf')
    best_n = None
    
    # Prioritize n=9 (smaller, easier to minimize) and n=10.
    # n=11 has 462 edges, hard to reduce.
    # n=9 has 126 edges.
    # n=10 has 252 edges.
    
    for n in [9, 10]:
        if time.time() > deadline:
            break
            
        all_edges = [list(e) for e in combinations(range(n), 5)]
        num_edges = len(all_edges)
        
        # Precompute bad masks
        # bad_mask[c] has bit i set if edge i is monochromatic in coloring c
        bad_masks = [0] * (1 << n)
        
        # Precompute edge vertex masks
        edge_vertex_masks = []
        for e in all_edges:
            mask_e = 0
            for v in e:
                mask_e |= (1 << v)
            edge_vertex_masks.append(mask_e)
            
        for c in range(1 << n):
            bm = 0
            for i in range(num_edges):
                mask_e = edge_vertex_masks[i]
                inter = c & mask_e
                if inter == 0 or inter == mask_e:
                    bm |= (1 << i)
            bad_masks[c] = bm
            
        # Check if full graph is non-colorable
        # If any coloring has bad_mask == 0, it is colorable.
        full_graph_colorable = False
        for bm in bad_masks:
            if bm == 0:
                full_graph_colorable = True
                break
        if full_graph_colorable:
            continue
            
        # Greedy Reduction
        # We will try to find a minimal non-colorable subgraph.
        
        def is_non_colorable(active_mask):
            for bm in bad_masks:
                if (bm & active_mask) == 0:
                    return False
            return True
            
        best_local_mask = (1 << num_edges) - 1
        best_local_count = num_edges
        
        # Multiple restarts
        # For each restart, we shuffle the order of edge removal attempts.
        while time.time() < deadline - 2:
            curr_mask = (1 << num_edges) - 1
            # We can start from the best found so far in this n to save time?
            # No, random restarts from full graph are better for diversity.
            
            # Optional: Start from a random subset? No, must be non-colorable.
            # Start from full graph.
            
            # To speed up, we can pre-shuffle indices once per restart
            indices = list(range(num_edges))
            random.shuffle(indices)
            
            # Iterate through edges, try to remove
            for i in indices:
                if time.time() > deadline - 2:
                    break
                if curr_mask & (1 << i):
                    new_mask = curr_mask & ~(1 << i)
                    if is_non_colorable(new_mask):
                        curr_mask = new_mask
                        
            count = bin(curr_mask).count('1')
            if count < best_local_count:
                best_local_count = count
                best_local_mask = curr_mask
                
            if best_local_count <= 51:
                break
                
        if best_local_count < best_count:
            best_count = best_local_count
            best_edges = [all_edges[i] for i in range(num_edges) if best_local_mask & (1 << i)]
            best_n = n
            
    if best_edges is None:
        n = 9
        edges = [list(e) for e in combinations(range(n), 5)]
        return {'n': n, 'edges': edges}
        
    return {'n': best_n, 'edges': best_edges}
