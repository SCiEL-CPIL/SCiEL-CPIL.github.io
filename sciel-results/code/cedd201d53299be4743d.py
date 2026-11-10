import time
import random
from itertools import combinations

def construct():
    """
    Finds a 5-uniform hypergraph with the minimum number of edges 
    that is not 2-colorable, using a randomized greedy deletion 
    strategy on the complete 5-uniform hypergraph K_9^5.
    """
    n = 9
    all_edges = list(combinations(range(n), 5))
    num_edges = len(all_edges)
    
    # Bitmask representation for edges
    # Vertex i corresponds to bit i
    edge_masks = []
    for e in all_edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    def is_non_2_colorable(edge_indices):
        """
        Checks if the hypergraph defined by the set of edge indices 
        is NOT 2-colorable.
        Returns True if non-2-colorable, False if 2-colorable.
        Uses backtracking with unit propagation.
        """
        if not edge_indices:
            return False
        
        # Map vertex to list of edges incident to it
        # This helps with unit propagation
        # However, for small n=9, simple backtracking is fast enough.
        # We use a recursive backtracking with memoization or just standard DFS.
        # State: (vertex_index, colors_mask)
        # colors_mask: bit i is 1 if vertex i is colored 1, 0 if colored 0, 
        #              but we need to handle uncolored. 
        # Let's use two masks: color0 and color1.
        
        # Optimized Backtracking
        # We assign colors to vertices 0..n-1.
        # If an edge becomes monochromatic, it's a conflict.
        
        def backtrack(v_idx, c0, c1):
            if v_idx == n:
                return False # Successfully colored (contradiction to non-2-colorable)
            
            # Try color 0
            valid_c0 = True
            # Check edges containing v_idx
            for e_idx in edge_indices:
                mask = edge_masks[e_idx]
                if mask & (1 << v_idx):
                    # Check if all other vertices in this edge are in c0
                    # If so, adding v_idx to c0 makes it monochromatic
                    rest_mask = mask & ~(1 << v_idx)
                    if rest_mask and (rest_mask & ~c0) == 0:
                        valid_c0 = False
                        break
            if valid_c0:
                if not backtrack(v_idx + 1, c0 | (1 << v_idx), c1):
                    return False
            
            # Try color 1
            valid_c1 = True
            for e_idx in edge_indices:
                mask = edge_masks[e_idx]
                if mask & (1 << v_idx):
                    rest_mask = mask & ~(1 << v_idx)
                    if rest_mask and (rest_mask & ~c1) == 0:
                        valid_c1 = False
                        break
            if valid_c1:
                if not backtrack(v_idx + 1, c0, c1 | (1 << v_idx)):
                    return False
            
            return True # Both colors lead to failure? No, if both fail, it's non-2-colorable?
            # Wait. 
            # If valid_c0 is False, we can't use color 0.
            # If valid_c1 is False, we can't use color 1.
            # If both are False, we have a conflict at v_idx. This branch fails.
            # If one is valid and leads to a complete coloring, the function returns False (meaning "It IS 2-colorable").
            # We want to know if it is NON-2-colorable.
            # So if we find ANY valid coloring, return False.
            # If NO valid coloring exists, return True.
            
            # Let's re-verify the logic.
            # backtrack returns True if NO valid coloring exists from this state.
            # backtrack returns False if A valid coloring exists from this state.
            
            # If valid_c0 is True and backtrack(..., c0, c1) returns False (coloring found), then overall returns False.
            # If valid_c0 is True and backtrack(..., c0, c1) returns True (no coloring), we try c1.
            # If valid_c1 is True and backtrack(..., c0, c1) returns False (coloring found), then overall returns False.
            # If both branches return True (no coloring), then overall returns True.
            
            # My previous logic:
            # if valid_c0:
            #    if not backtrack(...): return False
            # if valid_c1:
            #    if not backtrack(...): return False
            # return True
            
            # This is correct.
            
        # Initial state: no vertices colored
        return backtrack(0, 0, 0)

    best_edges = None
    best_size = num_edges
    start_time = time.time()
    time_limit = 110  # Safety margin
    
    # Random restarts
    while time.time() - start_time < time_limit:
        current_edges = set(range(num_edges))
        # Shuffle edges to randomize deletion order
        shuffled = list(range(num_edges))
        random.shuffle(shuffled)
        
        for e_idx in shuffled:
            if e_idx not in current_edges:
                continue
            
            # Try to remove e_idx
            new_edges = current_edges - {e_idx}
            
            # Check if new_edges is non-2-colorable
            if is_non_2_colorable(list(new_edges)):
                current_edges = new_edges
                # If we reached 51, we can stop this restart early? 
                # No, we want to minimize further. 
                # But if we know 51 is optimal, we can stop.
                # Let's just continue to see if we can go lower (though unlikely).
        
        if len(current_edges) < best_size:
            best_size = len(current_edges)
            best_edges = list(current_edges)
            
            # If we found 51, we can stop searching for smaller ones 
            # assuming 51 is the theoretical lower bound for this n.
            # The task says "best known construction has 51 edges".
            # If we find 51, it's likely optimal.
            if best_size == 51:
                break
    
    if best_edges is None:
        # Fallback to complete graph
        best_edges = list(range(num_edges))
        
    result_edges = [list(all_edges[i]) for i in best_edges]
    
    return {'n': n, 'edges': result_edges}

# Pre-run to ensure random seed doesn't matter too much or to warm up?
# No, just return the function.
