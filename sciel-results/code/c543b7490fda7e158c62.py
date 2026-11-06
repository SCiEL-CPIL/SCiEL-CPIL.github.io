import numpy as np
from itertools import combinations

def construct():
    """
    Construct a 5-uniform hypergraph with no proper 2-colouring, minimizing edges.
    
    Approach: 
    1. Model as SAT: Vertex i is bit i. Edge e is monochromatic if (color & mask_e) == 0 or (color & mask_e) == mask_e.
    2. Greedy: Start with all 2^n colorings valid. Iteratively pick the edge that eliminates the most valid colorings.
    3. Use NumPy broadcasting to evaluate all edges against all valid colorings simultaneously for speed.
    4. Try small n (9, 10, 11) to find a minimal edge count.
    """
    
    def construct_for_n(n):
        # Generate all possible edges
        all_edges = list(combinations(range(n), 5))
        num_edges = len(all_edges)
        
        # Precompute masks for all edges as a NumPy array
        # Shape: (num_edges,)
        edge_masks = np.array([0] * num_edges, dtype=np.int64)
        for i, e in enumerate(all_edges):
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks[i] = mask
            
        # Initialize valid colorings as a NumPy array of all 2^n integers
        # For n=12, 2^12 = 4096.
        valid_colorings = np.arange(1 << n, dtype=np.int64)
        
        selected_edges = []
        used_edge_indices = set()
        
        # We will iterate until valid_colorings is empty
        # Max steps is num_edges
        for _ in range(num_edges):
            if len(valid_colorings) == 0:
                break
                
            # Filter out already used edges? 
            # Actually, we can just re-evaluate all edges and ignore used ones by setting their count to -1 or similar.
            # But creating a mask of unused edges is easy.
            
            # To speed up, we only consider unused edges.
            # However, broadcasting with a subset of edges is slightly more complex to index.
            # Given E is small (<500), we can just compute for all and mask the used ones.
            
            # V: (K, 1), M: (1, E)
            # inter: (K, E)
            # This might be memory heavy if K is large, but K decreases.
            # For n=12, K=4096, E=792. 4096*792 ~ 3.2M ints. 3.2M * 8 bytes = 25MB. Acceptable.
            
            V = valid_colorings[:, None]
            M = edge_masks[None, :]
            
            inter = V & M
            is_mono = (inter == 0) | (inter == M)
            
            # is_mono is (K, E) boolean array
            # Count how many valid colorings each edge kills
            counts = np.sum(is_mono, axis=0)
            
            # Mask out used edges
            # Create a boolean mask for unused edges
            unused_mask = np.ones(num_edges, dtype=bool)
            for idx in used_edge_indices:
                unused_mask[idx] = False
                
            # Set counts for used edges to -1 so they are never picked
            counts[~unused_mask] = -1
            
            best_edge_idx = np.argmax(counts)
            best_count = counts[best_edge_idx]
            
            if best_count <= 0:
                # No edge can reduce the solution space (or all are used)
                # This implies we stuck, but theoretically if valid_colorings > 0, 
                # there must be an edge that is monochromatic for at least one coloring?
                # Not necessarily. If the remaining colorings are all non-monochromatic on ALL unused edges, 
                # then we cannot eliminate them with unused edges. 
                # But wait, if a coloring is valid, it is NOT monochromatic on any *selected* edge.
                # It might be monochromatic on some *unused* edge.
                # If best_count is 0, it means no unused edge is monochromatic for ANY valid coloring.
                # This means we cannot proceed with greedy? 
                # Actually, if a coloring is valid, it satisfies the constraint "not mono" for all selected edges.
                # It doesn't constrain unused edges.
                # If no unused edge kills any valid coloring, then adding any unused edge won't help.
                # But we need to kill ALL valid colorings.
                # If we are stuck, greedy failed.
                break
                
            used_edge_indices.add(best_edge_idx)
            selected_edges.append(list(all_edges[best_edge_idx]))
            
            # Update valid_colorings: keep only those that are NOT monochromatic on the chosen edge
            # We already computed is_mono. 
            # is_mono[:, best_edge_idx] is the boolean mask of colorings killed by this edge.
            kill_mask = is_mono[:, best_edge_idx]
            valid_colorings = valid_colorings[~kill_mask]
            
        if len(valid_colorings) == 0:
            return {'n': n, 'edges': selected_edges}
        else:
            return None

    # Try n from small to larger
    # Smaller n might have fewer edges needed? 
    # For n=9, max edges is 126. 
    # For n=10, max edges is 252.
    # We want to MINIMIZE edges. 
    # It is known that m(5) is around 51? 
    # If we can find one with < 51 edges, we win.
    
    # Let's try n=9, 10, 11, 12
    for n in [9, 10, 11, 12]:
        res = construct_for_n(n)
        if res:
            return res
            
    # Fallback: Return a known valid hypergraph if greedy fails or is too slow
    # A complete graph K_9^5 has 126 edges. Definitely not 2-colorable.
    # But we want few edges. 
    # If greedy failed, we return a safe but potentially suboptimal result.
    # To ensure validity, we can use a known small non-2-colorable hypergraph if available, 
    # or just return the first one found by a less optimized method.
    # Given the time constraint, if the NumPy version fails, we return a dense one.
    
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
