import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    best_n = 0
    best_edges = []
    min_edge_count = float('inf')

    # Try small n values. Theoretical min edges is >= 29.
    # n=9 has 126 edges, 512 colorings.
    # n=10 has 252 edges, 1024 colorings.
    # n=11 has 462 edges, 2048 colorings.
    
    for n in range(9, 13):
        if time.time() > deadline:
            break
            
        vertices = list(range(n))
        all_edges = list(combinations(vertices, 5))
        num_colorings = 1 << n
        uncovered = (1 << num_colorings) - 1
        
        # Precompute edge masks
        edge_masks = []
        for e in all_edges:
            mask = 0
            edge_vertex_mask = 0
            for v in e:
                edge_vertex_mask |= (1 << v)
            
            # Optimization: Iterate colorings
            # A coloring c is monochromatic on e if (c & edge_vertex_mask) == 0 or == edge_vertex_mask
            for c in range(num_colorings):
                val = c & edge_vertex_mask
                if val == 0 or val == edge_vertex_mask:
                    mask |= (1 << c)
            edge_masks.append(mask)
            
        selected_edges = []
        # Greedy Set Cover
        while uncovered != 0:
            best_edge_idx = -1
            best_gain = 0
            for i, mask in enumerate(edge_masks):
                gain_mask = mask & uncovered
                if gain_mask:
                    gain = gain_mask.bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_edge_idx = i
            
            if best_edge_idx == -1:
                break
                
            selected_edges.append(all_edges[best_edge_idx])
            uncovered &= ~edge_masks[best_edge_idx]
            
            if time.time() > deadline:
                break
        
        # If we covered all colorings, it's a valid non-2-colorable hypergraph
        if uncovered == 0 and len(selected_edges) < min_edge_count:
            min_edge_count = len(selected_edges)
            best_n = n
            best_edges = selected_edges

    if best_edges:
        return {'n': best_n, 'edges': [list(e) for e in best_edges]}
    else:
        # Fallback: Return a known valid structure if greedy failed (unlikely)
        # Or just return n=9 full graph (126 edges) which is definitely non-2-colorable
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
