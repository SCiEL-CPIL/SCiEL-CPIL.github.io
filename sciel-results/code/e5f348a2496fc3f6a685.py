import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s buffer for safety

    # We'll work with n=9 (K_9^5 has 126 edges, which is non-2-colorable by pigeonhole)
    n = 9
    all_edges = [list(e) for e in combinations(range(n), 5)]
    num_edges = len(all_edges)  # 126

    # Precompute edge masks (bitmask of vertices in each edge)
    edge_masks = []
    for e in all_edges:
        mask_e = 0
        for v in e:
            mask_e |= (1 << v)
        edge_masks.append(mask_e)

    # Precompute bad_masks: for each coloring c (0..2^9-1), bad_masks[c] is a bitmask
    # of edges that are monochromatic under coloring c
    bad_masks = [0] * (1 << n)
    for c in range(1 << n):
        bm = 0
        for i in range(num_edges):
            mask_e = edge_masks[i]
            inter = c & mask_e
            if inter == 0 or inter == mask_e:
                bm |= (1 << i)
        bad_masks[c] = bm

    # Quick check: if any coloring has no monochromatic edges, the full graph is colorable
    # In that case, no subset can be non-colorable? 
    # Wait, if the full graph is colorable, a subset is also colorable.
    # So if any bad_masks[c] == 0, we can skip this n entirely.
    if any(bm == 0 for bm in bad_masks):
        # This shouldn't happen for n=9, but just in case
        return {'n': n, 'edges': all_edges}

    def is_non_colorable(active_mask):
        """Check if the hypergraph with edges in active_mask is non-2-colorable."""
        for bm in bad_masks:
            if (bm & active_mask) == 0:
                return False
        return True

    # Greedy reduction: start with all edges, try to remove each edge
    current_mask = (1 << num_edges) - 1
    
    # Verify start is non-colorable
    if not is_non_colorable(current_mask):
        return {'n': n, 'edges': all_edges}

    # Multiple passes of greedy removal
    # Each pass: try to remove each edge in random order
    best_mask = current_mask
    best_count = num_edges
    
    # We'll do several passes with different random orderings
    max_passes = 50
    
    for _ in range(max_passes):
        if time.time() > deadline:
            break
            
        changed = True
        while changed:
            changed = False
            # Try removing edges in random order
            indices = list(range(num_edges))
            random.shuffle(indices)
            for i in indices:
                if time.time() > deadline:
                    break
                # Try removing edge i
                if current_mask & (1 << i):
                    new_mask = current_mask & ~(1 << i)
                    if is_non_colorable(new_mask):
                        current_mask = new_mask
                        changed = True
                        
        count = bin(current_mask).count('1')
        if count < best_count:
            best_count = count
            best_mask = current_mask
            
        # Early exit if we've reached a very good count
        if best_count <= 51:
            break

    # Extract edges from best_mask
    best_edges = [all_edges[i] for i in range(num_edges) if best_mask & (1 << i)]
    
    return {'n': n, 'edges': best_edges}
