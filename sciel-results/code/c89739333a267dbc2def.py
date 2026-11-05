import time
import random
from itertools import combinations

def _solve_set_cover(n):
    """
    Find a set of 5-uniform edges on n vertices such that every 2-coloring
    has at least one monochromatic edge.
    Uses Greedy Set Cover with reduction.
    Returns (list_of_edge_indices, list_of_coverage_masks, full_mask, all_combos)
    """
    total_colorings = 1 << n
    
    # Precompute all 5-uniform edges as bitmasks
    all_combos = list(combinations(range(n), 5))
    edges = []
    for e in all_combos:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edges.append(mask)
        
    edge_coverage = []
    full_mask = (1 << total_colorings) - 1
    
    for e_mask in edges:
        cover_mask = 0
        for c in range(total_colorings):
            if (e_mask & c) == e_mask or (e_mask & c) == 0:
                cover_mask |= (1 << c)
        edge_coverage.append(cover_mask)
        
    # Greedy Set Cover
    uncovered = full_mask
    chosen_indices = []
    used = [False] * len(edges)
    
    while uncovered:
        best_idx = -1
        best_count = -1
        for i in range(len(edges)):
            if used[i]:
                continue
            newly_covered = edge_coverage[i] & uncovered
            count = newly_covered.bit_count()
            if count > best_count:
                best_count = count
                best_idx = i
        
        if best_idx == -1:
            break
            
        chosen_indices.append(best_idx)
        used[best_idx] = True
        uncovered &= ~edge_coverage[best_idx]
        
    # Initial Reduction
    final_indices = chosen_indices[:]
    
    # We return the components for local search
    return final_indices, edge_coverage, full_mask, all_combos

def _local_search_set_cover(final_indices, edge_coverage, full_mask, all_combos, n, deadline):
    """
    Local search to minimize the number of edges.
    """
    # Work with a set of indices for O(1) lookup
    selected = set(final_indices)
    current_union = 0
    for idx in selected:
        current_union |= edge_coverage[idx]
        
    # Ensure it's a valid solution
    if current_union != full_mask:
        return list(selected)
        
    all_edge_indices = set(range(len(edge_coverage)))
    
    # Removal pass
    def try_removal(selected, current_union):
        removed = False
        # Iterate over a copy
        for idx in list(selected):
            if (current_union & ~edge_coverage[idx]) == full_mask:
                selected.remove(idx)
                current_union &= ~edge_coverage[idx]
                removed = True
        return selected, current_union, removed

    # Swap pass: Try to swap 1 out, 1 in
    def try_swap(selected, current_union):
        swapped = False
        # Randomize order for diversity
        out_candidates = list(selected)
        random.shuffle(out_candidates)
        in_candidates = list(all_edge_indices - selected)
        random.shuffle(in_candidates)
        
        for idx_out in out_candidates:
            if time.time() > deadline:
                break
            # Calculate holes if we remove idx_out
            U_out = current_union & ~edge_coverage[idx_out]
            holes = full_mask & ~U_out
            if holes == 0:
                continue
            
            for idx_in in in_candidates:
                # Check if idx_in covers all holes
                if (holes & ~edge_coverage[idx_in]) == 0:
                    # Valid swap
                    selected.remove(idx_out)
                    selected.add(idx_in)
                    current_union = full_mask
                    swapped = True
                    break
            if swapped:
                break
                
        return selected, current_union, swapped

    best_size = len(selected)
    
    while time.time() < deadline:
        # 1. Removal
        selected, current_union, rem = try_removal(selected, current_union)
        if rem:
            best_size = min(best_size, len(selected))
            continue # Try removal again
            
        # 2. Swap
        selected, current_union, swp = try_swap(selected, current_union)
        if swp:
            # After swap, try removal again
            selected, current_union, rem2 = try_removal(selected, current_union)
            if rem2:
                best_size = min(best_size, len(selected))
            continue
            
        # 3. Perturb: Add a random edge, then do removals
        in_candidates = list(all_edge_indices - selected)
        if in_candidates:
            idx_in = random.choice(in_candidates)
            selected.add(idx_in)
            current_union |= edge_coverage[idx_in]
            selected, current_union, rem3 = try_removal(selected, current_union)
            if len(selected) < best_size:
                best_size = len(selected)
        else:
            # No edges left to add? Should not happen for n>=5
            break

    return list(selected)

def construct():
    deadline = time.time() + 145
    best_edges = None
    best_n = 0
    best_count = float('inf')
    
    # Try n from 9 to 14
    for n in range(9, 15):
        if time.time() > deadline - 20:
            break
            
        try:
            final_indices, edge_coverage, full_mask, all_combos = _solve_set_cover(n)
            
            # Local Search
            optimized_indices = _local_search_set_cover(
                final_indices, edge_coverage, full_mask, all_combos, n, deadline - 5
            )
            
            if len(optimized_indices) < best_count:
                best_count = len(optimized_indices)
                best_edges = [list(all_combos[idx]) for idx in optimized_indices]
                best_n = n
        except Exception:
            continue
            
    # Fallback
    if best_edges is None:
        n = 9
        best_edges = [list(e) for e in combinations(range(n), 5)]
        best_n = n

    return {'n': best_n, 'edges': best_edges}
