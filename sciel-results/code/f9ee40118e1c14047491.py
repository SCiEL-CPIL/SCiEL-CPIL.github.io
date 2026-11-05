import time
import random
from itertools import combinations

def _solve_set_cover(n):
    """
    Find a set of 5-uniform edges on n vertices such that every 2-coloring
    has at least one monochromatic edge.
    Uses Greedy Set Cover with reduction.
    """
    total_colorings = 1 << n
    
    # Precompute all 5-uniform edges as bitmasks
    edges = []
    for e in combinations(range(n), 5):
        mask = 0
        for v in e:
            mask |= (1 << v)
        edges.append(mask)
    
    # For each edge, compute the bitmask of colorings it covers
    # A coloring c (bitmask) covers edge e if (e & c) == e or (e & c) == 0
    edge_coverage = []
    for e_mask in edges:
        cover_mask = 0
        # Iterate through all colorings
        # Optimization: We can generate this more efficiently, but n is small.
        # For n=10, 1024 colorings. 252 edges. 252 * 1024 = 258k ops. Very fast.
        for c in range(total_colorings):
            if (e_mask & c) == e_mask or (e_mask & c) == 0:
                cover_mask |= (1 << c)
        edge_coverage.append(cover_mask)
        
    # Greedy Set Cover
    uncovered = (1 << total_colorings) - 1
    chosen_indices = []
    used = [False] * len(edges)
    
    # Optimization: Precompute popcount? Python 3.8+ has int.bit_count()
    
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
            # Should not happen if hypergraph is complete, but safe check
            break
            
        chosen_indices.append(best_idx)
        used[best_idx] = True
        uncovered &= ~edge_coverage[best_idx]
        
    # Reduction Phase
    # Try to remove edges from the chosen set
    final_indices = chosen_indices[:]
    for i in range(len(final_indices) - 1, -1, -1):
        idx_to_remove = final_indices[i]
        # Check if removing this edge still covers everything
        # We need to check if the union of the other chosen edges covers all colorings
        # This is O(k^2) if done naively.
        # Since k is small (~50-100), we can just recompute the union.
        
        remaining_union = 0
        for j in range(len(final_indices)):
            if j != i:
                remaining_union |= edge_coverage[final_indices[j]]
                
        if remaining_union == (1 << total_colorings) - 1:
            # It is safe to remove
            final_indices.pop(i)
            
    # Convert indices back to edge lists
    result_edges = [list(e) for e, idx in zip(combinations(range(n), 5), range(len(edges))) if idx in final_indices]
    # Wait, the zip logic above is wrong because final_indices are indices into the 'edges' list, 
    # which corresponds to combinations.
    
    # Correct conversion:
    all_combos = list(combinations(range(n), 5))
    result_edges = [list(all_combos[idx]) for idx in final_indices]
    
    return result_edges

def construct():
    deadline = time.time() + 145
    best_edges = None
    best_n = 0
    
    # Try n from 10 to 14
    # n=9 is K9^5 which has 126 edges, likely reducible but Set Cover on 2^9=512 is fast.
    # Let's start from n=9.
    for n in range(9, 15):
        if time.time() > deadline - 10:
            break
            
        try:
            edges = _solve_set_cover(n)
            if best_edges is None or len(edges) < len(best_edges):
                best_edges = edges
                best_n = n
        except Exception:
            continue
            
    # Fallback if Set Cover fails or takes too long
    if best_edges is None:
        # Use a known dense construction or random
        n = 9
        best_edges = [list(e) for e in combinations(range(n), 5)]
        best_n = n

    return {'n': best_n, 'edges': best_edges}
