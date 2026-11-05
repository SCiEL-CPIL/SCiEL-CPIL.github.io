import time
import random
from itertools import combinations

def _solve_set_cover(n, deadline):
    """
    Find a set of 5-uniform edges on n vertices such that every 2-coloring
    has at least one monochromatic edge, using Greedy Set Cover and Reduction.
    """
    total_colorings = 1 << n
    full_mask = (1 << total_colorings) - 1
    
    # Generate all 5-uniform edges
    all_edges = list(combinations(range(n), 5))
    num_edges = len(all_edges)
    
    # Precompute coverage mask for each edge
    # Bit c of mask is 1 if coloring c makes this edge monochromatic
    edge_cover_masks = [0] * num_edges
    
    # For each edge, compute which colorings are monochromatic
    # Coloring c is an integer. Vertex v is 0 if (c >> v) & 1 == 0, else 1.
    # Edge e is monochromatic if all vertices are 0 or all are 1.
    # This is equivalent to: (c & mask_e) == 0 or (c & mask_e) == mask_e
    
    for i, e in enumerate(all_edges):
        mask_e = 0
        for v in e:
            mask_e |= (1 << v)
        
        cover = 0
        # Iterate through all colorings
        # Optimization: We can iterate through the 2^(n-5) assignments of the free vertices
        # and set the corresponding bits.
        # However, for n <= 12, iterating 2^n is fine (4096 max).
        # For n=13, 8192. n=14, 16384.
        # Let's just iterate all colorings for simplicity and safety, it's fast enough for n<=12.
        # If n is larger, we might need optimization, but let's cap n at 12 or 13 to be safe with time.
        
        for c in range(total_colorings):
            if (c & mask_e) == 0 or (c & mask_e) == mask_e:
                cover |= (1 << c)
        edge_cover_masks[i] = cover
        
        if time.time() > deadline:
            return None

    # Greedy Set Cover
    uncovered = full_mask
    chosen_indices = []
    used = [False] * num_edges
    
    while uncovered:
        best_idx = -1
        best_count = -1
        
        # Optimization: We can skip edges that don't cover any new colorings
        for i in range(num_edges):
            if used[i]:
                continue
            newly_covered = edge_cover_masks[i] & uncovered
            count = newly_covered.bit_count()
            if count > best_count:
                best_count = count
                best_idx = i
        
        if best_idx == -1:
            # Should not happen if the hypergraph is complete (K_n^5 covers all)
            # But if we run out of useful edges, we break.
            break
            
        chosen_indices.append(best_idx)
        used[best_idx] = True
        uncovered &= ~edge_cover_masks[best_idx]
        
        if time.time() > deadline:
            return None

    # Reduction Phase
    # Try to remove edges from the chosen set
    final_indices = chosen_indices[:]
    
    # We iterate and try to remove each edge.
    # If removing an edge still leaves full coverage, we remove it.
    # We do this in a loop until no more edges can be removed.
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(final_indices) - 1, -1, -1):
            idx_to_remove = final_indices[i]
            
            # Check if removing this edge still covers everything
            # We need to check if the union of the other chosen edges covers all colorings
            remaining_union = 0
            for j in range(len(final_indices)):
                if j != i:
                    remaining_union |= edge_cover_masks[final_indices[j]]
                    
            if remaining_union == full_mask:
                # It is safe to remove
                final_indices.pop(i)
                changed = True
                # Restart loop or continue? Restarting is safer for finding more removals.
                # But checking from the end is okay. Let's break inner loop and restart outer.
                break

    if time.time() > deadline:
        return None

    # Convert indices back to edge lists
    result_edges = [list(all_edges[idx]) for idx in final_indices]
    return result_edges

def _try_local_swap(n, edges, deadline):
    """
    Try to improve the edge set by swapping edges or removing edges via local search.
    This is a heuristic refinement.
    """
    total_colorings = 1 << n
    full_mask = (1 << total_colorings) - 1
    
    # Precompute masks for current edges
    current_masks = []
    for e in edges:
        mask_e = 0
        for v in e:
            mask_e |= (1 << v)
        cover = 0
        for c in range(total_colorings):
            if (c & mask_e) == 0 or (c & mask_e) == mask_e:
                cover |= (1 << c)
        current_masks.append(cover)
        
    # Precompute masks for all possible edges
    all_edges = list(combinations(range(n), 5))
    all_masks = []
    for e in all_edges:
        mask_e = 0
        for v in e:
            mask_e |= (1 << v)
        cover = 0
        for c in range(total_colorings):
            if (c & mask_e) == 0 or (c & mask_e) == mask_e:
                cover |= (1 << c)
        all_masks.append(cover)
        
    # Map edge tuple to its mask index in all_edges
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each edge
        for i in range(len(edges)):
            if time.time() > deadline:
                return edges
            # Check if removing edge i still covers all
            union = 0
            for j in range(len(edges)):
                if j != i:
                    union |= current_masks[j]
            if union == full_mask:
                # Remove it
                edges.pop(i)
                current_masks.pop(i)
                improved = True
                break # Restart removal loop
        
        # Try swapping: replace edge i with edge k (not in edges) if it allows further reduction?
        # Simpler: just try to find a smaller set by replacing an edge with one that covers more "unique" parts?
        # The reduction phase above is quite powerful. 
        # Let's add a random restart or swap if stuck.
        
        if not improved:
             # Try random swaps
             for _ in range(10):
                 if time.time() > deadline:
                     return edges
                 if len(edges) < 2:
                     break
                 i = random.randint(0, len(edges) - 1)
                 # Pick a random edge from all_edges not in current set
                 candidate_idx = random.randint(0, len(all_edges) - 1)
                 if tuple(edges[i]) == all_edges[candidate_idx]:
                     continue
                 
                 # Try swap
                 new_edges = edges[:]
                 new_edges[i] = list(all_edges[candidate_idx])
                 new_mask = all_masks[candidate_idx]
                 
                 # Check if new set covers all
                 union = 0
                 for j in range(len(new_edges)):
                     if j == i:
                         union |= new_mask
                     else:
                         union |= current_masks[j]
                         
                 if union == full_mask:
                     # Valid swap. Does it help? 
                     # We don't know if it helps until we run reduction again.
                     # Let's just accept it and let the reduction phase handle it in the next iteration.
                     edges = new_edges
                     current_masks[i] = new_mask
                     improved = True
                     break

    return edges

def construct():
    deadline = time.time() + 145.0
    best_edges = None
    best_n = 0
    min_edges_found = float('inf')
    
    # Try n from 9 to 14
    # n=9: C(9,5)=126 edges. 2^9=512 colorings.
    # n=10: C(10,5)=252 edges. 2^10=1024 colorings.
    # n=11: C(11,5)=462 edges. 2^11=2048 colorings.
    # n=12: C(12,5)=792 edges. 2^12=4096 colorings.
    # n=13: C(13,5)=1287 edges. 2^13=8192 colorings.
    
    for n in range(9, 15):
        if time.time() > deadline - 5.0:
            break
            
        try:
            edges = _solve_set_cover(n, deadline - 5.0)
            if edges:
                # Refine with local search
                edges = _try_local_swap(n, edges, deadline - 2.0)
                
                if edges and (best_edges is None or len(edges) < min_edges_found):
                    min_edges_found = len(edges)
                    best_edges = edges
                    best_n = n
        except Exception as e:
            # If it fails, continue to next n
            pass

    # Fallback: if nothing found, use a dense construction
    if best_edges is None:
        n = 9
        all_edges = list(combinations(range(n), 5))
        best_edges = [list(e) for e in all_edges]
        best_n = n

    return {'n': best_n, 'edges': best_edges}
