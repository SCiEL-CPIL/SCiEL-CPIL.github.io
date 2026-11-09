import time
import random
from itertools import combinations

_WALL_LIMIT = 150.0
_MARGIN = 10.0

def _is_non_2_colorable(edges, n):
    """
    Checks if the hypergraph is NOT 2-colorable using backtracking with unit propagation.
    Returns True if no proper 2-coloring exists.
    """
    if not edges:
        return False
        
    # Map vertex to list of edges it belongs to
    vertex_to_edges = {v: [] for v in range(n)}
    for idx, e in enumerate(edges):
        for v in e:
            vertex_to_edges[v].append(idx)
            
    # For unit propagation, we need to know if an edge is "satisfied" (monochromatic) 
    # or "conflict" (mixed). 
    # Wait, the condition for "non-2-colorable" is that EVERY coloring has a monochromatic edge.
    # So we are looking for a coloring where NO edge is monochromatic.
    # If we find one, it IS 2-colorable.
    # So this function should return False if a 2-coloring is found, True otherwise.
    
    # State: 
    # color[v] = 0 or 1 or -1 (unassigned)
    # We assign colors to vertices.
    # Constraint: For every edge, NOT (all 0) AND NOT (all 1).
    
    # This is equivalent to: For every edge, at least one 0 AND at least one 1.
    
    # Backtracking
    # To make it fast, we use unit propagation.
    # If an edge has 4 vertices colored 0, the 5th MUST be 1.
    # If an edge has 4 vertices colored 1, the 5th MUST be 0.
    
    def solve(coloring):
        # coloring: list of -1, 0, 1
        # Check for contradictions: An edge cannot be all 0s or all 1s.
        # Also, if an edge has 4 0s and 1 -1, force 1.
        
        # Propagation loop
        changed = True
        while changed:
            changed = False
            for idx, e in enumerate(edges):
                # Count colors in e
                c0 = 0
                c1 = 0
                unassigned = []
                for v in e:
                    if coloring[v] == 0:
                        c0 += 1
                    elif coloring[v] == 1:
                        c1 += 1
                    else:
                        unassigned.append(v)
                
                # Contradiction check
                if c0 == 5:
                    return False # This coloring path is invalid (edge is mono 0)
                if c1 == 5:
                    return False # This coloring path is invalid (edge is mono 1)
                
                # Unit propagation
                if c0 == 4 and len(unassigned) == 1:
                    v = unassigned[0]
                    if coloring[v] != -1:
                        if coloring[v] == 0:
                            return False # Contradiction: would make it 5 zeros
                    else:
                        coloring[v] = 1
                        changed = True
                elif c1 == 4 and len(unassigned) == 1:
                    v = unassigned[0]
                    if coloring[v] != -1:
                        if coloring[v] == 1:
                            return False
                    else:
                        coloring[v] = 0
                        changed = True
                elif c0 == 4 and len(unassigned) == 0:
                    return False
                elif c1 == 4 and len(unassigned) == 0:
                    return False

        # If no more propagation, pick an unassigned vertex
        unassigned_vertices = [v for v in range(n) if coloring[v] == -1]
        if not unassigned_vertices:
            # All assigned. Check if any edge is monochromatic.
            # Actually, the propagation above already caught "5 of same color".
            # But we must ensure no edge is monochromatic.
            # The propagation handles "4 assigned, 1 unassigned".
            # If all are assigned, we just need to verify no edge is mono.
            for e in edges:
                colors = [coloring[v] for v in e]
                if len(set(colors)) == 1:
                    return False
            return True # No proper 2-coloring found (i.e., it is NOT 2-colorable)
        
        # Heuristic: Pick vertex with highest degree in remaining constraints?
        # Just pick the first one for simplicity, or random.
        v = unassigned_vertices[0]
        
        # Try 0
        coloring[v] = 0
        if solve(coloring):
            return True
        coloring[v] = -1
        
        # Try 1
        coloring[v] = 1
        if solve(coloring):
            return True
        coloring[v] = -1
        
        return False # Backtrack
        
    coloring = [-1] * n
    return solve(coloring)

def construct():
    start = time.time()
    deadline = start + _WALL_LIMIT - _MARGIN
    rng = random.Random(12345)
    
    best_edges = None
    best_n = None
    best_size = float('inf')
    
    # Target n=11, as it's the smallest n where min edges is 51.
    # n=10 max edges 252, but min non-2-colorable is likely higher or n=10 doesn't have a 51-edge solution.
    # Actually, for n=10, the min number of edges for non-2-colorability is 25? No, 
    # The bound is m(5). For n=11, m(5) <= 51.
    
    n = 11
    all_edges = list(combinations(range(n), 5))
    num_edges_total = len(all_edges)
    
    # Precompute masks
    # Mask is a bitset of length 2^n
    # Bit c is 1 if coloring c is monochromatic for this edge.
    masks = []
    for e in all_edges:
        m = 0
        for c in range(1 << n):
            colors = [(c >> v) & 1 for v in e]
            if len(set(colors)) == 1:
                m |= (1 << c)
        masks.append(m)
        
    full_mask = (1 << (1 << n)) - 1
    
    # Greedy Set Cover with Randomization
    # We try to find a cover of size <= 51.
    
    def greedy_cover(masks, full_mask, deadline, rng, max_size_hint=51):
        uncovered = full_mask
        chosen = []
        chosen_indices = set()
        remaining = list(range(len(masks)))
        
        while uncovered:
            best_idx = -1
            best_count = 0
            # To improve quality, we can look at top-k candidates and pick randomly?
            # Or just standard greedy.
            # Standard greedy is fast.
            for idx in remaining:
                count = (uncovered & masks[idx]).bit_count()
                if count > best_count:
                    best_count = count
                    best_idx = idx
                    if best_count == uncovered.bit_count():
                        break
            if best_idx == -1:
                return None
            chosen.append(best_idx)
            chosen_indices.add(best_idx)
            uncovered &= ~masks[best_idx]
            remaining.remove(best_idx)
            
            if len(chosen) > max_size_hint + 5: # Early stop if clearly too big?
                 # But we might need to check if it can be pruned.
                 # For now, continue to get a full cover.
                 pass
                 
            if time.time() > deadline:
                return None
                
        return chosen

    # Run multiple greedy restarts
    # Since greedy is deterministic for a given order, we shuffle the initial remaining list?
    # Or just run it once. It's fast.
    # To get variety, we can perturb the weights or use random tie-breaking.
    
    initial_cover = greedy_cover(masks, full_mask, deadline, rng)
    
    if initial_cover is None:
         # Fallback: return a dense hypergraph
         # e.g. all edges on 9 vertices
         best_edges = [list(e) for e in combinations(range(9), 5)]
         best_n = 9
         return {'n': best_n, 'edges': best_edges}

    current_cover = set(initial_cover)
    
    # Local Search: Prune and Swap
    # Goal: Minimize size.
    
    def get_union_mask(cover_set, masks):
        u = 0
        for idx in cover_set:
            u |= masks[idx]
        return u
    
    if get_union_mask(current_cover, masks) != full_mask:
        # Should not happen if greedy succeeded
        pass
        
    improved = True
    while improved and time.time() < deadline:
        improved = False
        cover_list = list(current_cover)
        rng.shuffle(cover_list)
        
        for i_idx in cover_list:
            if i_idx not in current_cover:
                continue
            
            # Try to remove i_idx
            cover_without_i = current_cover - {i_idx}
            union_without_i = 0
            for idx in cover_without_i:
                union_without_i |= masks[idx]
                
            if union_without_i == full_mask:
                current_cover = cover_without_i
                improved = True
                break
            
            # Try to swap i_idx with j_idx
            missing = full_mask & ~union_without_i
            if missing:
                found_swap = False
                # To speed up, only consider j that covers significant part of missing
                for j_idx in range(len(masks)):
                    if j_idx in current_cover:
                        continue
                    if (masks[j_idx] & missing) == missing:
                        current_cover = cover_without_i | {j_idx}
                        improved = True
                        found_swap = True
                        break
                if found_swap:
                    break
                    
    if best_edges is None or len(current_cover) < best_size:
        best_edges = [list(all_edges[idx]) for idx in current_cover]
        best_n = n
        best_size = len(current_cover)
        
    # Verify the best solution found
    if best_edges and _is_non_2_colorable(best_edges, best_n):
        return {'n': best_n, 'edges': best_edges}
    else:
        # If the local search result is invalid (2-colorable), we need a fallback.
        # This shouldn't happen if the mask logic is correct, but just in case.
        # Return the greedy cover which is guaranteed to cover all colorings (thus non-2-colorable)
        # Wait, if greedy cover is valid, why would local search break it?
        # Swap logic ensures coverage is maintained.
        # So best_edges should be valid.
        
        # If for some reason it's invalid, return a known dense one.
        fallback_edges = [list(e) for e in combinations(range(11), 5)]
        return {'n': 11, 'edges': fallback_edges}

if __name__ == '__main__':
    construct()
