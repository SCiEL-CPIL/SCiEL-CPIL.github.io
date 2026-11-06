import itertools
import random
import time

def is_2_colorable_fast(n, edges):
    """
    Checks if a 5-uniform hypergraph is 2-colorable.
    Returns True if 2-colorable, False otherwise.
    """
    if not edges:
        return True
    
    # Convert edges to bitmasks for faster operations
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    # For small n, brute force is faster than backtracking overhead
    if n <= 12:
        # Try all 2^n colorings
        # A coloring is valid if no edge is monochromatic
        for c in range(1 << n):
            valid = True
            for em in edge_masks:
                # Edge is monochromatic if all bits are 0 or all bits are 1
                if (em & c) == 0 or (em & c) == em:
                    valid = False
                    break
            if valid:
                return True
        return False
    else:
        # Backtracking with unit propagation
        num_edges = len(edge_masks)
        
        # Precompute vertex to edges mapping
        vertex_edges = [[] for _ in range(n)]
        for i, em in enumerate(edge_masks):
            for v in range(n):
                if em & (1 << v):
                    vertex_edges[v].append(i)
        
        # State
        color = [-1] * n  # -1: unassigned, 0: color 0, 1: color 1
        c0 = [0] * num_edges  # count of color 0 in each edge
        c1 = [0] * num_edges  # count of color 1 in each edge
        
        def propagate():
            """
            Unit propagation: If an edge has 4 vertices of one color, the 5th must be the other.
            Returns False if conflict (5 of one color), True otherwise.
            """
            changed = True
            while changed:
                changed = False
                for i in range(num_edges):
                    if c0[i] == 5 or c1[i] == 5:
                        return False
                    if c0[i] == 4 and c1[i] == 0:
                        # Must have one more vertex as color 1
                        for v in range(n):
                            if color[v] == -1 and (edge_masks[i] & (1 << v)):
                                color[v] = 1
                                c1[i] += 1
                                for other_e in vertex_edges[v]:
                                    c1[other_e] += 1
                                changed = True
                                if c1[i] == 5:
                                    return False
                                break
                    elif c1[i] == 4 and c0[i] == 0:
                        # Must have one more vertex as color 0
                        for v in range(n):
                            if color[v] == -1 and (edge_masks[i] & (1 << v)):
                                color[v] = 0
                                c0[i] += 1
                                for other_e in vertex_edges[v]:
                                    c0[other_e] += 1
                                changed = True
                                if c0[i] == 5:
                                    return False
                                break
            return True

        def backtrack():
            # Check for immediate conflicts
            for i in range(num_edges):
                if c0[i] == 5 or c1[i] == 5:
                    return False
            
            # Find most constrained unassigned vertex
            # Heuristic: Pick vertex in an edge with fewest unassigned vertices, 
            # preferring edges with high imbalance (closer to conflict)
            best_v = -1
            best_unassigned = 6
            best_constraint = 0
            best_edge_idx = -1
            
            for i in range(num_edges):
                unassigned = 5 - c0[i] - c1[i]
                constraint = max(c0[i], c1[i])
                if unassigned < best_unassigned or (unassigned == best_unassigned and constraint > best_constraint):
                    best_unassigned = unassigned
                    best_constraint = constraint
                    best_edge_idx = i
            
            if best_edge_idx == -1:
                # All edges have 0 unassigned? This implies all vertices assigned.
                # Check if any edge is monochromatic (should have been caught by conflict check)
                # If we reach here, it means we found a valid coloring
                return True
            
            # Find an unassigned vertex in the best edge
            candidates = []
            for v in range(n):
                if color[v] == -1 and (edge_masks[best_edge_idx] & (1 << v)):
                    candidates.append(v)
            
            if not candidates:
                # No unassigned vertices in this edge? 
                # This means the edge is fully assigned. 
                # If it's not monochromatic, it's fine. 
                # We need to pick a vertex from another edge? 
                # Actually, if best_unassigned > 0, there must be candidates.
                # If best_unassigned == 0, we should have returned True if no conflict.
                # Let's just return True if no conflicts and no unassigned in critical edges?
                # Safer: pick any unassigned vertex if best_edge_idx logic failed?
                # But best_edge_idx is chosen based on unassigned count.
                # If unassigned is 0 for all, we are done.
                return True

            # Try assigning colors to the best candidate vertex
            # We pick the first candidate for simplicity, or one with high degree?
            # Let's pick the first one to keep it simple, or randomize?
            # Randomizing can help avoid symmetry traps.
            random.shuffle(candidates)
            v = candidates[0]
            
            # Try color 0
            saved_c0 = c0[:]
            saved_c1 = c1[:]
            color[v] = 0
            ok = True
            for e_idx in vertex_edges[v]:
                c0[e_idx] += 1
                if c0[e_idx] == 5:
                    ok = False
                    break
            if ok and propagate():
                if backtrack():
                    return True
            color[v] = -1
            c0[:] = saved_c0
            c1[:] = saved_c1
            
            # Try color 1
            saved_c0 = c0[:]
            saved_c1 = c1[:]
            color[v] = 1
            ok = True
            for e_idx in vertex_edges[v]:
                c1[e_idx] += 1
                if c1[e_idx] == 5:
                    ok = False
                    break
            if ok and propagate():
                if backtrack():
                    return True
            color[v] = -1
            c0[:] = saved_c0
            c1[:] = saved_c1
            
            return False
        
        return backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_edges = None
    best_n = 9
    best_m = 10**9
    
    # Start with known small non-2-colorable hypergraph: K9^5
    # It has C(9,5) = 126 edges.
    # We will try to minimize this.
    
    # List of n to try
    # n=9 is the smallest n where K_n^5 is not 2-colorable (Property B fails).
    # For n < 9, K_n^5 is 2-colorable.
    # We can also try sparse random graphs on larger n, but starting from dense is a good heuristic for minimization via removal.
    
    for n in range(9, 14):
        if time.time() > deadline - 20:
            break
        
        # Generate all possible edges for this n
        all_possible_edges = [list(e) for e in itertools.combinations(range(n), 5)]
        
        # Start with the complete hypergraph
        current_edges = all_possible_edges[:]
        
        # Check if complete hypergraph is non-2-colorable
        # For n=9, it is known to be non-2-colorable.
        # For larger n, it is also non-2-colorable.
        if is_2_colorable_fast(n, current_edges):
            continue
        
        best_m_for_n = len(current_edges)
        best_edges_for_n = current_edges[:]
        
        # Perform multiple local search rounds
        # Each round: Greedy removal + Swaps to escape local minima
        max_rounds = 50 # Limit rounds to avoid infinite loop, time check is primary
        
        for _ in range(max_rounds):
            if time.time() > deadline - 10:
                break
            
            # 1. Greedy Removal Phase
            # Try to remove edges one by one.
            # Shuffle order to try different removal paths
            indices = list(range(len(current_edges)))
            random.shuffle(indices)
            
            removed_any = False
            for i in indices:
                if time.time() > deadline - 10:
                    break
                
                if i >= len(current_edges):
                    continue
                
                # Try removing edge i
                e_to_remove = current_edges[i]
                new_edges = current_edges[:i] + current_edges[i+1:]
                
                # Check if new_edges is still non-2-colorable
                if not is_2_colorable_fast(n, new_edges):
                    current_edges = new_edges
                    removed_any = True
                    # If we removed an edge, the indices are now shifted.
                    # We can break and restart the greedy phase with a new shuffle,
                    # or continue carefully. Restarting is safer and often faster 
                    # because the structure changed significantly.
                    break
            
            if len(current_edges) < best_m_for_n:
                best_m_for_n = len(current_edges)
                best_edges_for_n = current_edges[:]
            
            if not removed_any:
                # Stuck in local minimum. Try Swaps.
                # A swap replaces an edge e1 with an edge e2 not in the graph.
                # This doesn't change the count, but might allow further removals.
                
                # Find an edge to swap out. 
                # Heuristic: Pick a random edge.
                # Or pick an edge that is "critical" (removing it makes it 2-colorable).
                # Actually, if we are stuck, ALL edges are critical (removing any makes it 2-colorable).
                # So we just pick a random edge e1.
                
                if len(current_edges) == 0:
                    break
                
                e1_idx = random.randint(0, len(current_edges) - 1)
                e1 = current_edges[e1_idx]
                
                # Find a candidate edge e2 to swap in.
                # e2 must not be in current_edges.
                # To be efficient, we can't check all C(n,5) edges.
                # We can pick a random edge from all_possible_edges and check if it's in current_edges.
                # If it is, pick another.
                
                # Optimization: Maintain a set of current edge bitmasks for O(1) lookup
                current_edge_set = set(tuple(sorted(e)) for e in current_edges)
                
                swap_found = False
                attempts = 0
                max_swap_attempts = 1000 # Don't spend too much time finding a swap
                
                while attempts < max_swap_attempts and time.time() < deadline - 10:
                    e2 = random.choice(all_possible_edges)
                    e2_tuple = tuple(sorted(e2))
                    if e2_tuple not in current_edge_set:
                        # Try the swap
                        new_edges = current_edges[:]
                        new_edges[e1_idx] = e2
                        new_edge_set = set(tuple(sorted(e)) for e in new_edges)
                        
                        # Check if new hypergraph is non-2-colorable
                        if not is_2_colorable_fast(n, new_edges):
                            current_edges = new_edges
                            current_edge_set = new_edge_set
                            swap_found = True
                            break
                    attempts += 1
                
                # After a swap, we might be able to remove more edges.
                # The loop will continue to the next iteration of the outer loop,
                # which starts with Greedy Removal again.
                
                # If no swap found, we are likely stuck. 
                # We can try a different n or stop.
                # For now, we just break the inner loop for this n if we can't improve?
                # No, we might have found a better structure in a previous round.
                # We should probably break the "local search" for this specific start if we can't move.
                # But maybe a different random swap works?
                # Given time constraints, if we can't find a swap, we stop trying to improve this specific instance.
                if not swap_found:
                    break # Move to next n or exit
    
        if best_m_for_n < best_m:
            best_m = best_m_for_n
            best_edges = best_edges_for_n
            best_n = n
            
    if best_edges is None:
        # Fallback to a known valid small hypergraph if search failed completely
        # K9^5 is always valid
        n = 9
        best_edges = [list(e) for e in itertools.combinations(range(n), 5)]
        best_n = n
        
    return {'n': best_n, 'edges': best_edges}

# Optional: Allow running as a script for testing
if __name__ == "__main__":
    import sys
    start = time.time()
    result = construct()
    end = time.time()
    print(f"Time taken: {end - start:.2f}s")
    print(f"n: {result['n']}")
    print(f"Edges: {len(result['edges'])}")
    # Verify
    if not is_2_colorable_fast(result['n'], result['edges']):
        print("Valid: Non-2-colorable")
    else:
        print("INVALID: 2-colorable")
