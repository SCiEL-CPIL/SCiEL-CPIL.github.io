import time
import random
from itertools import combinations

def _popcount(x):
    return bin(x).count('1')

def _is_non_2_colorable_fast(n, edges, deadline):
    """
    Checks if the hypergraph defined by 'edges' (list of bitmasks) on 'n' vertices
    is NOT 2-colorable.
    Returns True if NOT 2-colorable (UNSAT), False if 2-colorable (SAT).
    """
    if time.time() > deadline:
        return False # Assume colorable if timeout, to be safe for "finding" non-colorable? 
                     # Actually, if we are checking if it's non-colorable to KEEP it, 
                     # a timeout might lead to false negatives. 
                     # But for greedy deletion, we want to be sure.
                     # Let's assume the check is fast enough.
    
    m = len(edges)
    if m == 0:
        return False # Empty is colorable

    # Pre-compute edge masks
    # edges is a list of ints, each int is a bitmask of 5 vertices
    
    # State: mask0, mask1
    # full_mask = (1 << n) - 1
    full_mask = (1 << n) - 1
    
    # We will use a stack for backtracking to avoid recursion depth issues, 
    # or just recursion if depth is small (n <= 12). Recursion is fine.
    
    def propagate(mask0, mask1):
        """
        Applies unit propagation.
        Returns (new_mask0, new_mask1, contradiction)
        """
        changed = True
        while changed:
            changed = False
            for e in edges:
                # Count colors in this edge
                c0 = _popcount(e & mask0)
                c1 = _popcount(e & mask1)
                
                # Contradiction: all 5 same
                if c0 == 5 or c1 == 5:
                    return mask0, mask1, True
                
                # Unit propagation
                # If 4 are color 0, the 5th must be color 1
                if c0 == 4:
                    # The unassigned vertex in e must be 1
                    # unassigned in e = e & ~(mask0 | mask1)
                    unassigned_in_e = e & ~(mask0 | mask1)
                    if unassigned_in_e:
                        # There should be exactly 1 unassigned vertex
                        # If there are 0, c0 would be 5 (handled) or c1 would be 1? 
                        # If c0=4, c1 must be 0 (since 4+0=4 assigned, 1 unassigned).
                        # So unassigned_in_e is a single bit.
                        mask1 |= unassigned_in_e
                        changed = True
                elif c1 == 4:
                    unassigned_in_e = e & ~(mask0 | mask1)
                    if unassigned_in_e:
                        mask0 |= unassigned_in_e
                        changed = True
                        
        return mask0, mask1, False

    def backtrack(mask0, mask1):
        """
        Returns True if a valid 2-coloring exists (SAT).
        """
        if time.time() > deadline:
            return True # Conservative: assume colorable if timeout? 
                        # If we assume colorable, we might fail to identify a non-colorable graph.
                        # But for greedy deletion, if we think it's colorable, we keep the edge.
                        # This is safe (we might not remove enough, but we won't return an invalid graph).
                        
        # Check for immediate contradiction
        for e in edges:
            c0 = _popcount(e & mask0)
            c1 = _popcount(e & mask1)
            if c0 == 5 or c1 == 5:
                return False
        
        # Propagate
        new_mask0, new_mask1, contra = propagate(mask0, mask1)
        if contra:
            return False
        
        # Find next unassigned vertex
        # Heuristic: pick vertex in the most constrained edge?
        # Or just first unassigned.
        unassigned = ~(new_mask0 | new_mask1) & full_mask
        if unassigned == 0:
            # All assigned, no contradiction found, so it's a valid coloring
            return True
            
        # Pick a vertex
        # To speed up, pick a vertex that appears in many edges?
        # For now, just lowest bit.
        v = (unassigned & -unassigned).bit_length() - 1
        
        # Try color 0
        if backtrack(new_mask0 | (1 << v), new_mask1):
            return True
            
        # Try color 1
        if backtrack(new_mask0, new_mask1 | (1 << v)):
            return True
            
        return False

    # Start backtracking
    return not backtrack(0, 0)


def _greedy_minimize(n, all_edges_bitmasks, deadline):
    """
    Greedily removes edges from all_edges_bitmasks while maintaining non-2-colorability.
    Returns the minimized list of edges (bitmasks).
    """
    current_edges = list(all_edges_bitmasks)
    
    # Check if initial set is non-2-colorable
    if not _is_non_2_colorable_fast(n, current_edges, deadline):
        return None

    # Shuffle to get different minimal sets? Or just iterate.
    # Iterating in fixed order might get stuck in local minima.
    # Shuffling helps.
    indices = list(range(len(current_edges)))
    random.shuffle(indices)
    
    for idx in indices:
        if time.time() > deadline:
            break
        if idx >= len(current_edges):
            continue
            
        edge_to_remove = current_edges[idx]
        
        # Create new list without this edge
        # Since we are iterating by index, and list changes, this is tricky.
        # Better to iterate over a copy of the list of edges, or use a set?
        # Sets are unordered, but we can use a list and mark removed.
        
        # Let's use a list of edges and a list of active flags?
        # Or just reconstruct the list.
        # Since m is small (~100-500), reconstruction is fine.
        
        new_edges = [e for i, e in enumerate(current_edges) if i != idx]
        
        if not new_edges:
            break
            
        # Check if new_edges is non-2-colorable
        if _is_non_2_colorable_fast(n, new_edges, deadline):
            current_edges = new_edges
            
    return current_edges


def construct():
    deadline = time.time() + 140  # Safety margin
    
    best_result = None
    best_edge_count = float('inf')
    
    # Try small n
    for n in [9, 10, 11, 12]:
        if time.time() > deadline:
            break
            
        # Generate all 5-uniform edges on n vertices
        all_edges_bitmasks = []
        for e in combinations(range(n), 5):
            mask = 0
            for v in e:
                mask |= (1 << v)
            all_edges_bitmasks.append(mask)
            
        # Greedy minimize
        minimized = _greedy_minimize(n, all_edges_bitmasks, deadline)
        
        if minimized is not None:
            edge_count = len(minimized)
            if edge_count < best_edge_count:
                best_edge_count = edge_count
                # Convert back to list of lists
                edges_list = []
                for mask in minimized:
                    verts = [v for v in range(n) if (mask >> v) & 1]
                    edges_list.append(verts)
                best_result = {'n': n, 'edges': edges_list}
                
        # Early exit if we hit the known bound
        if best_edge_count <= 51:
            break

    if best_result is None:
        # Fallback: Complete 5-uniform on 9 vertices
        edges = [list(e) for e in combinations(range(9), 5)]
        best_result = {'n': 9, 'edges': edges}
        
    return best_result
