"""Few edges in 5-uniform hypergraphs without Property B (m(5))."""

import random
import itertools
from collections import deque

def construct():
    """
    Return a 5-uniform hypergraph on n vertices with no proper 2-colouring,
    minimizing the number of edges.
    
    Uses a randomized search to find a small non-2-colorable hypergraph.
    Falls back to a known construction if search fails.
    """
    
    def is_not_2_colorable(n, edges):
        """Check if hypergraph is NOT 2-colorable using backtracking."""
        m = len(edges)
        if m == 0:
            return False
        
        # Represent edges as bitmasks for fast operations
        edge_masks = []
        for e in edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks.append(mask)
        
        # Backtracking search for a 2-coloring where no edge is monochromatic.
        # If no such coloring exists, return True (not 2-colorable).
        
        # Order vertices by degree (number of edges they are in) to prune early
        degrees = [0] * n
        for mask in edge_masks:
            for v in range(n):
                if mask & (1 << v):
                    degrees[v] += 1
        
        order = sorted(range(n), key=lambda x: -degrees[x])
        
        # Map original vertex indices to positions in the order
        # Actually, it's easier to just iterate in order and track assigned colors.
        
        def search(idx, color_mask):
            # color_mask: bits 0..n-1 represent color 0? No, let's use two masks or a list.
            # Let's use a list of -1, 0, 1.
            pass
            
        # Better: Use bitmasks for the set of vertices colored 0 and 1.
        # State: (idx, mask0, mask1) where mask0 is set of vertices colored 0, mask1 for 1.
        # Unassigned vertices are ~(mask0 | mask1) & ((1<<n)-1)
        
        all_mask = (1 << n) - 1
        
        def backtrack(idx, mask0, mask1):
            if idx == n:
                # Check if all edges are mixed
                for mask in edge_masks:
                    # Edge is monochromatic if (mask & mask0) == mask or (mask & mask1) == mask
                    if (mask & mask0) == mask or (mask & mask1) == mask:
                        return False
                return True
            
            v = order[idx]
            
            # Try color 0
            if not (mask0 & (1 << v)) and not (mask1 & (1 << v)):
                # Check if assigning 0 to v creates a monochromatic edge
                # An edge becomes monochromatic 0 if all its vertices are in mask0 | {v}
                # and none in mask1.
                # Actually, we can just check if any edge containing v has all other vertices in mask0.
                # If so, assigning 0 makes it mono 0.
                # If any edge containing v has all other vertices in mask1, assigning 0 is fine (mixed).
                # Wait, if an edge has all other vertices in mask1, and we assign 0 to v, it becomes mixed.
                # If an edge has all other vertices in mask0, and we assign 0 to v, it becomes mono 0 (BAD).
                # If an edge has some in mask0 and some in mask1, it's already mixed (GOOD).
                # If an edge has some unassigned, it's not yet determined.
                
                # Optimization: If assigning 0 to v makes any edge monochromatic 0, we can prune?
                # No, we just need to ensure that AT THE END no edge is mono.
                # But if an edge is already mono 0 (all assigned 0), that's a fail.
                # So we can check: for any edge containing v, if (edge & ~(1<<v)) is subset of mask0, 
                # then assigning 0 makes it mono 0. This is a failure state for this branch?
                # No, it means this specific assignment leads to a mono edge. So we should not take this branch.
                
                valid = True
                for mask in edge_masks:
                    if mask & (1 << v):
                        # Check if all other vertices in this edge are in mask0
                        others = mask & ~(1 << v)
                        if others and (others & ~mask0) == 0:
                            # All others are 0, so this edge becomes mono 0
                            valid = False
                            break
                        # Also check if all others are in mask1? No, that would make it mixed.
                        # What if all others are unassigned? Then it's not mono yet.
                        
                if valid:
                    if backtrack(idx + 1, mask0 | (1 << v), mask1):
                        return True
                        
            # Try color 1
            if not (mask0 & (1 << v)) and not (mask1 & (1 << v)):
                valid = True
                for mask in edge_masks:
                    if mask & (1 << v):
                        others = mask & ~(1 << v)
                        if others and (others & ~mask1) == 0:
                            # All others are 1, so this edge becomes mono 1
                            valid = False
                            break
                        
                if valid:
                    if backtrack(idx + 1, mask0, mask1 | (1 << v)):
                        return True
                        
            return False
        
        return not backtrack(0, 0, 0)

    # Randomized search
    best_edges = None
    best_n = None
    
    # Try small n and m
    # We know n must be >= 9
    for n in range(9, 16):
        all_possible_edges = list(itertools.combinations(range(n), 5))
        # Try m from 30 up to 60
        for m in range(30, 61):
            if m > len(all_possible_edges):
                continue
            
            # Try a few random samples
            for _ in range(20):
                random.shuffle(all_possible_edges)
                candidate_edges = all_possible_edges[:m]
                if is_not_2_colorable(n, candidate_edges):
                    best_edges = candidate_edges
                    best_n = n
                    break
            if best_edges:
                break
        if best_edges:
            break
    
    if best_edges:
        return {'n': best_n, 'edges': [list(e) for e in best_edges]}
        
    # Fallback: Complete 5-uniform on 9 vertices (126 edges)
    n_start = 9
    all_edges_5 = list(itertools.combinations(range(n_start), 5))
    return {'n': n_start, 'edges': [list(e) for e in all_edges_5]}
