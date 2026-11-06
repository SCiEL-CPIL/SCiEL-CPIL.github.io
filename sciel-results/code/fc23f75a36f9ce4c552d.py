"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _is_not_2_colorable(n, edges):
    """Check if the 5-uniform hypergraph is NOT 2-colorable using backtracking with unit propagation."""
    edges_set = [tuple(e) for e in edges]
    # Precompute for each vertex, which edges contain it
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges_set):
        for v in e:
            vertex_edges[v].append(i)
    
    # Backtracking with unit propagation
    def backtrack(assignment):
        # assignment: list of -1 (unassigned), 0, or 1
        # Unit propagation: if an edge has 4 vertices of the same color, the 5th must be different
        # Actually for 2-coloring to be proper, each edge must have both colors.
        # So if 4 vertices in an edge are color 0, the 5th MUST be color 1.
        # If 4 vertices are color 1, the 5th MUST be color 0.
        # If all 5 are the same color, it's a contradiction.
        
        # Apply unit propagation
        changed = True
        while changed:
            changed = False
            for e in edges_set:
                colors = [assignment[v] for v in e]
                # Count 0s and 1s
                num_0 = sum(1 for c in colors if c == 0)
                num_1 = sum(1 for c in colors if c == 1)
                num_un = sum(1 for c in colors if c == -1)
                
                if num_0 == 5 or num_1 == 5:
                    return False  # Contradiction: monochromatic edge
                if num_0 == 4 and num_un == 1:
                    # The unassigned vertex must be 1
                    for v in e:
                        if assignment[v] == -1:
                            assignment[v] = 1
                            changed = True
                elif num_1 == 4 and num_un == 1:
                    # The unassigned vertex must be 0
                    for v in e:
                        if assignment[v] == -1:
                            assignment[v] = 0
                            changed = True
                # If num_0 >= 3 and num_1 >= 3, we can't propagate (but that's fine, edge is satisfied)
                # Actually we need at least one 0 and one 1 in each edge.
                # If num_0 == 0 and num_1 == 4, the last must be 0? No, wait.
                # For proper 2-coloring: each edge must NOT be monochromatic.
                # So each edge must have at least one 0 AND at least one 1.
                # If all assigned vertices in an edge are 0, and there's 1 unassigned, it MUST be 1.
                # If all assigned vertices in an edge are 1, and there's 1 unassigned, it MUST be 0.
                # If there are both 0s and 1s already, no constraint on unassigned.
                # If all are unassigned, no constraint yet.
        
        # Find next unassigned vertex
        next_v = -1
        for v in range(n):
            if assignment[v] == -1:
                next_v = v
                break
        
        if next_v == -1:
            return True  # All assigned, no contradiction found
        
        # Try both colors
        for color in [0, 1]:
            assignment[next_v] = color
            if backtrack(assignment):
                return True
            assignment[next_v] = -1
        
        return False
    
    assignment = [-1] * n
    return not backtrack(assignment)

def _search_small_hypergraphs(deadline):
    """Search for small non-2-colorable 5-uniform hypergraphs."""
    best_edges = None
    best_n = None
    
    for n in range(5, 12):
        if time.time() > deadline:
            break
        all_5sets = list(combinations(range(n), 5))
        if len(all_5sets) > 500:  # Too many to search exhaustively
            continue
        
        # Try subsets of increasing size
        # For small n, try all subsets of size k for k from 5 upward
        # This is expensive, so use a heuristic: try random subsets first
        import random
        random.seed(42)
        
        # First try all small subsets
        for k in range(5, min(20, len(all_5sets) + 1)):
            if time.time() > deadline:
                break
            # For very small k, enumerate all
            if len(all_5sets) <= 30 and k <= 15:
                for subset in combinations(range(len(all_5sets)), k):
                    if time.time() > deadline:
                        break
                    edges = [all_5sets[i] for i in subset]
                    if _is_not_2_colorable(n, edges):
                        if best_edges is None or k < len(best_edges):
                            best_edges = edges
                            best_n = n
                        # Found one, can stop searching this k
                        break
            else:
                # Random sampling
                for _ in range(1000):
                    if time.time() > deadline:
                        break
                    if k > len(all_5sets):
                        break
                    subset = random.sample(range(len(all_5sets)), k)
                    edges = [all_5sets[i] for i in subset]
                    if _is_not_2_colorable(n, edges):
                        if best_edges is None or k < len(best_edges):
                            best_edges = edges
                            best_n = n
                        break
        
        if best_edges is not None:
            break  # Found a solution, stop
    
    if best_edges is not None:
        return {'n': best_n, 'edges': [list(e) for e in best_edges]}
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Leave 50s margin for safety
    
    # Try to find a small non-2-colorable 5-uniform hypergraph
    result = _search_small_hypergraphs(deadline)
    
    if result is not None:
        return result
    
    # Fallback: complete 5-uniform hypergraph on 9 vertices
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
# EVOLVE-BLOCK-END
