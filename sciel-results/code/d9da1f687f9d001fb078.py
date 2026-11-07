"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations, product

def _is_2_colorable(edges, n=9):
    """Check if hypergraph can be 2-colored (no monochromatic edge)."""
    for coloring in product([0, 1], repeat=n):
        ok = True
        for e in edges:
            if coloring[e[0]] == coloring[e[1]] == coloring[e[2]]:
                ok = False
                break
        if ok:
            return True
    return False

def _is_3_colorable(edges, n=9):
    """Check if hypergraph can be 3-colored (no monochromatic edge)."""
    for coloring in product([0, 1, 2], repeat=n):
        ok = True
        for e in edges:
            if coloring[e[0]] == coloring[e[1]] == coloring[e[2]]:
                ok = False
                break
        if ok:
            return True
    return False

def construct():
    deadline = time.time() + 80  # safety margin
    
    all_edges = list(combinations(range(9), 3))
    
    # Try increasing number of edges
    for num_edges in range(3, 50):
        if time.time() > deadline:
            break
        
        # For small num_edges, we can enumerate
        if num_edges <= 6:
            for edge_set in combinations(all_edges, num_edges):
                if time.time() > deadline:
                    break
                edges = list(edge_set)
                
                # Check 3-colorable
                if not _is_3_colorable(edges):
                    continue
                
                # Check NOT 2-colorable
                if not _is_2_colorable(edges):
                    # Found a 3-chromatic hypergraph
                    # Find a valid 3-coloring
                    colors = None
                    for coloring in product([0, 1, 2], repeat=9):
                        ok = True
                        for e in edges:
                            if coloring[e[0]] == coloring[e[1]] == coloring[e[2]]:
                                ok = False
                                break
                        if ok:
                            colors = list(coloring)
                            break
                    
                    return {
                        'edges': [list(e) for e in edges],
                        'colors': colors
                    }
    
    # Fallback: return the original solution
    return {
        'edges': [list(e) for e in combinations(range(5), 3)],
        'colors': [0, 0, 1, 1, 2, 0, 0, 0, 0]
    }
# EVOLVE-BLOCK-END
