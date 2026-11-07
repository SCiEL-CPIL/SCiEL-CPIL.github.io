"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations, product

def construct():
    deadline = time.time() + 80
    
    def is_2_colorable(edges):
        """Check if hypergraph admits a 2-coloring (no monochromatic edge)."""
        n = 9
        for coloring in product([0, 1], repeat=n):
            ok = True
            for e in edges:
                if coloring[e[0]] == coloring[e[1]] == coloring[e[2]]:
                    ok = False
                    break
            if ok:
                return True
        return False
    
    def has_valid_3_coloring(edges):
        """Check if there exists a 3-coloring where no edge is monochromatic."""
        n = 9
        for coloring in product([0, 1, 2], repeat=n):
            ok = True
            for e in edges:
                if coloring[e[0]] == coloring[e[1]] == coloring[e[2]]:
                    ok = False
                    break
            if ok:
                return True, coloring
        return False, None
    
    # Fano plane - known 3-chromatic, 7 edges
    fano_edges = [
        [0, 1, 2], [0, 3, 4], [0, 5, 6],
        [1, 3, 5], [1, 4, 6], [2, 3, 6], [2, 4, 5]
    ]
    fano_colors = [0, 1, 2, 0, 1, 0, 2, 0, 0]
    
    # Verify Fano is valid
    assert not is_2_colorable(fano_edges)
    assert has_valid_3_coloring(fano_edges)[0]
    
    best_edges = fano_edges
    best_colors = fano_colors
    best_count = 7
    
    # Try to find 6-edge solutions
    if time.time() < deadline - 30:
        all_triples = list(combinations(range(9), 3))
        found = False
        for combo in combinations(all_triples, 6):
            if time.time() > deadline - 10:
                break
            edges = [list(e) for e in combo]
            # Quick check: verify 3-colorable
            valid, coloring = has_valid_3_coloring(edges)
            if not valid:
                continue
            # Check 3-chromatic (not 2-colorable)
            if not is_2_colorable(edges):
                best_edges = edges
                best_colors = list(coloring)
                best_count = 6
                found = True
                break
        if found:
            best_count = 6
    
    # Try 5-edge solutions (less likely but check)
    if best_count > 5 and time.time() < deadline - 10:
        all_triples = list(combinations(range(9), 3))
        found = False
        for combo in combinations(all_triples, 5):
            if time.time() > deadline - 5:
                break
            edges = [list(e) for e in combo]
            valid, coloring = has_valid_3_coloring(edges)
            if not valid:
                continue
            if not is_2_colorable(edges):
                best_edges = edges
                best_colors = list(coloring)
                best_count = 5
                found = True
                break
        if found:
            best_count = 5
    
    return {'edges': best_edges, 'colors': best_colors}

# EVOLVE-BLOCK-END
