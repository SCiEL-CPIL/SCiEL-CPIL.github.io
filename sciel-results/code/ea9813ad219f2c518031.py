import time
import random
from itertools import combinations

def is_2_colorable(edges):
    """Check if the hypergraph can be 2-colored with no monochromatic edge."""
    n = 9
    # Try all 2^9 colorings
    for mask in range(1 << n):
        color = [(mask >> i) & 1 for i in range(n)]
        ok = True
        for e in edges:
            c0, c1, c2 = color[e[0]], color[e[1]], color[e[2]]
            if c0 == c1 == c2:
                ok = False
                break
        if ok:
            return True
    return False

def is_valid_coloring(edges, colors):
    """Check that no edge is monochromatic under the given coloring."""
    for e in edges:
        c0, c1, c2 = colors[e[0]], colors[e[1]], colors[e[2]]
        if c0 == c1 == c2:
            return False
    return True

def construct():
    deadline = time.time() + 80.0  # Safety margin
    
    # Known good solution
    best_edges = [list(e) for e in combinations(range(5), 3)]
    best_colors = [0, 0, 1, 1, 2, 0, 0, 0, 0]
    
    all_triples = list(combinations(range(9), 3))
    
    # Try to find smaller solutions
    for k in range(1, len(best_edges)):
        if time.time() > deadline:
            break
        
        # Random search for k-edge 3-chromatic hypergraphs
        max_attempts = 50000 if k <= 6 else 20000
        for _ in range(max_attempts):
            if time.time() > deadline:
                break
            
            edges = random.sample(all_triples, k)
            edge_list = [list(e) for e in edges]
            
            if not is_2_colorable(edge_list):
                # Find a valid 3-coloring
                # Try random 3-colorings
                found = False
                for _ in range(10000):
                    colors = [random.randint(0, 2) for _ in range(9)]
                    if is_valid_coloring(edge_list, colors):
                        found = True
                        break
                if found:
                    best_edges = edge_list
                    best_colors = colors
                    break
        else:
            continue
        break
    
    # Ensure types are correct
    return {
        'edges': [list(e) for e in best_edges],
        'colors': [int(c) for c in best_colors]
    }
