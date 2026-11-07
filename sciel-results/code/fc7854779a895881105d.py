"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def is_2_colorable(edges, n=9):
    """Check if hypergraph can be 2-colored (no monochromatic edge)."""
    # Try all 2^n colorings is too slow for n=9 (512), so it's fine
    for mask in range(1 << n):
        valid = True
        for e in edges:
            colors = set()
            for v in e:
                colors.add((mask >> v) & 1)
            if len(colors) == 1:
                valid = False
                break
        if valid:
            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s safety margin
    
    # Try to find minimal edge count
    # Start with small numbers of edges and check
    all_triples = list(combinations(range(9), 3))
    
    # We know the answer is at most 10 (from seed)
    # Let's search for smaller solutions
    best_edges = None
    best_colors = None
    
    # Try edge counts from 1 upward, but be smart
    # Actually, let's use a known construction or search
    
    # Known: The minimum number of edges in a non-2-colorable 3-uniform hypergraph
    # on n vertices. For n=9, let's search.
    
    # Strategy: random sampling of edge sets of increasing size
    random.seed(42)
    
    for num_edges in range(1, 10):
        if time.time() > deadline:
            break
        
        # Try multiple random samples
        attempts = 10000 if num_edges < 5 else 5000
        for attempt in range(attempts):
            if time.time() > deadline:
                break
            
            edges = set(random.sample(all_triples, num_edges))
            edge_list = list(edges)
            
            if not is_2_colorable(edge_list):
                # Found a non-2-colorable hypergraph with num_edges edges
                # Now assign a 3-coloring (any proper 3-coloring works)
                # Simple: color by vertex index mod 3
                colors = [i % 3 for i in range(9)]
                
                # Verify no edge is monochromatic in this coloring
                valid = True
                for e in edge_list:
                    if colors[e[0]] == colors[e[1]] == colors[e[2]]:
                        valid = False
                        break
                
                if valid:
                    best_edges = edge_list
                    best_colors = colors
                    break
        
        if best_edges is not None:
            break
    
    # If we didn't find anything smaller, use the seed solution
    if best_edges is None:
        best_edges = [list(e) for e in combinations(range(5), 3)]
        best_colors = [0, 0, 1, 1, 2, 0, 0, 0, 0]
    
    # Ensure types are correct
    best_edges = [list(e) for e in best_edges]
    best_colors = [int(c) for c in best_colors]
    
    return {'edges': best_edges, 'colors': best_colors}
# EVOLVE-BLOCK-END
