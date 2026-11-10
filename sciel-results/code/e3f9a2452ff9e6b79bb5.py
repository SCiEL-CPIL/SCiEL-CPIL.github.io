"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _edge_index(i, j):
    """Return linear index for edge (i,j) in lexicographic order."""
    # Number of edges before vertex i: sum_{k=0}^{i-1} (16-k-1) = sum_{k=0}^{i-1} (15-k)
    # = i*15 - i*(i-1)//2
    # Then within vertex i, edge to j is at offset (j - i - 1)
    return i * 15 - i * (i - 1) // 2 + (j - i - 1)

def _is_valid(coloring):
    """Check if every K4 has at least 5 distinct colours."""
    for quad in combinations(range(16), 4):
        a, b, c, d = quad
        colors = {
            coloring[_edge_index(a, b)],
            coloring[_edge_index(a, c)],
            coloring[_edge_index(a, d)],
            coloring[_edge_index(b, c)],
            coloring[_edge_index(b, d)],
            coloring[_edge_index(c, d)],
        }
        if len(colors) < 5:
            return False
    return True

def _count_colors(coloring):
    return len(set(coloring))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: 10s before 150s limit
    
    # Initial coloring: (i+j) mod 17
    coloring = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    best = coloring[:]
    best_colors = _count_colors(best)
    
    # Precompute all K4 edge sets for fast validity checking
    k4_edges = []
    for quad in combinations(range(16), 4):
        a, b, c, d = quad
        k4_edges.append([
            _edge_index(a, b), _edge_index(a, c), _edge_index(a, d),
            _edge_index(b, c), _edge_index(b, d), _edge_index(c, d)
        ])
    
    # Local search: try to merge colors (reduce color count)
    # Strategy: try to recolor edges of one color to another, checking validity
    current = best[:]
    current_colors = best_colors
    
    # Try merging each pair of colors
    used_colors = sorted(set(current))
    
    for target_color in used_colors:
        if time.time() > deadline:
            break
        # Find all edges with this color
        edges_with_target = [i for i, c in enumerate(current) if c == target_color]
        
        # Try removing this color by recoloring its edges to other colors
        for other_color in used_colors:
            if other_color == target_color:
                continue
            if time.time() > deadline:
                break
            
            # Try recoloring all edges of target_color to other_color
            new_coloring = current[:]
            for ei in edges_with_target:
                new_coloring[ei] = other_color
            
            if _is_valid(new_coloring):
                new_count = _count_colors(new_coloring)
                if new_count < current_colors:
                    current = new_coloring
                    current_colors = new_count
                    best = current[:]
                    best_colors = new_count
                    used_colors = sorted(set(current))
    
    # Second pass: try more aggressive merging with randomization
    import random
    random.seed(42)
    
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        current = best[:]
        current_colors = best_colors
        improved = False
        
        # Try random color merges
        used_colors = sorted(set(current))
        random.shuffle(used_colors)
        
        for target_color in used_colors:
            if time.time() > deadline:
                break
            if target_color not in set(current):
                continue
            
            edges_with_target = [i for i, c in enumerate(current) if c == target_color]
            
            for other_color in used_colors:
                if other_color == target_color or other_color not in set(current):
                    continue
                
                new_coloring = current[:]
                for ei in edges_with_target:
                    new_coloring[ei] = other_color
                
                if _is_valid(new_coloring):
                    new_count = _count_colors(new_coloring)
                    if new_count < current_colors:
                        current = new_coloring
                        current_colors = new_count
                        best = current[:]
                        best_colors = new_count
                        improved = True
                        used_colors = sorted(set(current))
                        break
        
        if not improved:
            break
    
    return best
# EVOLVE-BLOCK-END
