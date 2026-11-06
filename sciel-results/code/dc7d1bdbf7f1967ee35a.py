"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import itertools
import numpy as np

N = 16
E = N*(N-1)//2  # 120 edges

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i*(2*N-1-i)//2 + j - i - 1

def precompute_cycles_for_edges():
    """
    Returns a list of lists. cycles_for_edge[e] is a list of tuples (f1, f2, f3)
    where (e, f1, f2, f3) forms a 4-cycle.
    """
    cycles_for_edge = [[] for _ in range(E)]
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b, b-c, c-d, d-a
        e1 = (edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a))
        # Cycle 2: a-b, b-d, d-c, c-a
        e2 = (edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a))
        # Cycle 3: a-c, c-b, b-d, d-a
        e3 = (edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a))
        
        for cyc in (e1, e2, e3):
            for i in range(4):
                e = cyc[i]
                others = [cyc[j] for j in range(4) if j != i]
                cycles_for_edge[e].append(tuple(others))
    return cycles_for_edge

def count_colors(colors):
    return len(set(colors))

def construct():
    # Safety margin: 60s limit, kill at 90s. Use 50s for computation.
    deadline = time.time() + 50.0
    
    cycles_for_edge = precompute_cycles_for_edges()
    
    # Start with a valid construction: color(i,j) = i (min vertex)
    # This uses 15 colors (0..14).
    colors = np.zeros(E, dtype=int)
    for i in range(N):
        for j in range(i+1, N):
            e = edge_idx(i, j)
            colors[e] = i
    
    best = colors.copy()
    best_count = count_colors(best)
    
    random.seed(42)
    max_color = 119
    
    # Local Search
    # We try to recolor edges to increase the number of distinct colors.
    
    while time.time() < deadline:
        # Pick a random edge
        e = random.randint(0, E-1)
        old_color = colors[e]
        
        # Try to find a new color that is valid and ideally new or helps diversity
        # Strategy: Try a color not currently used by this edge's neighbors in cycles?
        # Simplest: Try random colors, check validity.
        # To maximize distinct colors, we should prefer colors not in `set(colors)`.
        
        # Optimization: Pre-compute valid colors for this edge given current state?
        # That's expensive. Just try a few random ones.
        
        # Try to assign a color from the unused palette first
        # We can maintain a set of used colors for O(1) lookup, but updating it is O(1).
        # Let's just try random colors and check if they are in the used set.
        
        # Try 20 random colors to find a valid one
        found = False
        for _ in range(20):
            new_color = random.randint(0, max_color)
            if new_color == old_color:
                continue
            
            valid = True
            for others in cycles_for_edge[e]:
                c1, c2, c3 = others
                # Check if c1, c2, c3 are distinct and none equals new_color
                # If they are distinct and new_color is different from all, then the 4 colors are distinct -> Invalid
                if c1 != c2 and c1 != c3 and c2 != c3:
                    if new_color != c1 and new_color != c2 and new_color != c3:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if valid:
                colors[e] = new_color
                # Update best if count increases
                nc = count_colors(colors)
                if nc > best_count:
                    best = colors.copy()
                    best_count = nc
                found = True
                break
        
        # If we didn't find a valid new color, we might try to swap colors or just move on.
        # For now, just move on. This is a simple hill climber.
        
        # To escape local optima, we can occasionally accept a non-improving move (Simulated Annealing style)
        # but keeping it simple for speed.
        
    return best.tolist()
# EVOLVE-BLOCK-END
