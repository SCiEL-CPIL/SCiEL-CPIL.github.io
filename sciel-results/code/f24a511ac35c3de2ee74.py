"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 55.0
    n_edges = N * (N - 1) // 2
    
    # Map edge (i,j) to index
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)
    
    # Precompute all 4-cycles: for each 4-vertex set, the 3 cycle edge-sets
    # Cycle on {a,b,c,d}: 
    #   C1: (a,b),(b,c),(c,d),(d,a)
    #   C2: (a,b),(b,d),(d,c),(c,a)
    #   C3: (a,c),(c,b),(b,d),(d,a)
    cycles = []  # list of tuples of 4 edge indices
    edge_to_cycles = [[] for _ in range(n_edges)]
    
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        # Three 4-cycles (as unordered edge sets, but we need ordered for the cycle)
        # The three perfect matchings of the 4 vertices give the three 4-cycles
        # Actually, a 4-cycle is a cycle of length 4 using all 4 vertices
        # There are 3 distinct 4-cycles on 4 labeled vertices:
        # (a-b-c-d-a), (a-b-d-c-a), (a-c-b-d-a)
        
        # Cycle 1: a-b-c-d-a
        e1 = [edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a)]
        cycles.append(tuple(e1))
        for e in e1:
            edge_to_cycles[e].append(len(cycles)-1)
        
        # Cycle 2: a-b-d-c-a
        e2 = [edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a)]
        cycles.append(tuple(e2))
        for e in e2:
            edge_to_cycles[e].append(len(cycles)-1)
        
        # Cycle 3: a-c-b-d-a
        e3 = [edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a)]
        cycles.append(tuple(e3))
        for e in e3:
            edge_to_cycles[e].append(len(cycles)-1)
    
    num_cycles = len(cycles)
    
    # Initialize with all zeros
    colors = [0] * n_edges
    used_colors = set([0])
    
    # Greedy phase: try to assign new colors to edges, one at a time
    # For each edge, try colors from 1 upward until valid
    def is_valid_edge_change(edge, new_color, cur_colors):
        """Check if changing edge to new_color keeps all cycles valid."""
        for cyc in edge_to_cycles[edge]:
            e0, e1, e2, e3 = cycles[cyc]
            # Get colors of the 4 edges
            c0 = cur_colors[e0] if e0 != edge else new_color
            c1 = cur_colors[e1] if e1 != edge else new_color
            c2 = cur_colors[e2] if e2 != edge else new_color
            c3 = cur_colors[e3] if e3 != edge else new_color
            if len(set([c0, c1, c2, c3])) == 4:
                return False
        return True
    
    # Greedy: for each edge, assign the smallest color > 0 that keeps validity
    # Actually, let's try to maximize distinct colors. Start all 0, then try to change edges to new colors.
    
    # Better: use local search. Start random, then hill-climb on number of distinct colors.
    # Objective: maximize len(set(colors)) subject to no 4-cycle having 4 distinct colors.
    
    # Initialize randomly with small number of colors
    random.seed(42)
    num_init_colors = 5
    for i in range(n_edges):
        colors[i] = random.randint(0, num_init_colors - 1)
    
    def count_distinct():
        return len(set(colors))
    
    def check_all_valid():
        for cyc in range(num_cycles):
            e0, e1, e2, e3 = cycles[cyc]
            if len(set([colors[e0], colors[e1], colors[e2], colors[e3]])) == 4:
                return False
        return True
    
    # First, repair: make sure initial coloring is valid
    # If not valid, reset problematic edges
    while not check_all_valid():
        for cyc in range(num_cycles):
            e0, e1, e2, e3 = cycles[cyc]
            if len(set([colors[e0], colors[e1], colors[e2], colors[e3]])) == 4:
                # Change one of the edges to match another
                colors[e3] = colors[e0]
                break
    
    best_distinct = count_distinct()
    
    # Local search: try to change random edges to new colors
    max_colors = 120
    for iteration in range(100000):
        if time.time() > deadline:
            break
        
        # Pick a random edge
        e = random.randint(0, n_edges - 1)
        old_color = colors[e]
        
        # Try a random new color
        new_color = random.randint(0, max_colors - 1)
        if new_color == old_color:
            continue
        
        if is_valid_edge_change(e, new_color, colors):
            colors[e] = new_color
            new_distinct = count_distinct()
            if new_distinct > best_distinct:
                best_distinct = new_distinct
        else:
            # Try to find a valid new color that adds a new distinct color
            # Sample a few
            for _ in range(3):
                nc = random.randint(0, max_colors - 1)
                if nc != old_color and is_valid_edge_change(e, nc, colors):
                    colors[e] = nc
                    new_distinct = count_distinct()
                    if new_distinct > best_distinct:
                        best_distinct = new_distinct
                    break
    
    return colors
# EVOLVE-BLOCK-END

