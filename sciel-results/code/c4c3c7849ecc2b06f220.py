import time
import random
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

# Precompute edge indices
edge_to_idx = {}
edges = []
for i in range(N):
    for j in range(i + 1, N):
        edge_to_idx[(i, j)] = len(edges)
        edges.append((i, j))

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return edge_to_idx[(i, j)]

# Precompute all 4-cycles: each is a tuple of 4 edge indices
# For each 4-vertex set {a,b,c,d}, the 3 cycles are:
# (a,b,c,d): ab, bc, cd, da
# (a,b,d,c): ab, bd, dc, ca  
# (a,c,b,d): ac, cb, bd, da
all_cycles = []
for a, b, c, d in combinations(range(N), 4):
    # Cycle 1: a-b-c-d-a
    e1 = (edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a))
    all_cycles.append(e1)
    # Cycle 2: a-b-d-c-a
    e2 = (edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a))
    all_cycles.append(e2)
    # Cycle 3: a-c-b-d-a
    e3 = (edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a))
    all_cycles.append(e3)

# For each edge, which cycles contain it
edge_to_cycles = [[] for _ in range(NUM_EDGES)]
for idx, cycle in enumerate(all_cycles):
    for e in cycle:
        edge_to_cycles[e].append(idx)

def count_colors(colors):
    return len(set(colors))

def is_valid(colors):
    """Check if no 4-cycle has 4 distinct colors."""
    for cycle in all_cycles:
        c0, c1, c2, c3 = colors[cycle[0]], colors[cycle[1]], colors[cycle[2]], colors[cycle[3]]
        if len({c0, c1, c2, c3}) == 4:
            return False
    return True

def check_edge_change(colors, edge, new_color):
    """Check if setting edge to new_color would violate any constraint."""
    old_color = colors[edge]
    if old_color == new_color:
        return True
    colors[edge] = new_color
    valid = True
    for cycle_idx in edge_to_cycles[edge]:
        cycle = all_cycles[cycle_idx]
        c0, c1, c2, c3 = colors[cycle[0]], colors[cycle[1]], colors[cycle[2]], colors[cycle[3]]
        if len({c0, c1, c2, c3}) == 4:
            valid = False
            break
    colors[edge] = old_color
    return valid

def construct():
    start_time = time.time()
    DEADLINE = 80  # safety margin
    
    # Star construction:
    # Center at vertex 0.
    # Edges (0, j) for j in 1..15 get distinct colors 0..14.
    # All other edges get color 15.
    # Total colors: 16.
    
    colors = [15] * NUM_EDGES
    for j in range(1, N):
        idx = edge_idx(0, j)
        colors[idx] = j - 1
    
    best_count = count_colors(colors)
    best_colors = colors[:]
    
    # Try to add distinct colors to non-star edges via greedy + local search
    # Non-star edges are those not incident to vertex 0
    non_star_edges = [idx for idx in range(NUM_EDGES) 
                      if edges[idx][0] != 0 and edges[idx][1] != 0]
    
    # Greedy: try to assign new distinct colors to non-star edges
    used_colors = set(colors)
    available_colors = [c for c in range(120) if c not in used_colors]
    
    # Sort non-star edges by degree of constraint (fewer cycles first)
    non_star_edges.sort(key=lambda e: len(edge_to_cycles[e]))
    
    for idx in non_star_edges:
        if time.time() > DEADLINE:
            break
        # Try to assign a new color to this edge
        for new_color in available_colors:
            if check_edge_change(colors, idx, new_color):
                colors[idx] = new_color
                used_colors.add(new_color)
                available_colors.remove(new_color)
                break
    
    best_count = count_colors(colors)
    best_colors = colors[:]
    
    # Local search: try to improve by swapping colors
    if time.time() < DEADLINE - 5:
        random.seed(42)
        current_colors = colors[:]
        current_count = best_count
        
        # Try random edge recoloring
        for _ in range(10000):
            if time.time() > DEADLINE:
                break
            
            # Pick a random edge
            idx = random.randint(0, NUM_EDGES - 1)
            
            # Try a random new color
            new_color = random.randint(0, 119)
            
            if check_edge_change(current_colors, idx, new_color):
                old_color = current_colors[idx]
                current_colors[idx] = new_color
                new_count = count_colors(current_colors)
                
                if new_count > current_count:
                    current_count = new_count
                    if current_count > best_count:
                        best_count = current_count
                        best_colors = current_colors[:]
                else:
                    # Revert if no improvement (keep it simple)
                    current_colors[idx] = old_color
    
    if time.time() > DEADLINE - 2:
        return best_colors
    
    return best_colors
