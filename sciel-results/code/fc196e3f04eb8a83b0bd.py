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

# Precompute all 4-cycles
all_cycles = []
for a, b, c, d in combinations(range(N), 4):
    e1 = (edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a))
    all_cycles.append(e1)
    e2 = (edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a))
    all_cycles.append(e2)
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
    for cycle in all_cycles:
        c0, c1, c2, c3 = colors[cycle[0]], colors[cycle[1]], colors[cycle[2]], colors[cycle[3]]
        if len({c0, c1, c2, c3}) == 4:
            return False
    return True

def check_edge_change_fast(colors, edge, new_color):
    """Fast check: only look at cycles containing this edge."""
    old_color = colors[edge]
    if old_color == new_color:
        return True
    for cycle_idx in edge_to_cycles[edge]:
        cycle = all_cycles[cycle_idx]
        # Get the other 3 edges in this cycle
        other_edges = [e for e in cycle if e != edge]
        c0, c1, c2 = colors[other_edges[0]], colors[other_edges[1]], colors[other_edges[2]]
        # Check if adding new_color creates 4 distinct colors
        if len({new_color, c0, c1, c2}) == 4:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin
    
    # Start with star construction (16 colors)
    colors = [15] * NUM_EDGES
    for j in range(1, N):
        idx = edge_idx(0, j)
        colors[idx] = j - 1
    best_colors = colors[:]
    best_count = count_colors(best_colors)  # 16
    
    random.seed(42)
    
    # Track which colors are used
    used_color_set = set(colors)
    
    for iteration in range(5000000):
        if time.time() > deadline:
            break
        
        # Strategy 1: Try to give a random edge a brand new unique color
        edge = random.randint(0, NUM_EDGES - 1)
        current_color = colors[edge]
        
        # Pick a color not currently in use
        new_color = random.randint(0, 119)
        if new_color not in used_color_set:
            if check_edge_change_fast(colors, edge, new_color):
                colors[edge] = new_color
                used_color_set.add(new_color)
                new_count = len(used_color_set)
                if new_count > best_count:
                    best_count = new_count
                    best_colors = colors[:]
        
        # Strategy 2: Try to reassign an edge to an existing color (to potentially
        # free up a slot for a new color later, or just explore the space)
        if random.random() < 0.5:
            edge = random.randint(0, NUM_EDGES - 1)
            current_color = colors[edge]
            new_color = random.randint(0, 119)
            if new_color != current_color:
                if check_edge_change_fast(colors, edge, new_color):
                    old_color = colors[edge]
                    colors[edge] = new_color
                    # Update used_color_set
                    # Check if old_color is still used elsewhere
                    still_used = False
                    for i in range(NUM_EDGES):
                        if i != edge and colors[i] == old_color:
                            still_used = True
                            break
                    if not still_used:
                        used_color_set.discard(old_color)
                    used_color_set.add(new_color)
                    new_count = len(used_color_set)
                    if new_count > best_count:
                        best_count = new_count
                        best_colors = colors[:]
    
    return best_colors
