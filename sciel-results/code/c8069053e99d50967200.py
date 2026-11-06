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

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin
    
    # Initialize with Star construction (16 colors)
    colors = [15] * NUM_EDGES
    for j in range(1, N):
        idx = edge_idx(0, j)
        colors[idx] = j - 1
        
    # Frequency array for colors 0..119
    freq = [0] * 120
    for c in colors:
        freq[c] += 1
        
    best_colors = colors[:]
    best_count = sum(1 for f in freq if f > 0)
    
    random.seed(42)
    
    # Helper to check if a move is valid
    def is_move_valid(edge, new_color, current_colors, current_freq):
        old_color = current_colors[edge]
        if old_color == new_color:
            return True
        current_colors[edge] = new_color
        valid = True
        for cycle_idx in edge_to_cycles[edge]:
            cycle = all_cycles[cycle_idx]
            c0, c1, c2, c3 = current_colors[cycle[0]], current_colors[cycle[1]], current_colors[cycle[2]], current_colors[cycle[3]]
            if len({c0, c1, c2, c3}) == 4:
                valid = False
                break
        current_colors[edge] = old_color
        return valid

    for iteration in range(10000000):
        if time.time() > deadline:
            break
        
        edge = random.randint(0, NUM_EDGES - 1)
        
        # 30% chance to try a new color, 70% to try an existing color
        if random.random() < 0.3:
            # Pick a color that is currently unused or rare
            # Simple: pick random 0-119
            new_color = random.randint(0, 119)
        else:
            # Pick a random existing color
            # Weighted by frequency? Or just uniform from used?
            # Uniform from used is safer for validity.
            used = [c for c in range(120) if freq[c] > 0]
            if not used:
                continue
            new_color = random.choice(used)
            
        old_color = colors[edge]
        if new_color == old_color:
            continue
            
        if is_move_valid(edge, new_color, colors, freq):
            # Apply move
            freq[old_color] -= 1
            if freq[old_color] == 0:
                # Color removed
                pass 
            colors[edge] = new_color
            freq[new_color] += 1
            
            current_count = sum(1 for f in freq if f > 0)
            if current_count > best_count:
                best_count = current_count
                best_colors = colors[:]
    
    return best_colors
