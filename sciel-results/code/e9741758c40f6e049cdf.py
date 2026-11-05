import time
import random
import math

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLOR = 120

# Precompute edge indices
edge_indices = {}
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        edge_indices[(i, j)] = idx
        idx += 1

# Precompute for each edge, the list of "other 3 edges" for each 4-cycle containing it.
cycles_for_edge = [[] for _ in range(NUM_EDGES)]

for u in range(N):
    for v in range(u + 1, N):
        for w in range(v + 1, N):
            for x in range(w + 1, N):
                e1 = edge_indices[(u, v)]
                e2 = edge_indices[(v, w)]
                e3 = edge_indices[(w, x)]
                e4 = edge_indices[(x, u)]
                cycles_for_edge[e1].append((e2, e3, e4))
                cycles_for_edge[e2].append((e1, e3, e4))
                cycles_for_edge[e3].append((e1, e2, e4))
                cycles_for_edge[e4].append((e1, e2, e3))
                
                e1 = edge_indices[(u, v)]
                e2 = edge_indices[(v, x)]
                e3 = edge_indices[(x, w)]
                e4 = edge_indices[(w, u)]
                cycles_for_edge[e1].append((e2, e3, e4))
                cycles_for_edge[e2].append((e1, e3, e4))
                cycles_for_edge[e3].append((e1, e2, e4))
                cycles_for_edge[e4].append((e1, e2, e3))
                
                e1 = edge_indices[(u, w)]
                e2 = edge_indices[(w, v)]
                e3 = edge_indices[(v, x)]
                e4 = edge_indices[(x, u)]
                cycles_for_edge[e1].append((e2, e3, e4))
                cycles_for_edge[e2].append((e1, e3, e4))
                cycles_for_edge[e3].append((e1, e2, e4))
                cycles_for_edge[e4].append((e1, e2, e3))

def is_valid_change(colors, e_idx, new_color):
    for e2, e3, e4 in cycles_for_edge[e_idx]:
        c2 = colors[e2]
        c3 = colors[e3]
        c4 = colors[e4]
        if c2 != c3 and c3 != c4 and c2 != c4:
            if new_color != c2 and new_color != c3 and new_color != c4:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Greedy Initialization
    colors = [0] * NUM_EDGES
    color_counts = [0] * MAX_COLOR
    color_counts[0] = NUM_EDGES
    used_colors = {0}
    
    # Try to introduce new colors greedily
    # Randomize edge order
    edge_order = list(range(NUM_EDGES))
    random.shuffle(edge_order)
    
    for e_idx in edge_order:
        # Try colors 1..119
        for c in range(1, MAX_COLOR):
            if is_valid_change(colors, e_idx, c):
                colors[e_idx] = c
                color_counts[0] -= 1
                if color_counts[0] == 0:
                    used_colors.discard(0)
                color_counts[c] += 1
                if color_counts[c] == 1:
                    used_colors.add(c)
                break
                
    best_score = len(used_colors)
    best_colors = colors[:]
    
    # Local Search
    current_score = best_score
    
    while time.time() < deadline:
        e_idx = random.randint(0, NUM_EDGES - 1)
        
        # Try to introduce a new color
        unused_colors = [c for c in range(MAX_COLOR) if color_counts[c] == 0]
        
        if unused_colors:
            new_color = random.choice(unused_colors)
            if colors[e_idx] != new_color and is_valid_change(colors, e_idx, new_color):
                old_color = colors[e_idx]
                colors[e_idx] = new_color
                color_counts[old_color] -= 1
                if color_counts[old_color] == 0:
                    used_colors.discard(old_color)
                color_counts[new_color] += 1
                used_colors.add(new_color)
                current_score = len(used_colors)
                if current_score > best_score:
                    best_score = current_score
                    best_colors = colors[:]
        else:
            # No unused colors, try to shuffle
            new_color = random.randint(0, MAX_COLOR - 1)
            if colors[e_idx] != new_color and is_valid_change(colors, e_idx, new_color):
                old_color = colors[e_idx]
                colors[e_idx] = new_color
                color_counts[old_color] -= 1
                if color_counts[old_color] == 0:
                    used_colors.discard(old_color)
                color_counts[new_color] += 1
                if color_counts[new_color] == 1:
                    used_colors.add(new_color)
                current_score = len(used_colors)
                if current_score > best_score:
                    best_score = current_score
                    best_colors = colors[:]
                    
    return best_colors
