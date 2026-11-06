import time
import random
import itertools

N = 16

# Precompute edges in lexicographic order
edges = []
edge_to_idx = {}
for i in range(N):
    for j in range(i + 1, N):
        edge_to_idx[(i, j)] = len(edges)
        edges.append((i, j))

# Precompute K4s and their edge indices
K4_EDGE_INDICES = []
for combo in itertools.combinations(range(N), 4):
    a, b, c, d = combo
    e_idxs = [
        edge_to_idx[(a, b)],
        edge_to_idx[(a, c)],
        edge_to_idx[(a, d)],
        edge_to_idx[(b, c)],
        edge_to_idx[(b, d)],
        edge_to_idx[(c, d)]
    ]
    K4_EDGE_INDICES.append(e_idxs)

NUM_K4 = len(K4_EDGE_INDICES)

def get_seed_colors():
    colors = []
    for i in range(N):
        for j in range(i + 1, N):
            if i == 15 or j == 15:
                colors.append(j if i == 15 else i)
            else:
                colors.append((8 * (i + j)) % 15)
    return colors

def is_valid(col_list):
    for e_idxs in K4_EDGE_INDICES:
        if len(set(col_list[e] for e in e_idxs)) < 5:
            return False
    return True

def get_penalty(col_list):
    pen = 0
    for e_idxs in K4_EDGE_INDICES:
        if len(set(col_list[e] for e in e_idxs)) < 5:
            pen += 1
    return pen

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    seed_colors = get_seed_colors()
    
    # Try to find a coloring with fewer colors
    for K in [14, 13]:
        if time.time() > deadline - 5:
            break
            
        # Run a few restarts
        num_restarts = 3
        time_per_restart = (deadline - 5 - time.time()) / num_restarts
        
        for r in range(num_restarts):
            if time.time() > deadline - 1:
                break
                
            colors = [random.randint(0, K - 1) for _ in range(120)]
            
            if is_valid(colors):
                return colors
            
            current_penalty = get_penalty(colors)
            
            t_start = time.time()
            stuck = 0
            
            while time.time() - t_start < time_per_restart:
                if current_penalty == 0:
                    return colors
                
                e_idx = random.randint(0, 119)
                old_c = colors[e_idx]
                new_c = random.randint(0, K - 1)
                
                if new_c == old_c:
                    continue
                
                colors[e_idx] = new_c
                new_penalty = get_penalty(colors)
                
                if new_penalty < current_penalty:
                    current_penalty = new_penalty
                    stuck = 0
                else:
                    colors[e_idx] = old_c
                    stuck += 1
                    
                if stuck > 500:
                    # Restart
                    colors = [random.randint(0, K - 1) for _ in range(120)]
                    current_penalty = get_penalty(colors)
                    stuck = 0
                    t_start = time.time()
                    
    return seed_colors
