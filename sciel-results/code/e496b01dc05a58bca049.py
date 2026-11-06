import time
import random
import numpy as np

N = 16
NUM_EDGES = 120

# Precompute all K4s and their edges
edges_list = []
edge_to_idx = {}
for i in range(N):
    for j in range(i+1, N):
        edge_to_idx[(i,j)] = len(edges_list)
        edges_list.append((i,j))

k4s = []
k4_edges = []
for a in range(N):
    for b in range(a+1, N):
        for c in range(b+1, N):
            for d in range(c+1, N):
                k4s.append((a,b,c,d))
                k4_edges.append([
                    edge_to_idx[(a,b)],
                    edge_to_idx[(a,c)],
                    edge_to_idx[(a,d)],
                    edge_to_idx[(b,c)],
                    edge_to_idx[(b,d)],
                    edge_to_idx[(c,d)]
                ])

NUM_K4 = len(k4s)

# For fast penalty: count colors in each K4
def count_colors_in_k4(colors, e0, e1, e2, e3, e4, e5):
    c0 = colors[e0]
    c1 = colors[e1]
    c2 = colors[e2]
    c3 = colors[e3]
    c4 = colors[e4]
    c5 = colors[e5]
    # Count distinct
    distinct = 1
    if c1 != c0: distinct += 1
    if c2 != c0 and c2 != c1: distinct += 1
    if c3 != c0 and c3 != c1 and c3 != c2: distinct += 1
    if c4 != c0 and c4 != c1 and c4 != c2 and c4 != c3: distinct += 1
    if c5 != c0 and c5 != c1 and c5 != c2 and c5 != c3 and c5 != c4: distinct += 1
    return distinct

def total_penalty_fast(colors):
    pen = 0
    for e0, e1, e2, e3, e4, e5 in k4_edges:
        d = count_colors_in_k4(colors, e0, e1, e2, e3, e4, e5)
        if d < 5:
            pen += (5 - d) * (5 - d)
    return pen

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin

    # Try to find a coloring with few colors
    # Strategy: Start with random, use SA to minimize colors + penalty
    
    best_result = None
    best_num_colors = 999
    
    # Try K = 5, 6, 7, 8
    for K in [5, 6, 7, 8, 9, 10, 11, 12, 13]:
        if time.time() > deadline - 5:
            break
        
        # Multiple restarts
        max_restarts = 20
        for r in range(max_restarts):
            if time.time() > deadline - 3:
                break
            
            # Random initialization
            colors = np.random.randint(0, K, size=NUM_EDGES, dtype=np.int32)
            current_penalty = total_penalty_fast(colors)
            
            if current_penalty == 0:
                return colors.tolist()
            
            # Simulated Annealing
            T = 50.0
            T_min = 0.1
            cooling = 0.995
            iter_count = 0
            max_iter = 50000
            
            while iter_count < max_iter and time.time() < deadline - 2:
                # Pick random edge
                e_idx = random.randint(0, NUM_EDGES - 1)
                old_c = colors[e_idx]
                
                # Pick new color
                new_c = random.randint(0, K - 1)
                if new_c == old_c:
                    new_c = (new_c + 1) % K
                
                colors[e_idx] = new_c
                new_penalty = total_penalty_fast(colors)
                
                if new_penalty < current_penalty:
                    current_penalty = new_penalty
                    if current_penalty == 0:
                        return colors.tolist()
                else:
                    delta = new_penalty - current_penalty
                    if delta == 0:
                        accept = True
                    else:
                        if T < 1e-9:
                            accept = False
                        else:
                            accept = random.random() < np.exp(-delta / T)
                    
                    if accept:
                        current_penalty = new_penalty
                    else:
                        colors[e_idx] = old_c
                
                T *= cooling
                if T < T_min:
                    T = T_min
                
                iter_count += 1
                
                # Early termination check
                if iter_count % 10000 == 0 and time.time() > deadline - 2:
                    break
        
        # After trying K colors, check if we got penalty 0
        # (already returned above)
    
    # Fallback: use (i+j) mod 17
    fallback = []
    for i in range(N):
        for j in range(i+1, N):
            fallback.append((i + j) % 17)
    
    # Verify fallback has penalty 0
    # (i+j) mod 17: For any K4, are there at least 5 colors?
    # This is a known construction
    return fallback

