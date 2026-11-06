import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

# Precompute edge indices
edges = []
edge_to_idx = {}
for i in range(N):
    for j in range(i + 1, N):
        edge_to_idx[(i, j)] = len(edges)
        edges.append((i, j))

# Precompute K4s and the edges within them
k4s = []
k4_edges = []  # List of tuples of 6 edge indices
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                k4s.append((a, b, c, d))
                e_ab = edge_to_idx[(a, b)]
                e_ac = edge_to_idx[(a, c)]
                e_ad = edge_to_idx[(a, d)]
                e_bc = edge_to_idx[(b, c)]
                e_bd = edge_to_idx[(b, d)]
                e_cd = edge_to_idx[(c, d)]
                k4_edges.append((e_ab, e_ac, e_ad, e_bc, e_bd, e_cd))

NUM_K4 = len(k4s)

# Map edge index to list of K4 indices it participates in
edge_to_k4s = [[] for _ in range(NUM_EDGES)]
for k4_idx, edge_list in enumerate(k4_edges):
    for e_idx in edge_list:
        edge_to_k4s[e_idx].append(k4_idx)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin before 150s kill
    
    # Known valid 17-coloring
    seed_colors = np.array([(i + j) % 17 for i in range(N) for j in range(i + 1, N)], dtype=np.int32)
    
    best_result = None
    best_K = 17
    
    # Try to find valid colorings with fewer colors
    for K in [15, 14, 13]:
        if time.time() > deadline - 15:
            break
        
        found_valid = False
        num_restarts = 50
        remaining = deadline - time.time() - 5
        time_per_restart = remaining / max(1, num_restarts)
        
        for r in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with K colors
            colors = np.random.randint(0, K, size=NUM_EDGES, dtype=np.int32)
            
            # Compute initial penalty
            current_penalty = 0
            for k4_e in k4_edges:
                n_col = len({colors[k4_e[0]], colors[k4_e[1]], colors[k4_e[2]], 
                             colors[k4_e[3]], colors[k4_e[4]], colors[k4_e[5]]})
                if n_col < 5:
                    current_penalty += (5 - n_col) ** 2
            
            if current_penalty == 0:
                best_result = colors.tolist()
                best_K = K
                found_valid = True
                break
            
            # Simulated Annealing
            T = 50.0
            T_min = 1.0
            cooling_rate = 0.99
            
            t_start = time.time()
            
            while time.time() - t_start < time_per_restart:
                # Pick a random edge to change
                e_idx = random.randint(0, NUM_EDGES - 1)
                old_c = colors[e_idx]
                
                # Pick a new color
                new_c = random.randint(0, K - 1)
                while new_c == old_c:
                    new_c = random.randint(0, K - 1)
                    
                # Calculate penalty change incrementally
                delta_penalty = 0
                for k4_idx in edge_to_k4s[e_idx]:
                    k4_e = k4_edges[k4_idx]
                    
                    # Old state
                    old_set = set()
                    for e in k4_e:
                        if e == e_idx:
                            old_set.add(old_c)
                        else:
                            old_set.add(colors[e])
                    old_distinct = len(old_set)
                    old_pen = (5 - old_distinct) ** 2 if old_distinct < 5 else 0
                    
                    # New state
                    new_set = set()
                    for e in k4_e:
                        if e == e_idx:
                            new_set.add(new_c)
                        else:
                            new_set.add(colors[e])
                    new_distinct = len(new_set)
                            new_pen = (5 - new_distinct) ** 2 if new_distinct < 5 else 0
                    
                    delta_penalty += new_pen - old_pen
                
                # Acceptance criterion
                if delta_penalty <= 0 or random.random() < np.exp(-delta_penalty / T):
                    colors[e_idx] = new_c
                    current_penalty += delta_penalty
                    
                    if current_penalty == 0:
                        best_result = colors.tolist()
                        best_K = K
                        found_valid = True
                        break
                    
                T *= cooling_rate
                if T < T_min:
                    T = T_min
            
            if found_valid:
                break
        
        if found_valid:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: return the seed (17 colors, valid)
    return seed_colors.tolist()
