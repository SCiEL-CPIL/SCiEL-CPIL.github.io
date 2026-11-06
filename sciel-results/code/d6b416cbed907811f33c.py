import time
import random
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 115.0
    
    # 1. Precompute 4-APs
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
            
    # 2. Precompute neighbors for delta calculation
    # neighbors[i] contains tuples (u, v, w) for each AP containing i
    neighbors = [[] for _ in range(n)]
    for ap in aps:
        for i in range(4):
            pos = ap[i]
            others = [ap[j] for j in range(4) if j != i]
            neighbors[pos].append(tuple(others))
            
    # 3. Helper functions
    def get_cost(color):
        # Vectorized cost calculation
        # Create array of indices for all APs
        # This is done once outside if possible, but aps is a list of tuples.
        # Let's convert aps to numpy array for vectorization
        pass # We'll do this in the main flow or precompute
        
    # Precompute APs as numpy array for vectorized cost
    aps_arr = np.array(aps, dtype=np.int32)
    
    def calc_cost(color):
        # color is 1D array of 0/1
        # Get colors for all APs
        c_aps = color[aps_arr]
        # Check if all 4 are equal
        # c_aps[:, 0] == c_aps[:, 1] == c_aps[:, 2] == c_aps[:, 3]
        mono = (c_aps[:, 0] == c_aps[:, 1]) & (c_aps[:, 1] == c_aps[:, 2]) & (c_aps[:, 2] == c_aps[:, 3])
        return np.sum(mono)
        
    def calc_deltas(color):
        deltas = np.zeros(n, dtype=np.int32)
        for i in range(n):
            ci = color[i]
            for u, v, w in neighbors[i]:
                cu = color[u]
                if cu == color[v] == color[w]:
                    if ci == cu:
                        deltas[i] -= 1
                    else:
                        deltas[i] += 1
        return deltas

    def local_search(initial_color):
        color = initial_color.copy()
        cost = calc_cost(color)
        deltas = calc_deltas(color)
        
        while time.time() < deadline:
            min_idx = np.argmin(deltas)
            min_val = deltas[min_idx]
            
            if min_val >= 0:
                break
            
            color[min_idx] = 1 - color[min_idx]
            cost += min_val
            deltas = calc_deltas(color)
            
        return color, cost

    best_color = None
    best_cost = float('inf')
    
    # Start with Thue-Morse
    thue_morse = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
    c, cost = local_search(thue_morse)
    if cost < best_cost:
        best_cost = cost
        best_color = c.copy()
        
    if time.time() > deadline - 10:
        return list(best_color)

    # Random Restarts
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        random_color = np.random.randint(0, 2, n, dtype=np.int8)
        c, cost = local_search(random_color)
        if cost < best_cost:
            best_cost = cost
            best_color = c.copy()
            
    # Perturbation of Best
    if time.time() < deadline - 5 and best_color is not None:
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            perturbed = best_color.copy()
            num_flips = random.randint(3, 10)
            indices = random.sample(range(n), num_flips)
            for idx in indices:
                perturbed[idx] = 1 - perturbed[idx]
                
            c, cost = local_search(perturbed)
            if cost < best_cost:
                best_cost = cost
                best_color = c.copy()

    return list(best_color)
