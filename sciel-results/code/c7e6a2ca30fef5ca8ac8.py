import time
import random
import numpy as np
from itertools import combinations, permutations

def get_edge_index(i, j, n):
    if i > j:
        i, j = j, i
    return sum(n - 1 - k for k in range(i)) + (j - i - 1)

def precompute_cycles(n):
    """Precompute all 6-cycles and return as a numpy array of edge indices."""
    base_cycles = set()
    for perm in permutations([1, 2, 3, 4, 5]):
        cycle = (0, perm[0], perm[1], perm[2], perm[3], perm[4])
        rev = (0, perm[4], perm[3], perm[2], perm[1], perm[0])
        canon = min(cycle, rev)
        base_cycles.add(canon)
    
    cycles_list = []
    
    for comb in combinations(range(n), 6):
        mapping = {k: v for k, v in enumerate(comb)}
        for b_cycle in base_cycles:
            mapped_cycle = tuple(mapping[v] for v in b_cycle)
            edges = []
            for k in range(6):
                u = mapped_cycle[k]
                v = mapped_cycle[(k + 1) % 6]
                idx = get_edge_index(u, v, n)
                edges.append(idx)
            cycles_list.append(tuple(edges))
            
    return np.array(cycles_list, dtype=np.int32)

def construct():
    start_time = time.time()
    deadline = 140.0  # Keep safety margin before 150s kill
    
    # Try n from 17 down to 9
    for n in range(17, 8, -1):
        if time.time() > deadline:
            break
            
        num_edges = n * (n - 1) // 2
        cycles = precompute_cycles(n)
        num_cycles = len(cycles)
        
        # If n is small, cycles might be empty or very few, but for n>=6 it's fine.
        
        best_penalty = float('inf')
        best_colors = None
        
        def compute_penalty(colors):
            """Count monochromatic 6-cycles using numpy."""
            if num_cycles == 0:
                return 0
            col_edges = colors[cycles]
            # Check if all 6 edges in each cycle have the same color
            # col_edges shape: (M, 6)
            # We compare each column to the first column
            is_mono = np.all(col_edges == col_edges[:, 0:1], axis=1)
            return np.sum(is_mono)
        
        # Stochastic Local Search
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization
            colors = np.random.randint(0, 4, size=num_edges, dtype=np.int32)
            cur_penalty = compute_penalty(colors)
            
            if cur_penalty < best_penalty:
                best_penalty = cur_penalty
                best_colors = colors.copy()
            
            if cur_penalty == 0:
                break
            
            # Local search
            # We will try to improve by changing one edge at a time
            # To speed up, we limit the number of steps per attempt
            max_steps = 2000
            for step in range(max_steps):
                if time.time() > deadline:
                    break
                
                # Pick a random edge
                idx = np.random.randint(0, num_edges)
                old_color = colors[idx]
                
                # Try all other colors
                best_new_penalty = cur_penalty
                best_new_color = old_color
                
                for c in range(4):
                    if c == old_color:
                        continue
                    
                    colors[idx] = c
                    new_penalty = compute_penalty(colors)
                    
                    if new_penalty < best_new_penalty:
                        best_new_penalty = new_penalty
                        best_new_color = c
                
                # Apply best move
                if best_new_color != old_color:
                    colors[idx] = best_new_color
                    cur_penalty = best_new_penalty
                    
                    if cur_penalty < best_penalty:
                        best_penalty = cur_penalty
                        best_colors = colors.copy()
                    
                    if cur_penalty == 0:
                        break
                else:
                    # No improvement found for this edge.
                    # Occasionally accept a random move to escape local minima
                    if random.random() < 0.1:
                        new_c = random.randint(0, 3)
                        if new_c != old_color:
                            colors[idx] = new_c
                            cur_penalty = compute_penalty(colors)
                            if cur_penalty < best_penalty:
                                best_penalty = cur_penalty
                                best_colors = colors.copy()
                            if cur_penalty == 0:
                                break
            
            if cur_penalty == 0:
                break
        
        if best_penalty == 0:
            return {'n': n, 'colors': best_colors.tolist()}
        
        # If we didn't find a solution for this n, continue to smaller n
        # But we should keep the best found so far? No, we want max n.
        # So if 17 fails, we try 16. If 16 succeeds, we return 16.
        # We don't need to keep the invalid 17 solution.
        
    # Fallback: known good solution for n=9
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
