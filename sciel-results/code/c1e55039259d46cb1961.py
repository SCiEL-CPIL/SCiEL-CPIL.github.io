import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 110  # 110s out of 120s
    
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: idx for idx, e in enumerate(edges)}
    
    # Precompute K4s
    k4s = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    e1 = (a, b)
                    e2 = (a, c)
                    e3 = (a, d)
                    e4 = (b, c)
                    e5 = (b, d)
                    e6 = (c, d)
                    k4s.append([edge_index[e1], edge_index[e2], edge_index[e3], 
                                edge_index[e4], edge_index[e5], edge_index[e6]])
    
    def count_bad_k4s(colors):
        bad = 0
        for k4 in k4s:
            c0 = colors[k4[0]]
            c1 = colors[k4[1]]
            c2 = colors[k4[2]]
            c3 = colors[k4[3]]
            c4 = colors[k4[4]]
            c5 = colors[k4[5]]
            # Check if distinct colors < 5
            # Use a set for simplicity, it's fast enough for 6 elements
            if len({c0, c1, c2, c3, c4, c5}) < 5:
                bad += 1
        return bad

    def local_search(k_colors, time_limit):
        # Initialize
        colors = [random.randint(0, k_colors - 1) for _ in range(len(edges))]
        current_bad = count_bad_k4s(colors)
        best_colors = colors[:]
        best_bad = current_bad
        
        # Simulated Annealing parameters
        T = 1.0
        cooling = 0.9999
        min_T = 0.01
        
        while time.time() - start_time < deadline - 2: # Keep 2s margin
            if current_bad == 0:
                break
            
            # Pick random edge
            e_idx = random.randint(0, len(edges) - 1)
            old_color = colors[e_idx]
            new_color = random.randint(0, k_colors - 1)
            if old_color == new_color:
                continue
            
            colors[e_idx] = new_color
            new_bad = count_bad_k4s(colors)
            
            delta = new_bad - current_bad
            if delta <= 0 or random.random() < pow(2.718, -delta / T):
                current_bad = new_bad
                if current_bad < best_bad:
                    best_bad = current_bad
                    best_colors = colors[:]
            else:
                colors[e_idx] = old_color
            
            T *= cooling
            if T < min_T:
                T = min_T
        
        return best_colors, best_bad

    # Try 13, 14, 15 colors
    for k in [13, 14, 15]:
        if time.time() - start_time > deadline - 10:
            break
        colors, bad = local_search(k, 5) # 5s slice
        if bad == 0:
            # Remap to 0..k-1 is already done
            return colors
    
    # If all fail, return a seed-like result or the best found
    # The seed uses (i+j)%17. Let's just return that to be safe, though it's 17 colors.
    # But we should return the best valid one if we found one with bad>0? No, invalid scores 0.
    # If we didn't find a valid one, we must return something.
    # The prompt says "A valid colouring certifies...". If invalid, score 0.
    # So we MUST return a valid coloring.
    # If local search fails to find valid with 15, we might need to fall back to a known valid one.
    # The seed (i+j)%17 is valid.
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
