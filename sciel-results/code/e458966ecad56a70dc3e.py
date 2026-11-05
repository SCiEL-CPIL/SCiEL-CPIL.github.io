import random
import time
import math

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    n = 16
    # Precompute all quads and their edge indices
    quads = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    quads.append((i, j, k, l))
    
    def get_violation_count(colors):
        """Count number of quads with fewer than 5 colors."""
        v = 0
        for (i, j, k, l) in quads:
            c = {colors[i][j], colors[i][k], colors[i][l], colors[j][k], colors[j][l], colors[k][l]}
            if len(c) < 5:
                v += 1
        return v
    
    def is_valid_fast(colors):
        return get_violation_count(colors) == 0
    
    def flatten(colors):
        res = []
        for i in range(n):
            for j in range(i+1, n):
                res.append(colors[i][j])
        return res
    
    # Try k from 13 to 15
    for k in range(13, 16):
        # Multiple restarts with simulated annealing
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            colors = [[0]*n for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    colors[i][j] = random.randint(0, k-1)
                    colors[j][i] = colors[i][j]
            
            # Check initial
            if is_valid_fast(colors):
                return flatten(colors)
            
            current_violations = get_violation_count(colors)
            best_violations = current_violations
            best_colors = [row[:] for row in colors]
            
            # Simulated annealing
            T = 2.0
            T_min = 0.01
            cooling = 0.9995
            
            while T > T_min and time.time() < deadline - 5:
                # Pick a random edge
                i = random.randint(0, n-1)
                j = random.randint(0, n-1)
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                    
                old_c = colors[i][j]
                new_c = random.randint(0, k-1)
                if new_c == old_c:
                    continue
                
                colors[i][j] = new_c
                colors[j][i] = new_c
                
                new_violations = get_violation_count(colors)
                
                if new_violations <= current_violations:
                    current_violations = new_violations
                    if current_violations == 0:
                        return flatten(colors)
                    if current_violations < best_violations:
                        best_violations = current_violations
                        best_colors = [row[:] for row in colors]
                else:
                    # Accept with probability exp(-delta/T)
                    delta = new_violations - current_violations
                    if random.random() < math.exp(-delta / T):
                        current_violations = new_violations
                    else:
                        # Revert
                        colors[i][j] = old_c
                        colors[j][i] = old_c
                
                T *= cooling
            
            # If we found a valid solution in best, return it
            if best_violations == 0:
                return flatten(best_colors)
    
    # Fallback: Return the seed solution (17 colors) if no better found
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
