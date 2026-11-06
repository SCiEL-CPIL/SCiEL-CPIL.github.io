import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin
    
    n = 128
    
    # Precompute all nonconstant 4-term APs: (a, a+d, a+2d, a+3d) with 1<=a, a+3d<=128, d>=1
    aps = []
    for d in range(1, 43):  # max d: a+3d <= 128 => a <= 128-3d, need a>=1
        for a in range(1, 128 - 3*d + 1):
            aps.append((a-1, a-1+d, a-1+2*d, a-1+3*d))  # 0-indexed
    
    num_aps = len(aps)
    
    # Try with increasing number of colors: 3, 4, 5, ...
    for num_colors in [3, 4, 5, 6, 7, 8]:
        if time.time() > deadline - 5:
            break
        
        best_found = None
        attempts = 0
        
        while time.time() < deadline - 5:
            attempts += 1
            
            # Initialize randomly
            colors = np.random.randint(0, num_colors, size=n)
            
            # Local search: minimize number of violated APs
            # An AP is violated if it uses fewer than 3 distinct colors
            
            def count_violations(c):
                # For each AP, check if it has < 3 distinct colors
                viol = 0
                for (i, j, k, l) in aps:
                    distinct = len({c[i], c[j], c[k], c[l]})
                    if distinct < 3:
                        viol += 1
                return viol
            
            # Faster violation count using vectorization
            def count_violations_fast(c):
                # c is numpy array
                viol = 0
                for (i, j, k, l) in aps:
                    # Check if all same (1 color) or exactly 2 colors
                    vals = [c[i], c[j], c[k], c[l]]
                    if len(set(vals)) < 3:
                        viol += 1
                return viol
            
            current_viol = count_violations_fast(colors)
            
            if current_viol == 0:
                # Valid coloring found!
                used = set(colors.tolist())
                if best_found is None or len(used) < len(best_found[1]):
                    best_found = (colors.copy(), used)
                break
            
            # Simulated annealing / hill climbing
            temp = 2.0
            min_temp = 0.01
            cooling = 0.995
            
            for step in range(50000):
                if time.time() > deadline - 5:
                    break
                
                # Pick a random position and change its color
                pos = random.randint(0, n - 1)
                old_color = colors[pos]
                new_color = random.randint(0, num_colors - 1)
                if new_color == old_color:
                    continue
                
                colors[pos] = new_color
                
                # Compute delta in violations
                # Only APs containing pos are affected
                # Find APs containing pos
                delta_viol = 0
                for (i, j, k, l) in aps:
                    if pos in (i, j, k, l):
                        # Before
                        before = len({colors[i] if i != pos else old_color,
                                      colors[j] if j != pos else old_color,
                                      colors[k] if k != pos else old_color,
                                      colors[l] if l != pos else old_color})
                        # After
                        after = len({colors[i], colors[j], colors[k], colors[l]})
                        before_bad = 1 if before < 3 else 0
                        after_bad = 1 if after < 3 else 0
                        delta_viol += after_bad - before_bad
                
                if delta_viol <= 0:
                    current_viol += delta_viol
                    if current_viol == 0:
                        used = set(colors.tolist())
                        if best_found is None or len(used) < len(best_found[1]):
                            best_found = (colors.copy(), used)
                        break
                else:
                    # Accept with probability exp(-delta/temp)
                    if random.random() < np.exp(-delta_viol / temp):
                        current_viol += delta_viol
                    else:
                        colors[pos] = old_color
                
                temp *= cooling
                if temp < min_temp:
                    temp = min_temp
            
            if current_viol == 0:
                used = set(colors.tolist())
                if best_found is None or len(used) < len(best_found[1]):
                    best_found = (colors.copy(), used)
                break
            
            # If we've made many attempts, try a different strategy
            if attempts % 10 == 0:
                # Try a structured initialization
                # Use a periodic pattern
                period = random.choice([3, 4, 5, 6, 7, 8])
                pattern = list(range(num_colors))
                random.shuffle(pattern)
                # Create a repeating pattern
                colors = np.array([pattern[i % len(pattern)] for i in range(n)])
                # Perturb
                for _ in range(n // 2):
                    pos = random.randint(0, n - 1)
                    colors[pos] = random.randint(0, num_colors - 1)
                current_viol = count_violations_fast(colors)
    
    # If we found a valid coloring, return it
    if best_found is not None:
        c, used = best_found
        # Relabel to use consecutive small labels
        unique_colors = sorted(used)
        mapping = {old: new for new, old in enumerate(unique_colors)}
        result = [mapping[c[i]] for i in range(n)]
        return result
    
    # Fallback: use a simple 3-color pattern that might work
    # Pattern: 0,1,2,0,1,2,...
    # Check if this works
    colors = [i % 3 for i in range(n)]
    # Verify
    valid = True
    for (i, j, k, l) in aps:
        if len({colors[i], colors[j], colors[k], colors[l]}) < 3:
            valid = False
            break
    
    if valid:
        return colors
    
    # Last resort: use many colors, assign each position a unique-ish color
    # This always works but uses many colors
    # Actually, let's try a simple approach: use 4 colors in a pattern
    for nc in [4, 5, 6, 8, 10]:
        colors = [i % nc for i in range(n)]
        valid = True
        for (i, j, k, l) in aps:
            if len({colors[i], colors[j], colors[k], colors[l]}) < 3:
                valid = False
                break
        if valid:
            return colors
    
    # Absolute fallback: all distinct colors (always valid)
    return list(range(n))
