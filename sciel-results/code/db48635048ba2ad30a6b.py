import random
import time
import numpy as np

def construct():
    N = 128
    
    # Precompute all 4-term arithmetic progressions
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    # Function to calculate the number of bad progressions
    def count_bad(coloring):
        bad_count = 0
        for ap in aps:
            c0 = coloring[ap[0]]
            c1 = coloring[ap[1]]
            c2 = coloring[ap[2]]
            c3 = coloring[ap[3]]
            
            # Check if distinct count < 3
            if c0 == c1 == c2 == c3:
                bad_count += 1
            elif c0 == c1 == c2 or c0 == c1 == c3 or c0 == c2 == c3 or c1 == c2 == c3:
                bad_count += 1
            else:
                if len({c0, c1, c2, c3}) == 2:
                    bad_count += 1
        return bad_count

    def solve(k_colors, max_iter=30000, start_temp=1.0, end_temp=0.01):
        current = [random.randint(0, k_colors - 1) for _ in range(N)]
        current_bad = count_bad(current)
        
        if current_bad == 0:
            return current
        
        best = current.copy()
        best_bad = current_bad
        
        T = start_temp
        cooling = (end_temp / start_temp) ** (1.0 / max_iter)
        
        for i in range(max_iter):
            idx = random.randint(0, N - 1)
            old_color = current[idx]
            new_color = random.randint(0, k_colors - 1)
            
            if new_color == old_color:
                continue
                
            current[idx] = new_color
            new_bad = count_bad(current)
            delta = new_bad - current_bad
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_bad = new_bad
                if current_bad < best_bad:
                    best_bad = current_bad
                    best = current.copy()
                    if best_bad == 0:
                        break
            else:
                current[idx] = old_color
            
            T *= cooling
            
        return best if best_bad == 0 else None

    found_solution = None
    start_time = time.time()
    time_limit = 55.0

    # Try k=3
    for _ in range(5):
        if time.time() - start_time > time_limit:
            break
        sol = solve(3, max_iter=20000)
        if sol:
            found_solution = sol
            break
            
    if not found_solution:
        # Try k=4
        for _ in range(5):
            if time.time() - start_time > time_limit:
                break
            sol = solve(4, max_iter=20000)
            if sol:
                found_solution = sol
                break
                
    if not found_solution:
        # Try k=5
        for _ in range(5):
            if time.time() - start_time > time_limit:
                break
            sol = solve(5, max_iter=20000)
            if sol:
                found_solution = sol
                break

    if not found_solution:
        # Fallback: Greedy
        colors = []
        for x in range(1, 129):
            for color in range(128):
                valid = True
                for d in range(1, (x - 1) // 3 + 1):
                    idxs = [x - 1 - j * d for j in (1, 2, 3)]
                    existing = [colors[i] for i in idxs]
                    if len(set(existing + [color])) < 3:
                        valid = False
                        break
                if valid:
                    colors.append(color)
                    break
            else:
                colors.append(0)
        found_solution = colors

    return found_solution
