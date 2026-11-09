import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit
    
    N = 1000
    max_sum = 2 * N
    
    # 1. Greedy Initial Solution
    chosen = []
    counts = [0] * (max_sum + 1)
    
    for x in range(1, N + 1):
        valid = True
        for y in chosen:
            s = x + y
            if counts[s] == 2:
                valid = False
                break
        if valid and counts[2 * x] == 2:
            valid = False
            
        if valid:
            chosen.append(x)
            counts[2 * x] += 1
            for y in chosen[:-1]:
                counts[x + y] += 1
                
    best_set = set(chosen)
    best_size = len(chosen)
    
    # 2. Local Search (Simulated Annealing)
    current = list(chosen)
    in_set = [False] * (N + 1)
    for x in current:
        in_set[x] = True
        
    # List of available elements for O(1) selection
    available = [x for x in range(1, N + 1) if not in_set[x]]
    
    # Re-initialize counts
    counts = [0] * (max_sum + 1)
    for i in range(len(current)):
        x = current[i]
        counts[2 * x] += 1
        for j in range(i + 1, len(current)):
            y = current[j]
            counts[x + y] += 1
            
    def get_conflicts(counts):
        c = 0
        for s in range(2, max_sum + 1):
            if counts[s] > 2:
                c += counts[s] - 2
        return c

    current_conflicts = get_conflicts(counts)
    best_conflicts = current_conflicts
    best_solution = list(current)
    
    T = 1.0
    cooling = 0.99995
    min_T = 0.001
    
    # Helper to calculate delta conflicts for a move
    def calc_delta_add(x, current, counts):
        delta = 0
        sums_to_add = []
        # 2x
        s = 2 * x
        if counts[s] == 2:
            delta += 1
        sums_to_add.append(s)
        # x + y
        for y in current:
            s = x + y
            if counts[s] == 2:
                delta += 1
            sums_to_add.append(s)
        return delta, sums_to_add

    def calc_delta_swap(x, y, current, counts):
        # x is in current, y is not
        removed_sums = set()
        removed_sums.add(2 * x)
        for z in current:
            if z != x:
                removed_sums.add(x + z)
                
        added_sums = set()
        added_sums.add(2 * y)
        for z in current:
            if z != x:
                added_sums.add(y + z)
                
        all_sums = removed_sums.union(added_sums)
        
        delta = 0
        for s in all_sums:
            old_c = counts[s]
            new_c = old_c
            if s in removed_sums:
                new_c -= 1
            if s in added_sums:
                new_c += 1
            
            old_conf = max(0, old_c - 2)
            new_conf = max(0, new_c - 2)
            delta += new_conf - old_conf
            
        sums_to_dec = list(removed_sums)
        sums_to_inc = list(added_sums)
        
        return delta, sums_to_dec, sums_to_inc

    while time.time() < deadline:
        if random.random() < 0.5:
            # Add Move
            if not available:
                continue
            idx = random.randint(0, len(available) - 1)
            x = available[idx]
            
            delta, sums_to_add = calc_delta_add(x, current, counts)
            
            if delta == 0 or (delta > 0 and random.random() < math.exp(-delta / T)):
                # Accept
                in_set[x] = True
                current.append(x)
                available.pop(idx)
                
                for s in sums_to_add:
                    counts[s] += 1
                current_conflicts += delta
                
                if current_conflicts < best_conflicts:
                    best_conflicts = current_conflicts
                    if best_conflicts == 0:
                        best_solution = list(current)
                        if len(current) > best_size:
                            best_size = len(current)
        else:
            # Swap Move
            if not current or not available:
                continue
            x_idx = random.randint(0, len(current) - 1)
            x = current[x_idx]
            y_idx = random.randint(0, len(available) - 1)
            y = available[y_idx]
            
            delta, sums_to_dec, sums_to_inc = calc_delta_swap(x, y, current, counts)
            
            if delta < 0 or (delta == 0 and random.random() < 0.5) or (delta > 0 and random.random() < math.exp(-delta / T)):
                # Accept
                in_set[x] = False
                in_set[y] = True
                current[x_idx] = y
                available[y_idx] = x
                
                for s in sums_to_dec:
                    counts[s] -= 1
                for s in sums_to_inc:
                    counts[s] += 1
                    
                current_conflicts += delta
                
                if current_conflicts < best_conflicts:
                    best_conflicts = current_conflicts
                    if best_conflicts == 0:
                        best_solution = list(current)
                        if len(current) > best_size:
                            best_size = len(current)
                            
        T *= cooling
        if T < min_T:
            T = min_T

    # Final check
    final_set = set(best_solution)
    final_counts = [0] * (max_sum + 1)
    for i in range(len(best_solution)):
        x = best_solution[i]
        final_counts[2 * x] += 1
        for j in range(i + 1, len(best_solution)):
            y = best_solution[j]
            final_counts[x + y] += 1
            
    valid = True
    for s in range(2, max_sum + 1):
        if final_counts[s] > 2:
            valid = False
            break
            
    if valid:
        return best_solution
    else:
        return chosen

# EVOLVE-BLOCK-END
