"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time
import random
import math
sys.setrecursionlimit(10000)

def construct(n_rows, k):
    total_diffs = n_rows * (k * (k + 1) // 2)
    
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    def is_valid(rows):
        all_diffs = set()
        for row in rows:
            if len(row) != k + 1 or row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True
    
    def get_max_val(rows):
        return max(max(row) for row in rows)
    
    greedy_rows = greedy_construct()
    
    # Local search with simulated annealing
    random.seed(42)
    current_rows = [row[:] for row in greedy_rows]
    current_max = get_max_val(current_rows)
    best_rows = [row[:] for row in current_rows]
    best_max = current_max
    
    start = time.time()
    max_time = 100
    temperature = 50.0
    cooling = 0.999
    
    while time.time() - start < max_time and temperature > 0.5:
        r_idx = random.randint(0, n_rows - 1)
        pos = random.randint(1, k)
        
        old_val = current_rows[r_idx][pos]
        
        # Generate candidate: try to reduce max by moving towards center
        # or try a random perturbation
        if random.random() < 0.5:
            # Try to reduce: pick a value between neighbors
            lo = current_rows[r_idx][pos - 1] + 1
            hi = current_rows[r_idx][pos + 1] - 1 if pos < k else current_max + 5
            if lo > hi:
                continue
            new_val = random.randint(lo, hi)
        else:
            # Random perturbation
            new_val = random.randint(1, current_max + 20)
            if pos > 0 and new_val <= current_rows[r_idx][pos - 1]:
                continue
            if pos < k and new_val >= current_rows[r_idx][pos + 1]:
                continue
        
        current_rows[r_idx][pos] = new_val
        
        if is_valid(current_rows):
            new_max = get_max_val(current_rows)
            if new_max < best_max:
                best_max = new_max
                best_rows = [row[:] for row in current_rows]
            
            if new_max <= current_max:
                current_max = new_max
            else:
                delta = new_max - current_max
                if random.random() < math.exp(-delta / max(temperature, 0.01)):
                    current_max = new_max
                else:
                    current_rows[r_idx][pos] = old_val
        else:
            current_rows[r_idx][pos] = old_val
        
        temperature *= cooling
    
    # Final verification
    if not is_valid(best_rows):
        return greedy_rows
    
    return best_rows

# EVOLVE-BLOCK-END
