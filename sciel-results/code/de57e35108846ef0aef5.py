import time
import random
import numpy as np

def construct(n_rows, k):
    deadline = time.time() + 140  # safety margin before 150s kill
    n = k + 1  # number of elements per row
    n_diffs_per_row = n * (n - 1) // 2
    total_diffs = n_rows * n_diffs_per_row
    
    # Initialize with greedy solution
    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < n:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    rows = greedy_init()
    best_rows = [list(r) for r in rows]
    best_score = max(max(r) for r in best_rows)
    
    # Score function: maximize value = collisions * BIG + max_entry
    BIG = 1000000
    
    def compute_score(rows):
        """Return (max_entry, num_collisions)"""
        diffs = []
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.append(row[j] - row[i])
        max_entry = max(max(r) for r in rows)
        num_collisions = len(diffs) - len(set(diffs))
        return max_entry, num_collisions
    
    def score(rows):
        max_e, coll = compute_score(rows)
        return max_e + coll * BIG
    
    current_score = score(rows)
    
    # Simulated annealing
    T = 50.0
    T_min = 0.1
    cooling = 0.9995
    
    random.seed(42)
    np.random.seed(42)
    
    while time.time() < deadline and T > T_min:
        # Pick a random row and a random element to modify
        ri = random.randint(0, n_rows - 1)
        # Don't change index 0 (must be 0)
        ei = random.randint(1, n - 1)
        
        old_val = rows[ri][ei]
        
        # Generate new value: random perturbation
        # Keep it increasing: must be > rows[ri][ei-1] and < rows[ri][ei+1] if exists
        lo = rows[ri][ei - 1] + 1
        hi = rows[ri][ei + 1] - 1 if ei < n - 1 else best_score + 20
        
        if hi < lo:
            hi = lo + 10
        
        # Bias toward smaller values
        new_val = random.randint(lo, hi)
        
        if new_val == old_val:
            continue
        
        rows[ri][ei] = new_val
        
        new_score = score(rows)
        delta = new_score - current_score
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_score = new_score
            max_e, _ = compute_score(rows)
            if max_e < best_score and _ == 0:
                best_score = max_e
                best_rows = [list(r) for r in rows]
        else:
            rows[ri][ei] = old_val
        
        T *= cooling
    
    # Check if best is valid
    def is_valid(rows):
        diffs = []
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.append(row[j] - row[i])
        return len(diffs) == len(set(diffs))
    
    if is_valid(best_rows):
        return best_rows
    
    # Fallback: return greedy (always valid)
    return greedy_init()
