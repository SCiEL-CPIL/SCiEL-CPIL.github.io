import math
import random
import time
import numpy as np

def score_set(U):
    if not U: return 0
    max_u = max(U)
    if max_u == 0: return 1.0
    
    n = len(U)
    # For small sets, pure Python set operations are faster than numpy overhead
    if n < 20:
        sum_set = set()
        diff_set = set()
        for a in U:
            for b in U:
                sum_set.add(a + b)
                diff_set.add(a - b)
        size_sum = len(sum_set)
        size_diff = len(diff_set)
    else:
        arr = np.array(U, dtype=np.int64)
        sums = arr[:, None] + arr[None, :]
        diffs = arr[:, None] - arr[None, :]
        size_sum = len(np.unique(sums))
        size_diff = len(np.unique(diffs))
        
    if size_sum == 0: return 1.0
    
    ratio = size_diff / size_sum
    denom = math.log(2 * max_u + 1)
    if denom == 0: return 1.0
    
    return 1 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin for 150s limit
    
    best_score = 0
    best_set = [0, 1, 3]
    
    # Initialize with diverse candidates
    # Small sets often have high ratios
    candidates = [
        [0, 1, 3],
        [0, 1, 2, 5],
        [0, 1, 4, 6],
        [0, 1, 2, 4, 7],
        [0, 1, 3, 4, 7],
        [0, 1, 2, 5, 6],
        [0, 1, 3, 7, 12], # Geometric-ish
        [0, 1, 2, 4, 8, 16],
        [0, 1, 3, 9, 27],
        [0, 1, 2, 3, 4, 5, 6], # Consecutive
        [0, 1, 3, 5, 7, 9], # Arithmetic
    ]
    
    for c in candidates:
        s = score_set(c)
        if s > best_score:
            best_score = s
            best_set = c[:]
            
    def get_neighbors(U):
        neighbors = []
        n = len(U)
        max_val = max(U) if U else 10
        
        # 1. Move an element to a random nearby value
        for i in range(n):
            # Try small shifts
            for delta in [-3, -2, -1, 1, 2, 3]:
                new_val = U[i] + delta
                if new_val >= 0:
                    new_U = U[:]
                    new_U[i] = new_val
                    if new_val not in new_U[:i] + new_U[i+1:]:
                         neighbors.append(sorted(new_U))
            
            # Try a larger random jump (keep 0 fixed if it's index 0)
            if i > 0 or U[i] != 0:
                # Jump within a reasonable range
                jump_range = max(10, max_val // 2)
                new_val = random.randint(0, max_val + jump_range)
                if new_val >= 0 and new_val != U[i]:
                    new_U = U[:]
                    new_U[i] = new_val
                    if new_val not in new_U[:i] + new_U[i+1:]:
                         neighbors.append(sorted(new_U))

        # 2. Add an element
        if n < 15:
            # Add a small element
            for val in range(0, max_val + 5):
                if val not in U:
                    new_U = U + [val]
                    neighbors.append(sorted(new_U))
            # Add a random larger element
            new_val = random.randint(0, max_val + 20)
            if new_val not in U:
                new_U = U + [new_val]
                neighbors.append(sorted(new_U))
                
        # 3. Remove an element
        if n > 2:
            for i in range(n):
                if U[i] == 0: continue
                new_U = U[:i] + U[i+1:]
                neighbors.append(sorted(new_U))
                
        return neighbors

    # Simulated Annealing
    current_set = best_set[:]
    current_score = best_score
    T = 0.05
    cooling = 0.9995
    
    steps = 0
    while time.time() < deadline:
        neighbors = get_neighbors(current_set)
        if not neighbors:
             # If stuck, try a random restart locally
             new_set = current_set[:]
             if len(new_set) > 1:
                 idx = random.randint(1, len(new_set)-1)
                 new_set[idx] = random.randint(0, max(new_set) + 10)
                 new_set = sorted(new_set)
             neighbors = [new_set]
             
        next_set = random.choice(neighbors)
        next_score = score_set(next_set)
        
        delta = next_score - current_score
        if delta > 0 or random.random() < math.exp(delta / T):
            current_set = next_set
            current_score = next_score
            if current_score > best_score:
                best_score = current_score
                best_set = current_set[:]
        
        T *= cooling
        steps += 1
        
        # Occasionally restart with a random set to avoid local minima
        if steps % 200 == 0:
             # Generate random set of size 3-10
             n = random.randint(3, 10)
             # Sample from a wider range
             vals = random.sample(range(0, 50), n)
             if 0 not in vals:
                 vals[0] = 0
             r_set = sorted(vals)
             r_score = score_set(r_set)
             if r_score > current_score:
                 current_set = r_set
                 current_score = r_score
                 if r_score > best_score:
                     best_score = r_score
                     best_set = r_set[:]

    return best_set
