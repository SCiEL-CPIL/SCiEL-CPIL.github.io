import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    best_score = -1
    best_set = [0]
    
    def compute_score(U):
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        
        n = len(U)
        if n <= 30:
            sumset = set()
            diffset = set()
            U_list = list(U)
            for a in U_list:
                for b in U_list:
                    sumset.add(a + b)
                    diffset.add(a - b)
            n_sum = len(sumset)
            n_diff = len(diffset)
        else:
            U_arr = np.array(U, dtype=np.int64)
            sums = U_arr[:, None] + U_arr[None, :]
            sumset = set(np.unique(sums).tolist())
            diffs = U_arr[:, None] - U_arr[None, :]
            diffset = set(np.unique(diffs).tolist())
            n_sum = len(sumset)
            n_diff = len(diffset)
        
        if n_sum == 0:
            return -1
        
        base = 2 * max_u + 1
        if base <= 1:
            return 0
        
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        score = 1 + math.log(ratio) / math.log(base)
        return score

    def try_set(cand):
        nonlocal best_score, best_set
        cand = sorted(set(cand))
        if not cand or 0 not in cand:
            return -1
        if max(cand) == 0:
            return -1
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
        return s

    # Initial candidates
    candidates = []
    
    # Power of 2 minus 1
    for k in range(1, 8):
        s = [0]
        for i in range(1, k+1):
            s.append((1 << i) - 1)
        candidates.append(s)
        
    # Arithmetic progressions
    for n in range(2, 10):
        for step in range(1, 5):
            s = [0]
            for i in range(1, n):
                s.append(i * step)
            candidates.append(s)
            
    # Random sparse sets
    random.seed(42)
    for _ in range(50):
        n = random.randint(3, 8)
        max_val = random.randint(10, 50)
        s = {0}
        for _ in range(n-1):
            s.add(random.randint(1, max_val))
        candidates.append(sorted(s))
        
    # Add the known good set
    candidates.append([0, 1, 3, 7, 15])
    candidates.append([0, 1, 3, 7])

    # Local search
    current = list(best_set)
    if not current or max(current) == 0:
        current = candidates[0]
        try_set(current)
        
    no_improve = 0
    max_no_improve = 100
    
    while time.time() < deadline - 5 and no_improve < max_no_improve:
        improved = False
        cur_score = best_score
        max_cur = max(current)
        
        # Generate neighbors
        neighbors = set()
        
        # 1. Replace an element
        for i in range(1, len(current)):
            old_el = current[i]
            # Try nearby values
            for delta in range(-5, 15):
                new_el = old_el + delta
                if new_el < 1:
                    continue
                if new_el in current:
                    continue
                cand = list(current)
                cand[i] = new_el
                neighbors.add(tuple(sorted(cand)))
                
            # Try larger jumps
            for jump in [old_el * 2, old_el + max_cur // 2, max_cur + random.randint(1, 10)]:
                if jump < 1 or jump in current:
                    continue
                cand = list(current)
                cand[i] = jump
                neighbors.add(tuple(sorted(cand)))
                
        # 2. Add an element
        for _ in range(20):
            new_el = random.randint(1, max_cur + 20)
            if new_el not in current:
                cand = sorted(current + [new_el])
                neighbors.add(tuple(cand))
                
        # 3. Remove an element
        if len(current) > 2:
            for i in range(1, len(current)):
                cand = [x for j, x in enumerate(current) if j != i]
                neighbors.add(tuple(cand))
                
        # Evaluate neighbors
        for neighbor in neighbors:
            if time.time() > deadline - 10:
                break
            s = compute_score(list(neighbor))
            if s > best_score:
                best_score = s
                best_set = list(neighbor)
                current = list(neighbor)
                improved = True
                no_improve = 0
                break # Restart search from new best
                
        if not improved:
            no_improve += 1
            # Random perturbation
            if random.random() < 0.3:
                new_el = random.randint(1, max_cur + 50)
                if new_el not in current:
                    cand = sorted(current + [new_el])
                    s = compute_score(cand)
                    if s > best_score - 0.01: # Accept slightly worse to escape local optima
                        best_score = max(best_score, s)
                        best_set = list(cand)
                        current = list(cand)

    return list(best_set)
