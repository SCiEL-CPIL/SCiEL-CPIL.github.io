import time
import random
import numpy as np
from math import log, exp
from itertools import combinations

def score_set(U):
    """
    Computes the score for a set U.
    Score = 1 + log(|U-U| / |U+U|) / log(2*max(U)+1)
    """
    if not U:
        return -1
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
        
    sums = set()
    diffs = set()
    
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def construct():
    deadline = time.time() + 145
    best_score = -1
    best_set = [0, 1, 4, 9, 11]
    
    # Initial best
    s = score_set(best_set)
    if s > best_score:
        best_score = s
        best_set = list(best_set)

    # 1. Try some structured candidates
    # Based on the insight that base-2 and small base sets have high ratios
    structured_candidates = [
        # Binary
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256],
        [0, 1, 2, 4, 8, 16, 32, 64],
        # Ternary
        [0, 1, 3, 9, 27, 81, 243],
        [0, 1, 3, 9, 27, 81],
        # Mixed bases / other structures
        [0, 1, 4, 9, 16, 25, 36, 49],
        [0, 1, 3, 7, 15, 31, 63],
        [0, 1, 4, 10, 20, 35, 56],
        [0, 1, 5, 10, 16, 23, 31],
        [0, 1, 6, 12, 19, 27, 36],
        [0, 1, 3, 6, 10, 15, 21],
        # The seed set
        [0, 1, 4, 9, 11],
        # Variations of seed
        [0, 1, 4, 9, 11, 16],
        [0, 1, 4, 9, 11, 15],
        [0, 1, 4, 9, 11, 12],
        [0, 1, 4, 9, 11, 13],
        [0, 1, 4, 9, 11, 14],
        [0, 1, 4, 9, 11, 17],
    ]
    
    for cand in structured_candidates:
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    def local_search(U, deadline, max_val_limit=2000, max_n_limit=20):
        """
        Perform local search with simulated annealing.
        """
        U = sorted(set(U))
        current_score = score_set(U)
        best_local_score = current_score
        best_local_set = list(U)
        
        # Temperature for simulated annealing
        T = 0.5
        cooling_rate = 0.99
        min_T = 0.001
        
        no_improvement = 0
        max_no_improvement = 200
        
        while time.time() < deadline - 2 and T > min_T:
            if no_improvement > max_no_improvement:
                break
                
            # Generate a perturbation
            move_type = random.random()
            new_U = list(U)
            
            if move_type < 0.25:
                # Add an element
                if len(U) < max_n_limit:
                    # Choose from arithmetic neighborhood or random
                    candidates = set()
                    # Try sums and diffs of existing elements
                    for a in U:
                        for b in U:
                            s = a + b
                            d = abs(a - b)
                            if 0 < s <= max_val_limit and s not in U:
                                candidates.add(s)
                            if 0 < d <= max_val_limit and d not in U:
                                candidates.add(d)
                    # Add some random
                    for _ in range(10):
                        r = random.randint(1, max_val_limit)
                        if r not in U:
                            candidates.add(r)
                    if candidates:
                        new_U.append(random.choice(list(candidates)))
                        new_U.sort()
            elif move_type < 0.45:
                # Remove an element
                if len(U) > 2:
                    removable = [x for x in U if x != 0]
                    if removable:
                        idx = random.choice(removable)
                        new_U = [x for x in U if x != idx]
            elif move_type < 0.85:
                # Replace an element
                if len(U) > 1:
                    idx = random.randint(1, len(U) - 1)
                    old_val = U[idx]
                    # Try nearby values with varying steps
                    steps = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, -10, 10, -20, 20, 50, -50]
                    random.shuffle(steps)
                    for step in steps:
                        new_val = old_val + step
                        if new_val < 1 or new_val in U:
                            continue
                        new_U = U.copy()
                        new_U[idx] = new_val
                        new_U.sort()
                        break
                    else:
                        new_U = U.copy()
            else:
                # Swap two elements
                if len(U) > 2:
                    i, j = random.sample(range(1, len(U)), 2)
                    new_U = U.copy()
                    new_U[i], new_U[j] = new_U[j], new_U[i]
                    new_U.sort()
            
            if new_U == U:
                continue
                
            new_score = score_set(new_U)
            if new_score is None or new_score < -1:
                continue
                
            # Simulated annealing acceptance
            if new_score > current_score:
                U = new_U
                current_score = new_score
                no_improvement = 0
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local_set = list(U)
            else:
                # Accept with probability exp((new_score - current_score) / T)
                delta = new_score - current_score
                if delta > -10:  # Avoid underflow
                    prob = min(1.0, exp(-delta / T) if delta < 0 else 1.0)
                    if random.random() < prob:
                        U = new_U
                        current_score = new_score
                no_improvement += 1
            
            T *= cooling_rate
        
        return best_local_score, best_local_set

    # Restart loop with local search
    restarts = 0
    while time.time() < deadline - 5:
        restarts += 1
        
        # Choose starting set
        if restarts % 5 == 0:
            base = random.choice(structured_candidates)
            U = list(base)
        elif restarts % 5 == 1:
            # Quadratic progression
            n = random.randint(5, 12)
            a = random.randint(1, 3)
            b = random.randint(1, 5)
            U = [0] + [a*i*i + b*i for i in range(1, n) if a*i*i + b*i <= 1000]
            U = sorted(set(U))
        elif restarts % 5 == 2:
            # Geometric progression (binary-like)
            r = random.uniform(1.5, 2.5)
            U = [0]
            val = 1
            while val < 1000 and len(U) < 15:
                U.append(int(val))
                val *= r
            U = sorted(set(U))
        elif restarts % 5 == 3:
            # Random
            n = random.randint(4, 12)
            elements = random.sample(range(1, 1000), n - 1)
            U = [0] + elements
            U.sort()
        else:
            # Perturb best set
            U = list(best_set)
            if len(U) > 2:
                idx = random.randint(1, len(U) - 1)
                U[idx] = max(1, U[idx] + random.randint(-20, 20))
                U = sorted(set(U))
        
        local_score, local_set = local_search(U, deadline)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set

    # Final polish
    if time.time() < deadline - 2:
        U = list(best_set)
        U.sort()
        current_score = best_score
        
        for _ in range(500):
            if time.time() > deadline - 2:
                break
            idx = random.randint(1, len(U) - 1)
            old_val = U[idx]
            found = False
            steps = [-3, -2, -1, 1, 2, 3, -5, 5, -10, 10, -20, 20]
            random.shuffle(steps)
            for step in steps:
                new_val = old_val + step
                if new_val < 1 or new_val in U:
                    continue
                new_U = U.copy()
                new_U[idx] = new_val
                new_U.sort()
                new_score = score_set(new_U)
                if new_score > current_score:
                    U = new_U
                    current_score = new_score
                    found = True
                    break
            if found:
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(U)

    return best_set
