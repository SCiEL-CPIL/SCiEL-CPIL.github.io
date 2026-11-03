import time
import random
import numpy as np
from math import log

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
    deadline = time.time() + 110
    best_score = -1
    best_set = [0, 1, 4, 9, 11]
    
    # Initial best
    s = score_set(best_set)
    if s > best_score:
        best_score = s
        best_set = list(best_set)

    # 1. Try some structured candidates
    structured_candidates = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81],
        [0, 1, 2, 4, 8, 16, 32, 64, 128],
        [0, 1, 3, 7, 15, 31, 63, 127],
        [0, 1, 4, 10, 20, 35, 56, 84],
        [0, 1, 3, 4, 9, 10, 12, 13],
        [0, 1, 4, 10, 18, 28, 40, 54],
        # Add more structured candidates based on lattice packing ideas
        [0, 1, 5, 10, 16, 23, 31, 40],  # Quadratic
        [0, 1, 6, 12, 19, 27, 36, 46],  # Quadratic
        [0, 1, 3, 6, 10, 15, 21, 28],   # Triangular
    ]
    
    for cand in structured_candidates:
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # 2. Lattice Packing Inspired Search
    # Map elements to 2D points and optimize for minimal sumset bounding box
    # while maximizing difference set sparsity
    
    def lattice_score(U):
        """
        Score based on lattice packing principles.
        We map U to points (i, U[i]) in 2D and analyze the geometry.
        """
        if len(U) < 2:
            return -1
        n = len(U)
        
        # Compute sumset and diffset sizes
        sums = set()
        diffs = set()
        for i in range(n):
            for j in range(i, n):
                sums.add(U[i] + U[j])
            for j in range(i + 1, n):
                d = U[j] - U[i]
                diffs.add(d)
                diffs.add(-d)
        diffs.add(0)
        
        sum_count = len(sums)
        diff_count = len(diffs)
        
        if sum_count == 0:
            return -1
            
        ratio = diff_count / sum_count
        max_u = max(U)
        D = 2 * max_u + 1
        if D <= 1:
            return -1
            
        # Lattice packing score: reward for tight sumset packing and sparse diffset
        # The bounding box of sumset is [0, 2*max_u], size 2*max_u + 1
        # The bounding box of diffset is [-max_u, max_u], size 2*max_u + 1
        # We want sumset to be dense (small ratio of unique to possible)
        # and diffset to be sparse (large ratio of unique to possible)
        
        sum_density = sum_count / (2 * max_u + 1)
        diff_density = diff_count / (2 * max_u + 1)
        
        # Score: high diff_density, low sum_density
        lattice_score = diff_density - sum_density + log(ratio) / log(D)
        
        return lattice_score

    def lattice_search(U, deadline, max_val_limit=500, max_n_limit=30):
        """
        Perform lattice packing inspired search.
        """
        U = sorted(set(U))
        current_score = score_set(U)
        best_local_score = current_score
        best_local_set = list(U)
        
        # Temperature for simulated annealing
        T = 1.0
        cooling_rate = 0.995
        min_T = 0.01
        
        no_improvement = 0
        max_no_improvement = 100
        
        while time.time() < deadline - 2 and T > min_T:
            if no_improvement > max_no_improvement:
                break
                
            # Generate a perturbation
            move_type = random.random()
            new_U = list(U)
            
            if move_type < 0.3:
                # Add an element
                if len(U) < max_n_limit:
                    # Choose from arithmetic neighborhood
                    candidates = set()
                    for a in U:
                        for b in U:
                            s = a + b
                            d = abs(a - b)
                            if 0 < s <= max_val_limit and s not in U:
                                candidates.add(s)
                            if 0 < d <= max_val_limit and d not in U:
                                candidates.add(d)
                    # Add some random
                    for _ in range(20):
                        r = random.randint(1, max_val_limit)
                        if r not in U:
                            candidates.add(r)
                    if candidates:
                        new_U.append(random.choice(list(candidates)))
                        new_U.sort()
            elif move_type < 0.5:
                # Remove an element
                if len(U) > 2:
                    removable = [x for x in U if x != 0]
                    if removable:
                        idx = random.choice(removable)
                        new_U = [x for x in U if x != idx]
            elif move_type < 0.8:
                # Replace an element
                if len(U) > 1:
                    idx = random.randint(1, len(U) - 1)
                    old_val = U[idx]
                    # Try nearby values
                    for step in [-3, -2, -1, 1, 2, 3, 5, 10, -5, -10]:
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
                if delta > -T:  # Avoid underflow
                    prob = min(1.0, exp(-delta / T) if delta < 0 else 1.0)
                    if random.random() < prob:
                        U = new_U
                        current_score = new_score
                no_improvement += 1
            
            T *= cooling_rate
        
        return best_local_score, best_local_set

    # Restart loop with lattice search
    restarts = 0
    while time.time() < deadline - 5:
        restarts += 1
        
        # Choose starting set
        if restarts % 4 == 0:
            base = random.choice(structured_candidates)
            U = list(base)
        elif restarts % 4 == 1:
            # Quadratic progression
            n = random.randint(5, 15)
            a = random.randint(1, 3)
            b = random.randint(1, 5)
            U = [0] + [a*i*i + b*i for i in range(1, n) if a*i*i + b*i <= 500]
            U = sorted(set(U))
        elif restarts % 4 == 2:
            # Geometric progression
            r = random.uniform(1.5, 3.0)
            U = [0]
            val = 1
            while val < 500 and len(U) < 20:
                U.append(int(val))
                val *= r
            U = sorted(set(U))
        else:
            # Random
            n = random.randint(4, 15)
            elements = random.sample(range(1, 500), n - 1)
            U = [0] + elements
            U.sort()
        
        local_score, local_set = lattice_search(U, deadline)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set

    # Final polish
    if time.time() < deadline - 2:
        U = list(best_set)
        U.sort()
        current_score = best_score
        
        for _ in range(200):
            if time.time() > deadline - 2:
                break
            idx = random.randint(1, len(U) - 1)
            old_val = U[idx]
            found = False
            for step in [-3, -2, -1, 1, 2, 3, 5, 10, -5, -10, -20, 20]:
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

# Helper for simulated annealing
def exp(x):
    import math
    return math.exp(x)
