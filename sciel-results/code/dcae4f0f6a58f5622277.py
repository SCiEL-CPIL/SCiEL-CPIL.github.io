import time
from math import log
import random

def compute_score(U):
    """Compute the score for a given set U."""
    U = sorted(set(U))
    if 0 not in U:
        U.append(0)
        U = sorted(U)
    n = len(U)
    if n == 0 or U[0] != 0:
        return 0.0
    
    # Compute sumset U+U
    sumset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
    
    # Compute difference set U-U
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 1:
        return 1.0
    
    denominator = log(2 * max_u + 1)
    if denominator <= 0:
        return 0.0
    
    return 1.0 + log(ratio) / denominator

def local_search(U, time_limit, seed=42):
    """Perform local search to improve the set U."""
    random.seed(seed)
    start_time = time.time()
    best_set = sorted(U)
    best_score = compute_score(best_set)
    current_set = sorted(U)
    current_score = best_score
    
    # Parameters
    step_size_max = 20
    T = 1.0 # Temperature for simulated annealing
    cooling_rate = 0.99
    
    while time.time() - start_time < time_limit:
        # Generate a move
        if len(current_set) > 1:
            i = random.randint(1, len(current_set) - 1)
            delta = random.randint(-step_size_max, step_size_max)
            if delta == 0:
                continue
            new_val = current_set[i] + delta
            
            # Check constraints
            if new_val < 0:
                continue
            # Check uniqueness and order
            valid = True
            if i > 0 and new_val <= current_set[i-1]:
                valid = False
            if i < len(current_set) - 1 and new_val >= current_set[i+1]:
                valid = False
            if valid and new_val in current_set:
                valid = False
            
            if not valid:
                continue
                
            # Create new set
            new_set = current_set.copy()
            new_set[i] = new_val
            new_score = compute_score(new_set)
            
            # Acceptance criterion
            if new_score > current_score:
                current_set = new_set
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = new_set.copy()
            else:
                # Simulated annealing acceptance
                delta_score = new_score - current_score
                if delta_score < 0:
                    prob = pow(2.71828, delta_score / T)
                    if random.random() < prob:
                        current_set = new_set
                        current_score = new_score
        
        # Cool down
        T *= cooling_rate
        if T < 0.01:
            break
            
    return best_set, best_score

def construct():
    start_time = time.time()
    time_limit = 115
    
    best_set = [0, 1, 3, 7, 15, 31, 63, 127]
    best_score = compute_score(best_set)
    
    # Initial candidates
    candidates = []
    
    # Pattern 1: 2^k - 1
    for k in range(4, 10):
        U = [2**i - 1 for i in range(1, k)]
        U.insert(0, 0)
        candidates.append(U)
        
    # Pattern 2: Powers of 2
    for k in range(4, 10):
        U = [2**i for i in range(k)]
        U.insert(0, 0)
        candidates.append(U)
        
    # Pattern 3: 3^k - 1
    for k in range(3, 8):
        U = [3**i - 1 for i in range(1, k)]
        U.insert(0, 0)
        candidates.append(U)
        
    # Pattern 4: Random sparse
    random.seed(123)
    for _ in range(5):
        n = random.randint(5, 10)
        max_val = random.randint(50, 200)
        U = [0]
        remaining = max_val
        for i in range(n - 1):
            max_pick = remaining // (n - 1 - i) + 1
            if max_pick < 1: max_pick = 1
            pick = random.randint(1, max_pick)
            U.append(U[-1] + pick)
            remaining -= pick
        candidates.append(sorted(set(U)))
        
    # Run local search on each candidate
    for i, U in enumerate(candidates):
        if time.time() - start_time > time_limit:
            break
        # Allocate remaining time / remaining candidates
        # Simple heuristic: give each candidate a fixed chunk or just run until time is up
        # Better: run local search with a small time limit per candidate
        time_per_candidate = max(1, (time_limit - (time.time() - start_time)) / (len(candidates) - i))
        res_set, res_score = local_search(U, time_per_candidate, seed=i+1)
        if res_score > best_score:
            best_score = res_score
            best_set = res_set
            
    return best_set
