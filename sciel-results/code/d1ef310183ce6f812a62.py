import time
from math import log
import random
import numpy as np

def score_set(U):
    """Compute the score for a set U using efficient bitset operations."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    max_u = max(U)
    
    # Compute sumset using bitset convolution
    sum_mask = 1 << 0
    for x in U:
        sum_mask |= (sum_mask << x)
    
    # Count bits in sum_mask
    sum_count = bin(sum_mask).count('1')
    
    # Compute difference set
    diff_mask = 1 << max_u  # represent 0 as bit at position max_u
    for i in range(n):
        for j in range(i + 1, n):
            d = U[j] - U[i]
            diff_mask |= (1 << (max_u + d))
            diff_mask |= (1 << (max_u - d))
    
    diff_count = bin(diff_mask).count('1')
    
    if sum_count == 0 or diff_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def geometric_set(b, k, offset=0):
    """Generate a geometric set."""
    U = [0]
    val = 1
    for i in range(k):
        U.append(val + offset)
        val *= b
    return sorted(set(U))

def generate_candidates():
    """Generate diverse candidate sets."""
    candidates = []
    
    # Pure geometric
    for b in range(2, 10):
        for k in range(3, 15):
            candidates.append(geometric_set(b, k))
            
    # Modified geometric: 0, 1, b, b^2...
    for b in range(2, 10):
        for k in range(3, 15):
            U = [0, 1]
            val = b
            for i in range(k-2):
                U.append(val)
                val *= b
            candidates.append(sorted(set(U)))
            
    # Variable base
    for _ in range(20):
        b = random.uniform(1.5, 5.0)
        k = random.randint(4, 12)
        U = [0]
        val = 1
        for i in range(k):
            U.append(int(val))
            val *= b
        candidates.append(sorted(set(U)))
        
    return candidates

def local_search_dynamic(U, deadline):
    """Local search with dynamic time allocation."""
    U = sorted(set(U))
    current_score = score_set(U)
    best_score = current_score
    best_set = list(U)
    
    start_time = time.time()
    
    while time.time() < deadline:
        n = len(U)
        move_type = random.random()
        
        if move_type < 0.3:
            # Replace an element
            if n < 2: continue
            idx = random.randint(1, n - 1)
            old_val = U[idx]
            # Perturb
            factor = random.uniform(0.8, 1.2)
            new_val = int(old_val * factor)
            if new_val <= 0: new_val = 1
            new_set = sorted(set([U[i] if i != idx else new_val for i in range(n)]))
            if len(new_set) != n: continue
            new_U = new_set
        elif move_type < 0.6:
            # Add an element
            if n > 12: continue
            max_val = max(U) * 2
            new_val = random.randint(1, max_val)
            if new_val in U: continue
            new_U = sorted(U + [new_val])
        else:
            # Remove an element
            if n <= 3: continue
            idx = random.randint(1, n - 1)
            new_U = [U[i] for i in range(n) if i != idx]
            
        new_score = score_set(new_U)
        if new_score > current_score:
            current_score = new_score
            U = new_U
            if current_score > best_score:
                best_score = current_score
                best_set = list(U)
        # Simple simulated annealing accept
        elif random.random() < np.exp((new_score - current_score) / 0.1):
            U = new_U
            current_score = new_score
            
    return best_score, best_set

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    candidates = generate_candidates()
    
    best_score = -1
    best_set = None
    
    for cand in candidates:
        if time.time() > deadline - 10:
            break
        s, _ = local_search_dynamic(cand, deadline - 5)
        if s > best_score:
            best_score = s
            best_set = _
            
    # Final polish
    if best_set and time.time() < deadline - 2:
        s, set_final = local_search_dynamic(best_set, deadline - 1)
        if s > best_score:
            best_score = s
            best_set = set_final
            
    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
