import time
import random
import math
import numpy as np

def _score(A):
    """Compute the objective value: max(|A+A|, |A*A|)"""
    A_arr = np.array(A)
    n = len(A_arr)
    # Vectorized sum and product computation
    sums = np.add.outer(A_arr, A_arr)
    prods = np.multiply.outer(A_arr, A_arr)
    n_sums = len(np.unique(sums))
    n_prods = len(np.unique(prods))
    return max(n_sums, n_prods)

def _sa_search(A, start_temp=10.0, end_temp=0.1, steps=50000, seed=None, deadline=None):
    """Simulated Annealing search with deadline guard."""
    if seed is not None:
        random.seed(seed)
    
    current_A = A[:]
    current_score = _score(current_A)
    best_A = current_A[:]
    best_score = current_score
    
    n = len(current_A)
    
    for step in range(steps):
        if deadline is not None and time.time() > deadline:
            break
            
        # Cooling schedule
        T = start_temp * (end_temp / start_temp) ** (step / steps)
        
        # Generate neighbor
        k = random.randint(0, n - 1)
        old_val = current_A[k]
        
        # Choose new value - mix of local and global moves
        if random.random() < 0.7:
            # Local perturbation - smaller steps
            delta = random.randint(-20, 20)
            new_val = old_val + delta
        elif random.random() < 0.5:
            # Medium jump
            new_val = random.randint(1, max(1, old_val // 2)) if old_val > 10 else random.randint(1, 100)
        else:
            # Global jump
            new_val = random.randint(1, 10000)
            
        # Ensure bounds and distinctness
        if new_val < 1 or new_val > 10000:
            continue
            
        # Check distinctness efficiently
        if new_val in current_A and new_val != old_val:
            continue
            
        # Apply move
        current_A[k] = new_val
        new_score = _score(current_A)
        
        # Acceptance probability
        delta_score = new_score - current_score
        if delta_score < 0 or random.random() < math.exp(-delta_score / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_A = current_A[:]
        else:
            # Reject move
            current_A[k] = old_val
            
    return best_A, best_score

def _local_search(A, max_iter=5000, deadline=None):
    """Local search: try single-element changes and accept improvements."""
    current_A = A[:]
    current_score = _score(current_A)
    best_A = current_A[:]
    best_score = current_score
    n = len(current_A)
    
    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for k in range(n):
            old_val = current_A[k]
            # Try a few random alternatives
            for _ in range(5):
                if random.random() < 0.5:
                    new_val = old_val + random.randint(-10, 10)
                else:
                    new_val = random.randint(1, 10000)
                
                if new_val < 1 or new_val > 10000:
                    continue
                if new_val in current_A and new_val != old_val:
                    continue
                    
                current_A[k] = new_val
                new_score = _score(current_A)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    break
                else:
                    current_A[k] = old_val
            
            if improved:
                if current_score < best_score:
                    best_score = current_score
                    best_A = current_A[:]
                break
        
        if not improved:
            break
            
    return best_A, best_score

def construct():
    n = 20
    max_val = 10000
    
    # Deadline guard - 90s limit, use 85s for safety
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_A = None
    best_score = float('inf')
    
    # Better initial candidates
    candidates = []
    
    # 1. Small consecutive integers (minimizes sumset)
    candidates.append(list(range(1, n + 1)))
    
    # 2. Geometric progression (minimizes product set diversity)
    gp = [2 ** i for i in range(n)]
    if gp[-1] <= max_val:
        candidates.append(gp)
    
    # 3. Log-spaced values
    log_vals = [int(max_val ** (i / (n - 1))) + 1 for i in range(n)]
    # Ensure distinct
    seen = set()
    log_distinct = []
    for v in log_vals:
        if v not in seen and v <= max_val:
            seen.add(v)
            log_distinct.append(v)
    while len(log_distinct) < n:
        v = random.randint(1, max_val)
        if v not in seen:
            seen.add(v)
            log_distinct.append(v)
    candidates.append(log_distinct)
    
    # 4. Powers of small base
    for base in [3, 4, 5]:
        pwr = [base ** i for i in range(n)]
        if pwr[-1] <= max_val:
            candidates.append(pwr)
    
    # 5. Random samples from different ranges
    for _ in range(5):
        r = random.sample(range(1, max_val + 1), n)
        candidates.append(r)
    
    # 6. Mixed: small numbers + some larger
    mixed = list(range(1, 15)) + random.sample(range(100, 1000), 5)
    if len(set(mixed)) == n:
        candidates.append(mixed)
    
    # Validate candidates
    valid_candidates = []
    for c in candidates:
        if len(c) == n and len(set(c)) == n and all(1 <= x <= max_val for x in c):
            valid_candidates.append(c[:])
    
    if not valid_candidates:
        valid_candidates = [list(range(1, n + 1))]
    
    # Evaluate initial candidates
    for c in valid_candidates:
        s = _score(c)
        if s < best_score:
            best_score = s
            best_A = c[:]
    
    # Run local search on best initial candidate
    if best_A is not None:
        A_ls, score_ls = _local_search(best_A, max_iter=3000, deadline=deadline)
        if score_ls < best_score:
            best_score = score_ls
            best_A = A_ls
    
    # Run SA from multiple starting points
    sa_starts = [best_A] if best_A else []
    for c in valid_candidates[:3]:
        if c != best_A:
            sa_starts.append(c)
    
    for i, init_A in enumerate(sa_starts):
        if time.time() > deadline:
            break
            
        # Different temperature schedules for diversity
        if i == 0:
            A_res, score_res = _sa_search(init_A, start_temp=5.0, end_temp=0.01, steps=30000, seed=i*100, deadline=deadline)
        else:
            A_res, score_res = _sa_search(init_A, start_temp=8.0, end_temp=0.05, steps=20000, seed=i*100, deadline=deadline)
        
        if score_res < best_score:
            best_score = score_res
            best_A = A_res[:]
    
    # Final polish: local search on best solution
    if best_A is not None and time.time() < deadline - 2:
        A_polish, score_polish = _local_search(best_A, max_iter=5000, deadline=deadline)
        if score_polish < best_score:
            best_score = score_polish
            best_A = A_polish
    
    if best_A is None:
        best_A = list(range(1, n + 1))
        
    return best_A
