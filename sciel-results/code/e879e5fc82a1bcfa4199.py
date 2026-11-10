import time
import random
import math

def calc_score(A):
    """
    Calculate the max of the size of the sumset and product set of A.
    A is a list of distinct positive integers.
    """
    sumset = set()
    prodset = set()
    for i in range(len(A)):
        ai = A[i]
        for j in range(len(A)):
            aj = A[j]
            sumset.add(ai + aj)
            prodset.add(ai * aj)
    return max(len(sumset), len(prodset))

def generate_smooth_numbers(limit, primes, max_count=50000):
    """Generate smooth numbers up to limit using given primes."""
    smooth = [1]
    i = 0
    while i < len(smooth) and len(smooth) < max_count:
        for p in primes:
            val = smooth[i] * p
            if val <= limit:
                smooth.append(val)
            else:
                break
        i += 1
    return sorted(set(smooth))

def local_search_best_improve(A, time_limit, rng, candidate_pool):
    """
    Local search using Best Improvement strategy for single swaps.
    Falls back to random pair swaps if stuck.
    """
    A = list(A)
    current_score = calc_score(A)
    best_score = current_score
    best_A = list(A)
    start_time = time.time()
    n = len(A)
    
    # Precompute current sumset and prodset
    sumset = set(a + b for a in A for b in A)
    prodset = set(a * b for a in A for b in A)
    
    # Set of current values for quick check
    current_vals = set(A)
    
    improved = True
    no_improve_count = 0
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        no_improve_count += 1
        
        # Shuffle indices to avoid bias
        indices = list(range(n))
        rng.shuffle(indices)
        
        for i in indices:
            if (time.time() - start_time) > time_limit:
                return best_A, best_score
            
            old_val = A[i]
            
            best_new_val = None
            best_est_score = current_score
            best_new_sumset = None
            best_new_prodset = None
            best_new_A = None
            
            # Iterate through candidate pool
            # To speed up, we can limit candidates to those 'close' to old_val 
            # or just iterate all. Pool is small enough.
            for new_val in candidate_pool:
                if new_val == old_val or new_val in current_vals:
                    continue
                
                if new_val < 1 or new_val > 10000:
                    continue
                
                # Compute delta
                new_sumset_add = set()
                new_prodset_add = set()
                old_sumset_remove = set()
                old_prodset_remove = set()
                
                for other in A:
                    if other is not A[i]:
                        new_sumset_add.add(new_val + other)
                        new_prodset_add.add(new_val * other)
                        old_sumset_remove.add(old_val + other)
                        old_prodset_remove.add(old_val * other)
                
                new_sumset_add.add(2 * new_val)
                new_prodset_add.add(new_val * new_val)
                old_sumset_remove.add(2 * old_val)
                old_prodset_remove.add(old_val * old_val)
                
                # Estimate
                # Note: new_sumset_add might contain values already in sumset.
                # old_sumset_remove might contain values still in sumset via other pairs.
                
                # Efficient estimation:
                # new_size = len(sumset) + len(new_add - sumset) - len(old_remove & (sumset - new_add))
                # This is exact for the change in set size if we assume no other interactions, 
                # which is true for single element swap.
                
                est_sumset_size = len(sumset) + len(new_sumset_add - sumset) - len(old_sumset_remove & (sumset - new_sumset_add))
                est_prodset_size = len(prodset) + len(new_prodset_add - prodset) - len(old_prodset_remove & (prodset - new_prodset_add))
                
                est_score = max(est_sumset_size, est_prodset_size)
                
                if est_score < best_est_score:
                    best_est_score = est_score
                    best_new_val = new_val
                    # We don't compute full sets yet to save time, only if we apply
                    
            if best_new_val is not None and best_est_score < current_score:
                # Verify with exact score
                new_A = A.copy()
                new_A[i] = best_new_val
                score = calc_score(new_A)
                
                if score < current_score:
                    current_score = score
                    A = new_A
                    improved = True
                    no_improve_count = 0
                    current_vals = set(A)
                    
                    # Update sumset and prodset
                    sumset = set(a + b for a in A for b in A)
                    prodset = set(a * b for a in A for b in A)
                    
                    if score < best_score:
                        best_score = score
                        best_A = list(A)
            
            # If we made an improvement, break out of index loop? 
            # Best improvement usually restarts the scan or continues.
            # Restarting is safer to find global best single swap.
            # But continuing allows multiple swaps in one pass.
            # Let's continue, but the sets are updated.
            
        # If no single swap improved, try pair swaps
        if not improved and (time.time() - start_time) < time_limit:
            # Try a few random pair swaps
            for _ in range(10):
                if (time.time() - start_time) > time_limit:
                    break
                i, j = rng.sample(range(n), 2)
                old_i, old_j = A[i], A[j]
                
                new_i = rng.choice(candidate_pool)
                new_j = rng.choice(candidate_pool)
                
                if new_i == old_i or new_j == old_j or new_i == new_j:
                    continue
                if new_i in current_vals or new_j in current_vals:
                    continue
                if not (1 <= new_i <= 10000 and 1 <= new_j <= 10000):
                    continue
                
                new_A = A.copy()
                new_A[i] = new_i
                new_A[j] = new_j
                
                score = calc_score(new_A)
                if score < current_score:
                    current_score = score
                    A = new_A
                    improved = True
                    no_improve_count = 0
                    current_vals = set(A)
                    sumset = set(a + b for a in A for b in A)
                    prodset = set(a * b for a in A for b in A)
                    if score < best_score:
                        best_score = score
                        best_A = list(A)
                    break # Restart single swap scan
    
    return best_A, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    rng = random.Random(42)
    
    # Generate a rich pool of smooth numbers
    primes = [2, 3, 5, 7, 11]
    smooth_numbers = generate_smooth_numbers(10000, primes, max_count=50000)
    
    # Add small integers for diversity
    candidate_pool = list(range(1, 51)) + smooth_numbers
    candidate_pool = sorted(set(candidate_pool))
    
    best_A = None
    best_score = float('inf')
    
    # Multiple initial candidates
    candidates = []
    
    # 1. Smallest smooth numbers
    cand1 = smooth_numbers[:20]
    candidates.append(cand1)
    
    # 2. Powers of 2 and 3
    cand2 = []
    for e2 in range(15):
        for e3 in range(10):
            val = (2 ** e2) * (3 ** e3)
            if val <= 10000 and val not in cand2:
                cand2.append(val)
            if len(cand2) >= 20:
                break
        if len(cand2) >= 20:
            break
    cand2 = sorted(cand2[:20])
    candidates.append(cand2)
    
    # 3. Include 1 and next smooth
    cand3 = [1] + smooth_numbers[1:19]
    candidates.append(cand3)
    
    # 4. Random selection
    cand4 = sorted(rng.sample(smooth_numbers[:200], 20))
    candidates.append(cand4)
    
    # 5. Dense small
    cand5 = list(range(1, 21))
    candidates.append(cand5)
    
    # 6. Geometric-like
    cand6 = []
    val = 1
    while val <= 10000 and len(cand6) < 20:
        cand6.append(val)
        if val % 2 == 0:
            val = val * 3
        elif val % 3 == 0:
            val = val * 2
        else:
            val = val * 2
    candidates.append(cand6)
    
    # 7. Highly composite
    cand7 = sorted([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36])
    candidates.append(cand7)
    
    # 8. Log-spaced
    cand8 = []
    for target_log in [i * 0.3 for i in range(20)]:
        target = int(round(2 ** target_log))
        best_smooth = min(smooth_numbers, key=lambda x: abs(x - target))
        if best_smooth not in cand8:
            cand8.append(best_smooth)
    cand8 = sorted(cand8[:20])
    candidates.append(cand8)
    
    # Validate
    valid_candidates = []
    for c in candidates:
        if len(set(c)) == 20 and all(1 <= x <= 10000 for x in c):
            valid_candidates.append(c)
    
    # Evaluate initial
    initial_scores = []
    for c in valid_candidates:
        s = calc_score(c)
        initial_scores.append((s, c))
    
    initial_scores.sort(key=lambda x: x[0])
    
    # Run local search
    time_per_candidate = min(15.0, (deadline - time.time()) * 0.9 / max(1, len(valid_candidates)))
    
    for score, cand in initial_scores:
        if time.time() > deadline - 5.0:
            break
        
        if score < best_score:
            best_score = score
            best_A = list(cand)
        
        remaining = deadline - time.time()
        if remaining < 10.0:
            break
        
        alloc_time = min(time_per_candidate, remaining * 0.8)
        
        improved_A, ls_score = local_search_best_improve(cand, alloc_time, rng, candidate_pool)
        
        if ls_score < best_score:
            best_score = ls_score
            best_A = improved_A
    
    # Final random restarts if time permits
    if time.time() < deadline - 10.0:
        for attempt in range(5):
            if time.time() > deadline - 5.0:
                break
            
            if best_A is not None:
                start_A = best_A.copy()
                # Perturb
                for _ in range(2):
                    idx = rng.randint(0, 19)
                    new_val = rng.choice(candidate_pool)
                    while new_val in start_A:
                        new_val = rng.choice(candidate_pool)
                    start_A[idx] = new_val
            else:
                start_A = list(range(1, 21))
            
            if len(set(start_A)) != 20:
                continue
            
            s = calc_score(start_A)
            if s < best_score:
                best_score = s
                best_A = start_A
            
            remaining = deadline - time.time()
            if remaining < 5.0:
                break
            
            improved_A, ls_score = local_search_best_improve(start_A, min(10.0, remaining * 0.8), rng, candidate_pool)
            if ls_score < best_score:
                best_score = ls_score
                best_A = improved_A
    
    if best_A is None:
        best_A = list(range(1, 21))
    
    return best_A
