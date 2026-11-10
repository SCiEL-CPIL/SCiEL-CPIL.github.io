import time
import random
from itertools import combinations

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

def calc_score_fast(A):
    """Faster score calculation using precomputed pairs."""
    n = len(A)
    sumset = set()
    prodset = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            s = ai + aj
            p = ai * aj
            sumset.add(s)
            prodset.add(p)
            if i != j:
                sumset.add(s)  # symmetric, same value
                prodset.add(p)
    return max(len(sumset), len(prodset))

def generate_smooth_numbers(limit, primes, max_exp=12):
    """Generate smooth numbers up to limit using given primes."""
    smooth = set([1])
    i = 0
    while i < len(smooth) and len(smooth) < 200000:
        for p in primes:
            val = list(smooth)[i] * p
            if val <= limit:
                smooth.add(val)
            else:
                break
        i += 1
    return sorted(smooth)

def local_search_fast(A, time_limit, rng, candidate_pool):
    """
    Fast local search with precomputed candidate pool.
    Uses single-element replacement and pair swaps.
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
    
    improved = True
    no_improve_count = 0
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        no_improve_count += 1
        
        # Single-element replacement
        shuffled_pool = candidate_pool.copy()
        rng.shuffle(shuffled_pool)
        
        for i in range(n):
            if (time.time() - start_time) > time_limit:
                return best_A, best_score
            
            old_val = A[i]
            
            for new_val in shuffled_pool:
                if new_val == old_val or new_val in A:
                    continue
                
                if new_val < 1 or new_val > 10000:
                    continue
                
                # Compute delta
                new_sumset_add = set()
                new_prodset_add = set()
                old_sumset_remove = set()
                old_prodset_remove = set()
                
                for other in A:
                    if other != old_val:
                        new_sumset_add.add(new_val + other)
                        new_prodset_add.add(new_val * other)
                        old_sumset_remove.add(old_val + other)
                        old_prodset_remove.add(old_val * other)
                
                new_sumset_add.add(2 * new_val)
                new_prodset_add.add(new_val * new_val)
                old_sumset_remove.add(2 * old_val)
                old_prodset_remove.add(old_val * old_val)
                
                est_sumset_size = len(sumset) + len(new_sumset_add - sumset) - len(old_sumset_remove & (sumset - new_sumset_add))
                est_prodset_size = len(prodset) + len(new_prodset_add - prodset) - len(old_prodset_remove & (prodset - new_prodset_add))
                
                est_score = max(est_sumset_size, est_prodset_size)
                
                if est_score < current_score:
                    new_A = A.copy()
                    new_A[i] = new_val
                    score = calc_score(new_A)
                    
                    if score < current_score:
                        current_score = score
                        A = new_A
                        improved = True
                        no_improve_count = 0
                        
                        sumset = set(a + b for a in A for b in A)
                        prodset = set(a * b for a in A for b in A)
                        
                        if score < best_score:
                            best_score = score
                            best_A = list(A)
                        
                        break
        
        # Pair swaps (only if time permits)
        if (time.time() - start_time) < time_limit * 0.7:
            for _ in range(5):
                if (time.time() - start_time) > time_limit:
                    break
                i, j = rng.sample(range(n), 2)
                new_val_i = rng.choice(candidate_pool)
                new_val_j = rng.choice(candidate_pool)
                
                if new_val_i in A or new_val_j in A or new_val_i == new_val_j:
                    continue
                if new_val_i == A[i] or new_val_j == A[j]:
                    continue
                
                new_A = A.copy()
                new_A[i] = new_val_i
                new_A[j] = new_val_j
                
                score = calc_score(new_A)
                if score < current_score:
                    current_score = score
                    A = new_A
                    improved = True
                    no_improve_count = 0
                    sumset = set(a + b for a in A for b in A)
                    prodset = set(a * b for a in A for b in A)
                    if score < best_score:
                        best_score = score
                        best_A = list(A)
        
        if no_improve_count >= 3:
            break
    
    return best_A, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # 8s safety margin
    
    rng = random.Random(42)
    
    # Generate a rich pool of smooth numbers with more primes
    primes = [2, 3, 5, 7, 11]
    smooth_numbers = generate_smooth_numbers(10000, primes)
    
    # Also add small integers and highly composite numbers for diversity
    highly_composite = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36,
                        40, 42, 45, 48, 50, 54, 56, 60, 63, 64, 70, 72, 75, 80, 81, 84, 90, 96, 100]
    candidate_pool = list(range(1, 61)) + smooth_numbers + highly_composite
    candidate_pool = sorted(set(x for x in candidate_pool if 1 <= x <= 10000))
    
    best_A = None
    best_score = float('inf')
    
    # Multiple initial candidates with different structures
    
    # Candidate 1: Smallest smooth numbers
    cand1 = smooth_numbers[:20]
    
    # Candidate 2: Powers of 2 and 3 mixed
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
    
    # Candidate 3: Include 1 and some smooth numbers
    cand3 = [1] + smooth_numbers[1:19]
    
    # Candidate 4: Random selection from smooth numbers
    cand4 = sorted(rng.sample(smooth_numbers[:150], 20))
    
    # Candidate 5: Dense small numbers
    cand5 = list(range(1, 21))
    
    # Candidate 6: Geometric-like with smooth numbers
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
    
    # Candidate 7: Optimized - use numbers that are highly composite
    cand7 = sorted([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36])
    
    # Candidate 8: Mix of small and medium smooth numbers
    cand8 = sorted(rng.sample(smooth_numbers[:50], 10) + rng.sample(smooth_numbers[50:200], 10))
    
    candidates = [cand1, cand2, cand3, cand4, cand5, cand6, cand7, cand8]
    
    # Remove invalid candidates
    valid_candidates = []
    for c in candidates:
        if len(set(c)) == 20 and all(1 <= x <= 10000 for x in c):
            valid_candidates.append(c)
    
    # Evaluate initial scores
    initial_scores = []
    for c in valid_candidates:
        s = calc_score(c)
        initial_scores.append((s, c))
    
    initial_scores.sort(key=lambda x: x[0])
    
    # Run local search on the best candidates
    num_candidates = len(valid_candidates)
    time_per_candidate = min(10.0, (deadline - time.time()) * 0.85 / max(1, num_candidates))
    
    for score, cand in initial_scores:
        if time.time() > deadline - 8.0:
            break
        
        if score < best_score:
            best_score = score
            best_A = list(cand)
        
        remaining = deadline - time.time()
        if remaining < 8.0:
            break
        
        alloc_time = min(time_per_candidate, remaining * 0.6)
        
        improved_A, ls_score = local_search_fast(cand, alloc_time, rng, candidate_pool)
        
        if ls_score < best_score:
            best_score = ls_score
            best_A = improved_A
    
    # Final polish: intensive local search on the best solution
    if best_A is not None and time.time() < deadline - 10.0:
        remaining = deadline - time.time() - 5.0
        if remaining > 3.0:
            polished_A, polish_score = local_search_fast(best_A, remaining, rng, candidate_pool)
            if polish_score < best_score:
                best_score = polish_score
                best_A = polished_A
    
    # Random restarts if time permits
    if best_A is not None and time.time() < deadline - 15.0:
        for attempt in range(2):
            if time.time() > deadline - 8.0:
                break
            
            start_A = best_A.copy()
            for _ in range(rng.randint(1, 2)):
                idx = rng.randint(0, 19)
                new_val = rng.choice(candidate_pool)
                attempts = 0
                while new_val in start_A and attempts < 50:
                    new_val = rng.choice(candidate_pool)
                    attempts += 1
                if new_val not in start_A:
                    start_A[idx] = new_val
            
            if len(set(start_A)) != 20:
                continue
            
            s = calc_score(start_A)
            if s < best_score:
                best_score = s
                best_A = start_A
            
            remaining = deadline - time.time()
            if remaining < 5.0:
                break
            
            improved_A, ls_score = local_search_fast(start_A, min(10.0, remaining * 0.6), rng, candidate_pool)
            if ls_score < best_score:
                best_score = ls_score
                best_A = improved_A
    
    if best_A is None:
        best_A = list(range(1, 21))
    
    return best_A
