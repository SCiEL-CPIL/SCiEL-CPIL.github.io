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
        
    # Use numpy for vectorized sum and diff calculation
    U_arr = np.array(U, dtype=np.int64)
    
    # Sums: U[i] + U[j] for all i <= j
    # We can compute all pairwise sums and then filter/unique
    # Since we need unique counts, set() on numpy array is fine for moderate n
    # But for speed, let's use set comprehension on numpy results? 
    # Actually, simple python sets are fast for n < 100.
    # Let's stick to python sets but optimized loops? 
    # The current loop is O(n^2). 
    # For n=30, 900 operations. Very fast.
    # The bottleneck is likely the number of evaluations, not the score computation itself.
    # So we keep the fast python implementation.
    
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
    deadline = time.time() + 110  # Safety margin for 120s limit
    best_score = -1
    best_set = [0, 1, 4, 9, 11]
    
    # Initial best
    s = score_set(best_set)
    if s > best_score:
        best_score = s
        best_set = list(best_set)

    # 1. Try some structured candidates, including Base-Expansion
    structured_candidates = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81], # Squares
        [0, 1, 2, 4, 8, 16, 32, 64, 128], # Powers of 2
        [0, 1, 3, 7, 15, 31, 63, 127], # Mersenne-like
        [0, 1, 4, 10, 20, 35, 56, 84], # Triangular
        [0, 1, 3, 4, 9, 10, 12, 13], # Small Sidon-like
        [0, 1, 4, 10, 18, 28, 40, 54], # Quadratic-ish
    ]
    
    # Generate Base-Expansion candidates
    # u_k = sum b_j B^j
    for B in [3, 4, 5, 10]:
        max_digits = 4
        elements = [0]
        for mask in range(1, 1 << max_digits):
            val = 0
            for j in range(max_digits):
                if mask & (1 << j):
                    val += B ** j
            if val not in elements:
                elements.append(val)
        elements.sort()
        structured_candidates.append(elements)
        
    for cand in structured_candidates:
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # 2. Local Search with Add/Remove/Replace
    # We will run multiple restarts.
    max_val_limit = 1000
    max_n_limit = 30
    
    def get_candidates(U, limit):
        """Generate candidate elements for addition."""
        U_set = set(U)
        candidates = set()
        # Sums and differences of existing elements
        for a in U:
            for b in U:
                s = a + b
                d = abs(a - b)
                if s > 0 and s <= limit and s not in U_set:
                    candidates.add(s)
                if d > 0 and d <= limit and d not in U_set:
                    candidates.add(d)
        # Random small integers
        for _ in range(50):
            r = random.randint(1, limit)
            if r not in U_set:
                candidates.add(r)
        return list(candidates)

    def local_search(U, deadline, max_val_limit, max_n_limit):
        """
        Perform local search on U.
        Returns (best_score, best_set)
        """
        U = sorted(set(U))
        current_score = score_set(U)
        best_local_score = current_score
        best_local_set = list(U)
        
        # Candidate pool
        candidates = get_candidates(U, max_val_limit)
        
        no_improvement_count = 0
        max_no_improvement = 80
        
        while time.time() < deadline - 2:
            if no_improvement_count > max_no_improvement:
                break
                
            move_type = random.random()
            improved = False
            
            if move_type < 0.4:
                # Try to ADD an element
                if len(U) < max_n_limit and candidates:
                    # Shuffle candidates to pick a random one
                    random.shuffle(candidates)
                    # Try top few candidates
                    for c in candidates[:5]:
                        if c in U:
                            continue
                        new_U = U + [c]
                        new_U.sort()
                        new_score = score_set(new_U)
                        if new_score > current_score:
                            U = new_U
                            current_score = new_score
                            improved = True
                            # Update candidates
                            candidates = get_candidates(U, max_val_limit)
                            break
            elif move_type < 0.6:
                # Try to REMOVE an element
                if len(U) > 2:
                    # Pick a random element to remove (not 0)
                    removable = [x for x in U if x != 0]
                    if removable:
                        c = random.choice(removable)
                        new_U = [x for x in U if x != c]
                        new_score = score_set(new_U)
                        if new_score > current_score:
                            U = new_U
                            current_score = new_score
                            improved = True
                            # Update candidates
                            candidates = get_candidates(U, max_val_limit)
            else:
                # Try to REPLACE an element
                if len(U) > 1:
                    idx = random.randint(1, len(U) - 1) # Don't replace 0
                    old_val = U[idx]
                    # Get candidates
                    if not candidates:
                        candidates = get_candidates(U, max_val_limit)
                    
                    # Pick a random candidate
                    if candidates:
                        new_val = random.choice(candidates)
                        if new_val == old_val or new_val in U:
                            continue
                        new_U = U.copy()
                        new_U[idx] = new_val
                        new_U.sort()
                        new_score = score_set(new_U)
                        if new_score > current_score:
                            U = new_U
                            current_score = new_score
                            improved = True
                            # Update candidates
                            candidates = get_candidates(U, max_val_limit)
            
            if improved:
                no_improvement_count = 0
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local_set = list(U)
            else:
                no_improvement_count += 1
                
        return best_local_score, best_local_set

    # Restart loop
    restarts = 0
    while time.time() < deadline - 5:
        restarts += 1
        # Choose a starting set
        # Mix of random and structured
        if restarts % 3 == 0:
            # Start from a structured set
            base = random.choice(structured_candidates)
            U = list(base)
        else:
            # Start from random
            n = random.randint(4, 15)
            if n - 1 > max_val_limit:
                n = max_val_limit
            elements = random.sample(range(1, max_val_limit + 1), n - 1)
            U = [0] + elements
            U.sort()
        
        # Run local search
        local_score, local_set = local_search(U, deadline, max_val_limit, max_n_limit)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set

    # Final polish: Try very small moves on the best set found
    if time.time() < deadline - 2:
        # Try replacing with neighbors
        U = list(best_set)
        current_score = best_score
        # Sort
        U.sort()
        # Try small perturbations
        for _ in range(100):
            if time.time() > deadline - 2:
                break
            idx = random.randint(1, len(U) - 1)
            old_val = U[idx]
            # Try nearby values
            found_improvement = False
            for step in [-2, -1, 1, 2, 5, 10, -5, -10]:
                new_val = old_val + step
                if new_val < 1:
                    continue
                if new_val in U:
                    continue
                new_U = U.copy()
                new_U[idx] = new_val
                new_U.sort()
                new_score = score_set(new_U)
                if new_score > current_score:
                    U = new_U
                    current_score = new_score
                    found_improvement = True
                    break
            if found_improvement:
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(U)

    return best_set
