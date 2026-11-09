import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin within 90s limit
    
    def compute_scores(A):
        """Compute max(|A+A|, |A*A|) using NumPy for speed."""
        A = np.array(A)
        # A+A
        # We need distinct sums. 
        # Using broadcasting creates a 20x20 matrix.
        sums = A[:, None] + A[None, :]
        # We only care about unique values. 
        # Flattening and using np.unique is fast for small N.
        unique_sums = np.unique(sums)
        
        # A*A
        prods = A[:, None] * A[None, :]
        unique_prods = np.unique(prods)
        
        return max(len(unique_sums), len(unique_prods))

    def is_valid(A):
        """Check if A is valid: 20 distinct positive integers <= 10000"""
        if len(A) != 20:
            return False
        if len(set(A)) != 20:
            return False
        if any(x <= 0 or x > 10000 for x in A):
            return False
        return True
        
    def get_5_smooth_numbers(limit=10000, count=100):
        """Generate 5-smooth numbers (Hamming numbers) up to limit."""
        # Sieve-like approach or heap
        # For simplicity and speed, generate all 5-smooth numbers <= 10000
        # 2^a * 3^b * 5^c <= 10000
        smooths = set([1])
        for a in range(0, 14): # 2^13 = 8192
            p2 = 2**a
            if p2 > limit: break
            for b in range(0, 9): # 3^8 = 6561
                p3 = 3**b
                if p2 * p3 > limit: break
                for c in range(0, 6): # 5^5 = 3125
                    p5 = 5**c
                    val = p2 * p3 * p5
                    if val > limit: break
                    smooths.add(val)
        return sorted(list(smooths))

    # 1. Generate Initial Candidates
    
    best_score = float('inf')
    best_set = set(range(1, 21))
    
    smooth_numbers = get_5_smooth_numbers()
    
    # Strategy A: Random subsets of smooth numbers
    # Smooth numbers have small products. Sums of small numbers are also relatively compact.
    for _ in range(200):
        if time.time() > deadline: return sorted(best_set)
        # Pick 20 random smooth numbers
        if len(smooth_numbers) >= 20:
            A = random.sample(smooth_numbers, 20)
            if is_valid(A):
                score = compute_scores(A)
                if score < best_score:
                    best_score = score
                    best_set = set(A)
    
    # Strategy B: Structured AP/GP mix (keep from original but faster)
    # Pure AP
    for d in range(1, 10):
        for a in range(1, 50):
            A = [a + i*d for i in range(20)]
            if A[-1] > 10000: continue
            if is_valid(A):
                score = compute_scores(A)
                if score < best_score:
                    best_score = score
                    best_set = set(A)
            if time.time() > deadline: return sorted(best_set)
            
    # Pure GP (small r)
    for r in range(2, 5):
        for b in range(1, 20):
            A = [b * (r ** i) for i in range(20)]
            if A[-1] > 10000: break
            if is_valid(A):
                score = compute_scores(A)
                if score < best_score:
                    best_score = score
                    best_set = set(A)
            if time.time() > deadline: return sorted(best_set)

    # 2. Greedy Coordinate Descent with Local Search
    A = sorted(best_set)
    current_score = best_score
    
    # Pre-compute some prime factors for multiplicative neighbors
    primes = [2, 3, 5, 7, 11, 13]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(20):
            if time.time() > deadline: return sorted(best_set)
            old_val = A[i]
            best_candidate_val = old_val
            best_candidate_score = current_score
            
            # Generate candidates for position i
            candidates = set()
            
            # 1. Additive neighbors
            for delta in range(-10, 11):
                if delta == 0: continue
                new_val = old_val + delta
                if 1 <= new_val <= 10000:
                    candidates.add(new_val)
            
            # 2. Multiplicative neighbors (smooth number lattice moves)
            for p in primes:
                for q in primes:
                    if p == q: continue
                    # Move from old_val to old_val * p / q
                    # Ensure divisibility or just round? 
                    # Better: check if old_val is divisible by q, then *p
                    if old_val % q == 0:
                        new_val = (old_val // q) * p
                        if 1 <= new_val <= 10000:
                            candidates.add(new_val)
                    # Also try just *p and //p
                    new_val = old_val * p
                    if 1 <= new_val <= 10000:
                        candidates.add(new_val)
                    if old_val % p == 0:
                        new_val = old_val // p
                        if 1 <= new_val <= 10000:
                            candidates.add(new_val)
                        
            # 3. Random local perturbations
            for _ in range(5):
                # Pick a random value in a window around old_val
                window = max(1, old_val // 10)
                new_val = old_val + random.randint(-window, window)
                if 1 <= new_val <= 10000:
                    candidates.add(new_val)
                
            # Filter candidates: must be distinct from other elements
            A_set = set(A)
            for c in candidates:
                if c in A_set and c != old_val:
                    continue
                # Construct new set
                A_new = A[:i] + [c] + A[i+1:]
                # Quick validity check (distinctness already handled by set logic above, but ensure positive int)
                if len(set(A_new)) != 20:
                    continue
                if any(x <= 0 or x > 10000 for x in A_new):
                    continue
                
                score = compute_scores(A_new)
                if score < best_candidate_score:
                    best_candidate_score = score
                    best_candidate_val = c
            
            if best_candidate_val != old_val:
                A[i] = best_candidate_val
                A = sorted(A)
                current_score = best_candidate_score
                best_set = set(A)
                best_score = current_score
                improved = True
                
    return sorted(best_set)
