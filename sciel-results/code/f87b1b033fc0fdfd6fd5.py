"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import numpy as np

def _score_set(U):
    """Compute the score for a given set U using NumPy for speed."""
    if not U or 0 not in U:
        return 0.0
    
    U_arr = np.array(U, dtype=np.int64)
    n = len(U_arr)
    
    if n < 2:
        return 0.0

    # Compute U + U and U - U using broadcasting
    # Note: For large N, this creates an NxN array. 
    # If N is too large (e.g. > 2000), this might use too much memory or time.
    # We estimate N ~ 50-100 is safe for 120s limit.
    
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    
    # Get unique elements
    # Using np.unique is efficient
    unique_sums = np.unique(sums)
    unique_diffs = np.unique(diffs)
    
    len_sum = len(unique_sums)
    len_diff = len(unique_diffs)
    
    max_u = U_arr.max()
    if max_u == 0:
        return 0.0
    
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 0.0
    
    ratio = len_diff / len_sum
    score = 1 + math.log(ratio) / log_denom
    return score

def _generate_primes(limit):
    """Generate primes up to limit."""
    if limit < 2:
        return []
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [i for i, prime in enumerate(is_prime) if prime]

def construct():
    """
    Construct a finite set U of nonnegative integers containing 0
    to maximize the sumset/difference-set exponent lower bound.
    
    Uses a greedy approach with prime gaps, optimized by local search.
    """
    # Generate a pool of primes. 
    # Smaller primes create smaller max(U), which helps the denominator log(2*maxU+1).
    # However, we need enough distinct primes to keep the set sparse.
    primes = _generate_primes(1000)
    
    # Initial greedy construction
    # Start with 0
    U = [0]
    current_score = 0.0
    
    # We want to add elements one by one. 
    # The element added is U[-1] + p.
    # We try to maximize the score.
    
    max_elements = 80 # Keep N moderate to allow fast scoring
    
    for _ in range(max_elements):
        best_prime = None
        best_score = current_score
        best_new_U = None
        
        # Try adding each prime from the pool
        # We can skip primes that are too large to keep max(U) small, 
        # but the greedy score will naturally handle the trade-off.
        for p in primes:
            new_U = U + [U[-1] + p]
            # Quick check: ensure uniqueness
            if len(set(new_U)) != len(new_U):
                continue
            
            s = _score_set(new_U)
            if s > best_score:
                best_score = s
                best_prime = p
                best_new_U = new_U
        
        if best_prime is None:
            break
            
        U = best_new_U
        current_score = best_score
        
        # If score stops improving significantly, we might break early
        # but let's run full greedy first.

    # Local Search: Try swapping or adjusting gaps
    # Gaps are U[i] - U[i-1]
    # We can try replacing a gap with a nearby prime or a different prime from the pool.
    
    for iteration in range(50): # Limit iterations
        improved = False
        gaps = [U[i] - U[i-1] for i in range(1, len(U))]
        
        for i in range(len(gaps)):
            old_gap = gaps[i]
            # Try replacing gap i with other primes
            # To save time, only try primes in a reasonable range
            candidate_primes = [p for p in primes if abs(p - old_gap) < old_gap * 2 + 10]
            
            best_candidate_gap = old_gap
            best_score = current_score
            best_new_U = None
            
            for p in candidate_primes:
                if p == old_gap:
                    continue
                
                # Construct new U
                # U[0..i] stays same
                # U[i+1] = U[i] + p
                # U[j] for j > i+1 needs to be recalculated? 
                # Actually, U is defined by cumulative sums.
                # If we change gap i, all subsequent elements change.
                
                new_gaps = gaps[:]
                new_gaps[i] = p
                
                new_U = [0]
                for g in new_gaps:
                    new_U.append(new_U[-1] + g)
                    
                if len(set(new_U)) != len(new_U):
                    continue
                
                s = _score_set(new_U)
                if s > best_score:
                    best_score = s
                    best_candidate_gap = p
                    best_new_U = new_U
            
            if best_new_U:
                U = best_new_U
                current_score = best_score
                gaps = [U[i] - U[i-1] for i in range(1, len(U))]
                improved = True
        
        # Try adding a new element (gap) at the end
        if len(U) < max_elements + 10:
             best_prime = None
             best_score = current_score
             best_new_U = None
             
             for p in primes:
                 new_U = U + [U[-1] + p]
                 if len(set(new_U)) != len(new_U):
                     continue
                 s = _score_set(new_U)
                 if s > best_score:
                     best_score = s
                     best_prime = p
                     best_new_U = new_U
             
             if best_new_U:
                 U = best_new_U
                 current_score = best_score
                 improved = True

        # Try removing the last element
        if len(U) > 5:
             new_U = U[:-1]
             s = _score_set(new_U)
             if s > current_score:
                 U = new_U
                 current_score = s
                 improved = True

        if not improved:
            break
            
    return U
# EVOLVE-BLOCK-END
