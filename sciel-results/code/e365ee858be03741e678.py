"""Find N <= 10^18 with many representations as sum of two powerful numbers."""
import time
import random
from math import isqrt

def _get_primes_1_mod_4(limit):
    """Generate primes <= limit that are 1 mod 4."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(5, limit + 1) if sieve[i] and i % 4 == 1]

def _find_best_K(limit_K):
    """
    Find K <= limit_K maximizing the number of representations
    of K as a sum of two squares using a greedy heuristic.
    """
    if limit_K < 1:
        return 1, 1
    
    # Primes 1 mod 4 up to 1000
    primes = _get_primes_1_mod_4(1000)
    
    # Greedy construction
    # We want to maximize product of (e_i + 1)
    # Start with K=1, score=1
    K = 1
    exponents = {p: 0 for p in primes}
    current_score = 1
    
    # We will try to increase exponents.
    # At each step, find the prime p such that K * p <= limit_K
    # and the gain in score is maximized.
    # Gain is multiplicative: score_new = score_old * (e_p + 2) / (e_p + 1)
    # We want to maximize the new total score.
    # Since score is multiplicative, we can just track the potential.
    
    # A simple greedy: repeatedly multiply by the smallest prime that fits?
    # No, that minimizes K, not maximizes score.
    # We want to balance exponents.
    
    # Let's use a priority queue or just iterate.
    # Given the small number of primes, we can just loop.
    
    improved = True
    while improved:
        improved = False
        best_p = None
        best_new_score = current_score
        best_new_K = K
        
        for p in primes:
            if K * p > limit_K:
                continue
            e = exponents[p]
            # New score factor for this prime would be (e+2) instead of (e+1)
            # Ratio is (e+2)/(e+1)
            # We want to maximize the total score.
            # Since all other factors are constant, we just pick the one that allows the multiplication.
            # Actually, we should pick the one that gives the best "value per log unit"?
            # For highly composite numbers, we usually keep exponents non-increasing.
            # Let's try a simpler heuristic: try all primes, pick the one that results in the largest K * p (closest to limit) ? No.
            # Pick the one that maximizes the score? All single-step moves increase score.
            # The "best" move is the one that leaves the most room for future moves?
            # Or the one with the smallest p?
            # Let's try: pick the smallest p that fits. This tends to keep K small and exponents balanced.
            
            # Actually, a known good strategy for this specific problem is to keep exponents as equal as possible for the smallest primes.
            # Let's just pick the smallest prime p such that K*p <= limit_K.
            # This is a standard greedy for highly composite-like numbers.
            
            if best_p is None or p < best_p:
                best_p = p
                best_new_K = K * p
                # We don't really need to compare scores here if we just pick smallest p
                # But let's stick to the "smallest p" heuristic as it's robust.
                break # Found the smallest p that fits
        
        if best_p is not None:
            K = best_new_K
            exponents[best_p] += 1
            current_score *= (exponents[best_p] + 1) / exponents[best_p] # This is not integer math, but we just need to track if we improved
            # Actually, we don't need the exact score for the loop, just to return it.
            # Let's recalculate score at the end to be safe.
            improved = True
            
    # Recalculate exact score
    final_score = 1
    for p in primes:
        if exponents[p] > 0:
            final_score *= (exponents[p] + 1)
            
    return K, final_score

def _find_sum_of_squares_pairs(K):
    """Find all pairs (u, v) with u <= v, u^2 + v^2 = K."""
    pairs = []
    max_u = isqrt(K // 2)
    for u in range(0, max_u + 1):
        v2 = K - u * u
        v = isqrt(v2)
        if v * v == v2 and u <= v:
            pairs.append((u, v))
    return pairs

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    # As analyzed, Q=1 is optimal for maximizing the count of representations.
    # We just need to find the best K <= 10^18.
    limit_K = 10**18
    K, score = _find_best_K(limit_K)
    
    N = K
    uv_pairs = _find_sum_of_squares_pairs(K)
    
    pw_pairs = []
    for u, v in uv_pairs:
        a = u * u
        b = v * v
        if a <= b:
            pw_pairs.append([a, b])
    
    # Ensure we have at least one pair
    if not pw_pairs:
        # Fallback to N=2, 1+1
        return {'N': 2, 'pairs': [[1, 1]]}
        
    return {'N': N, 'pairs': pw_pairs}
