import math
import random
import time

# Global cache for samples
SAMPLE_CACHE = {}
START_TIME = time.time()
DEADLINE = 300.0 # Safety margin for 330s limit

def _is_balanced(vector, n, threshold_factor=1.0):
    """
    Check if a vector is sufficiently balanced.
    Uses chi-squared distance from ideal distribution (n/3 of each).
    """
    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
            
    ideal = n / 3.0
    # Chi-squared statistic
    chi_sq = (c0 - ideal)**2 + (c1 - ideal)**2 + (c2 - ideal)**2
    
    # Threshold: We want vectors that are "close" to balanced.
    # Variance of counts in a random vector is roughly n/9.
    # Sum of 3 variances is n/3.
    # We want vectors significantly better than random.
    # Let's say within 0.5 * n/3? Or fixed absolute?
    # For large n, chi_sq ~ n/3 is average. We want < n/6 maybe?
    # Let's use a relative threshold.
    # If n is small, this needs care.
    
    # A standard deviation of counts is approx sqrt(n/12) for each? No.
    # Let's just use a threshold that selects the top ~50% or so of balanced vectors.
    # Actually, for the sample, we want VERY balanced vectors to represent the "core" of the cap set.
    # For the conflict check, we want to know if the partner w is "good enough" to be in the cap set.
    
    return chi_sq < (n / 3.0) * 0.5 # Heuristic threshold

def _get_sample(n, sample_size=200):
    """
    Generate a deterministic sample of balanced vectors for the given n.
    """
    global SAMPLE_CACHE
    if n in SAMPLE_CACHE:
        return SAMPLE_CACHE[n]
        
    # Deadline check
    if time.time() > DEADLINE:
        return []

    samples = []
    # Use a fixed seed based on n for determinism
    rng = random.Random(42 + n)
    
    # We need to generate vectors that are balanced.
    # Simple rejection sampling: generate random vectors, keep if balanced.
    attempts = 0
    max_attempts = sample_size * 100
    while len(samples) < sample_size and attempts < max_attempts:
        # Generate random vector
        vec = tuple(rng.choice([0, 1, 2]) for _ in range(n))
        if _is_balanced(vec, n, 0.5):
            samples.append(vec)
        attempts += 1
        
    SAMPLE_CACHE[n] = samples
    return samples

def priority(vector, n):
    """
    Prioritizes vectors that have fewer potential conflicts with the "balanced core" of the space.
    A conflict exists if there is a balanced vector u such that -u-v is also balanced.
    """
    # Deadline guard
    if time.time() > DEADLINE:
        # Fallback to simple balance if we are running out of time
        c0 = sum(1 for v in vector if v == 0)
        c1 = sum(1 for v in vector if v == 1)
        c2 = n - c0 - c1
        ideal = n / 3.0
        return -((c0 - ideal)**2 + (c1 - ideal)**2 + (c2 - ideal)**2)

    # First, ensure the candidate vector itself is reasonably balanced.
    # If it's not balanced, it's likely a poor choice for a large cap set.
    # We can penalize unbalanced vectors heavily.
    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
            
    ideal = n / 3.0
    chi_sq_self = (c0 - ideal)**2 + (c1 - ideal)**2 + (c2 - ideal)**2
    
    # If the vector is very unbalanced, give it low priority immediately
    if chi_sq_self > n: # Arbitrary threshold for "very unbalanced"
        return -1000.0 - chi_sq_self

    # Get the sample of "good" vectors
    samples = _get_sample(n, 100) # Use smaller sample for speed
    
    if not samples:
        # Fallback
        return -chi_sq_self

    # Count conflicts
    # For each u in samples, check if w = -u-v is balanced.
    conflicts = 0
    for u in samples:
        # Compute w = (-u - v) % 3
        # We can do this coordinate-wise
        # Optimization: pre-compute -u mod 3?
        # Let's just compute w on the fly.
        
        # Check if w is balanced.
        # w_i = (3 - u_i - v_i) % 3
        
        wc0 = 0
        wc1 = 0
        wc2 = 0
        for i in range(n):
            wi = (3 - u[i] - vector[i]) % 3
            if wi == 0:
                wc0 += 1
            elif wi == 1:
                wc1 += 1
            else:
                wc2 += 1
        
        w_ideal = n / 3.0
        w_chi_sq = (wc0 - w_ideal)**2 + (wc1 - w_ideal)**2 + (wc2 - w_ideal)**2
        
        # If w is balanced, it's a conflict
        if w_chi_sq < (n / 3.0) * 0.5:
            conflicts += 1
            
    # The fewer conflicts, the higher the priority.
    # Scale the conflict count to dominate the score.
    # We also want to prefer the vector's own balance as a tie-breaker.
    
    # Normalize conflict score
    # Max conflicts is len(samples)
    conflict_score = -conflicts * 10.0
    
    # Add own balance as secondary
    balance_score = -chi_sq_self
    
    return conflict_score + balance_score
