import time
import math

_start_time = time.time()
_DEADLINE = _start_time + 280.0

def priority(vector, n):
    if time.time() > _DEADLINE:
        return 0.0

    # Fast counting of symbols
    # vector is a tuple of 0s, 1s, 2s
    # We want to maximize the "balancedness" of the vector.
    # A vector with counts close to n/3 is preferred.
    
    c0 = 0
    c1 = 0
    c2 = 0
    
    # Manual loop is faster than list.count for small n? 
    # Or just sum(1 for v in vector if v==0)
    # Given n is small, simple loop is fine.
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
            
    ideal = n / 3.0
    
    # Squared deviation from ideal
    dev = (c0 - ideal)**2 + (c1 - ideal)**2 + (c2 - ideal)**2
    
    # We want to minimize deviation. 
    # Score should be high for low deviation.
    # Max possible deviation: one count is n, others 0.
    # dev_max = (n - n/3)^2 + (0 - n/3)^2 + (0 - n/3)^2
    #         = (2n/3)^2 + (-n/3)^2 + (-n/3)^2
    #         = 4n^2/9 + n^2/9 + n^2/9 = 6n^2/9 = 2n^2/3
    max_dev = 2.0 * n * n / 3.0
    
    if max_dev == 0:
        norm_dev = 0.0
    else:
        norm_dev = dev / max_dev
        
    # Base score: 1.0 for perfect balance, 0.0 for worst imbalance
    base_score = 1.0 - norm_dev
    
    # Tie-breaking: 
    # We need a deterministic, high-quality hash to break ties between vectors with same balance.
    # Using Python's built-in hash is not stable across runs (randomized).
    # We use a simple deterministic mix.
    # To make it fast, we can use the vector's value as a base-3 number?
    # Or just sum of (v_i * weight_i)?
    # A better tie-breaker is a hash that looks random.
    # Let's use a simple 64-bit mix of the vector components.
    
    h = 0x811c9dc5  # FNV offset basis
    for v in vector:
        h ^= v
        h = (h * 0x01000193) & 0xFFFFFFFFFFFFFFFF
        
    # Map hash to a small range [0, 1) to add tiny perturbation
    # This ensures that if two vectors have the same balance, the one with a "better" hash 
    # (arbitrarily defined) is picked first. This prevents systematic biases in the greedy choice
    # for vectors with identical balance profiles.
    tie_breaker = (h / 0x10000000000000000) * 0.01
    
    return float(base_score + tie_breaker)
