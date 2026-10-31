"""Cap set construction via greedy selection with structured priorities."""
import time
import hashlib

# EVOLVE-BLOCK-START

# Precompute some magic numbers for fast hashing
_MAGIC_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
                 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
                 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181,
                 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251,
                 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317,
                 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397,
                 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
                 467, 479, 487, 491, 499]

def _fast_hash(vector, n, seed=0):
    """
    Fast deterministic pseudo-random priority using a simple linear combination
    with magic primes. This is much faster than MD5 for small n.
    """
    h = seed
    for i in range(n):
        h = (h * 31 + vector[i] * _MAGIC_PRIMES[i % len(_MAGIC_PRIMES)]) % (2**32)
    return h / (2**32)

def _structured_priority(vector, n):
    """
    Structured priority for cap set construction.
    Favors vectors with higher Hamming weight (more 2s and 1s) to break
    symmetry and encourage uniform exploration of {0,1,2}^n, combined with
    balance and hash layers for diversity.
    """
    n1 = 0
    n2 = 0
    total_sum = 0
    for v in vector:
        if v == 1:
            n1 += 1
        elif v == 2:
            n2 += 1
        total_sum += v
    
    # Hamming weight: count of non-zero coordinates
    hamming_weight = n1 + n2
    # Normalize to [0, 1]
    weight_score = hamming_weight / max(1, n)
    
    # Bonus for 2s specifically (they are "farther" in F_3 geometry)
    twos_score = n2 / max(1, n)
    
    # Balance: prefer n1 and n2 to be close to each other and to n/3
    target = n / 3.0
    balance = abs(n1 - n2) / max(1, n)
    
    # Sum mod 3: vectors with sum % 3 == 1 or 2 may be safer
    coord_sum = total_sum % 3
    
    # Multiple hash layers for robustness
    h1 = _fast_hash(vector, n, seed=1)
    h2 = _fast_hash(vector, n, seed=2)
    h3 = _fast_hash(vector, n, seed=3)
    
    # Combine structural and hash components
    # Weight component: favor higher Hamming weight and more 2s
    weight_component = (weight_score * 0.5 + twos_score * 0.5) * 0.4
    # Balance component: prefer balanced 1s and 2s
    balance_component = (1.0 - balance) * 0.3
    # Small bonus for sum mod 3 != 0
    mod_bonus = 0.05 if coord_sum != 0 else 0.0
    # Hash component for diversity
    hash_score = (h1 * 0.4 + h2 * 0.35 + h3 * 0.25) * 0.25
    
    return weight_component + balance_component + mod_bonus + hash_score

def priority(vector, n):
    """
    Priority function for greedy cap set construction.
    Higher priority = selected earlier.
    """
    return _structured_priority(vector, n)

# EVOLVE-BLOCK-END
