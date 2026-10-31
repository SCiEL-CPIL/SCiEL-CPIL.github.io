"""Cap set construction via greedy selection with structured priorities."""
import time

# EVOLVE-BLOCK-START

# Precomputed primes for fast hashing
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
           73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
           157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
           239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317,
           331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419,
           421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503]

def _fast_hash(vector, n, seed=0):
    """Fast deterministic pseudo-random priority via linear prime-based hash."""
    h = 0
    for i, v in enumerate(vector):
        if i < len(_PRIMES):
            h += v * _PRIMES[i]
        else:
            h += v * (i * 31 + seed + 17)
    # Mix with seed and reduce to [0, 1)
    h = (h * 1000003 + seed * 999983) % (1 << 60)
    return h / (1 << 60)

def _structured_priority(vector, n):
    """
    Structured priority for cap set construction.
    Favors vectors with balanced 1s and 2s, and uses multiple hash layers
    for diversity.
    """
    n1 = 0
    n2 = 0
    coord_sum = 0
    for v in vector:
        if v == 1:
            n1 += 1
            coord_sum += 1
        elif v == 2:
            n2 += 1
            coord_sum += 2
    
    # Balance: prefer n1 and n2 to be close to each other
    balance = abs(n1 - n2) / max(1, n)
    
    # Sum mod 3: vectors with sum % 3 == 1 or 2 may be safer
    coord_sum_mod = coord_sum % 3
    
    # Multiple hash layers for robustness (fast)
    h1 = _fast_hash(vector, n, seed=1)
    h2 = _fast_hash(vector, n, seed=2)
    h3 = _fast_hash(vector, n, seed=3)
    
    # Combine structural and hash components
    struct_score = (1.0 - balance) * 0.4
    # Small bonus for sum mod 3 != 0
    mod_bonus = 0.1 if coord_sum_mod != 0 else 0.0
    hash_score = (h1 * 0.4 + h2 * 0.35 + h3 * 0.25) * 0.5
    
    return struct_score + mod_bonus + hash_score

def priority(vector, n):
    """
    Priority function for greedy cap set construction.
    Higher priority = selected earlier.
    """
    return _structured_priority(vector, n)

# EVOLVE-BLOCK-END
