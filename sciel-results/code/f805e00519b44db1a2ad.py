"""Cap set construction via greedy selection with structured priorities."""
import time
import hashlib
from itertools import product

# EVOLVE-BLOCK-START

def _hash_priority(vector, n, seed=0):
    """Deterministic pseudo-random priority via MD5 hash with seed."""
    key = bytes(vector) + bytes([seed])
    digest = hashlib.md5(key).digest()
    val = int.from_bytes(digest[:8], 'big')
    return val / (1 << 64)

def _structured_priority(vector, n):
    """
    Structured priority for cap set construction.
    Favors vectors with balanced 1s and 2s, and uses multiple hash layers
    for diversity.
    """
    n1 = sum(1 for v in vector if v == 1)
    n2 = sum(1 for v in vector if v == 2)
    
    # Balance: prefer n1 and n2 to be close to each other and to n/3
    target = n / 3.0
    balance = abs(n1 - n2) / max(1, n)
    
    # Sum mod 3: vectors with sum % 3 == 1 or 2 may be safer
    coord_sum = sum(vector) % 3
    
    # Multiple hash layers for robustness
    h1 = _hash_priority(vector, n, seed=1)
    h2 = _hash_priority(vector, n, seed=2)
    h3 = _hash_priority(vector, n, seed=3)
    
    # Combine structural and hash components
    struct_score = (1.0 - balance) * 0.4
    # Small bonus for sum mod 3 != 0
    mod_bonus = 0.1 if coord_sum != 0 else 0.0
    hash_score = (h1 * 0.4 + h2 * 0.35 + h3 * 0.25) * 0.5
    
    return struct_score + mod_bonus + hash_score

def priority(vector, n):
    """
    Priority function for greedy cap set construction.
    Higher priority = selected earlier.
    """
    return _structured_priority(vector, n)

# EVOLVE-BLOCK-END
