"""Cap set construction via greedy selection with structured priorities."""
import time

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
    Favors vectors with balanced 1s and 2s, and uses multiple hash layers
    for diversity.
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
    # Count occurrences of each value
    count_0 = 0
    count_1 = 0
    count_2 = 0
    for v in vector:
        if v == 0:
            count_0 += 1
        elif v == 1:
            count_1 += 1
        else:
            count_2 += 1
    
    # Product of distances to all-zeros and all-twos
    dist_to_zero = count_1 + 2 * count_2
    dist_to_two = 2 * count_0 + count_1
    center_score = dist_to_zero * dist_to_two
    
    # Diversity bonus: reward having all three values present
    diversity_bonus = 0.0
    if count_0 > 0 and count_1 > 0 and count_2 > 0:
        # Scale bonus by how balanced the distribution is
        min_count = min(count_0, count_1, count_2)
        diversity_bonus = min_count * 2.0
    elif count_1 > 0 and (count_0 > 0 or count_2 > 0):
        # Partial bonus for having 1s and at least one other type
        diversity_bonus = 0.5 * min(count_0 + count_2, count_1)
    
    # Combine the geometric center score with the structured hash-based priority
    # The center score is scaled down to avoid dominating the hash component
    return center_score * 0.3 + diversity_bonus * 0.2 + _structured_priority(vector, n)

# EVOLVE-BLOCK-END
