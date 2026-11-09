import time

_START_TIME = time.time()
_TIME_LIMIT = 280.0  # Safety margin

def _xorshift(seed):
    """Simple XOR-based hash for deterministic tie-breaking."""
    x = seed & 0xFFFFFFFF
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= (x >> 17)
    x ^= (x << 5) & 0xFFFFFFFF
    return x & 0xFFFFFFFF

def _better_hash(vector):
    """More robust hash for tie-breaking."""
    h = 0x811c9dc5  # FNV offset basis
    for v in vector:
        h ^= v
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h

def priority(vector, n):
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        return 0.0
    
    # Fast symbol count
    n0 = 0
    n1 = 0
    n2 = 0
    for v in vector:
        if v == 0:
            n0 += 1
        elif v == 1:
            n1 += 1
        else:
            n2 += 1
    
    # Spectral penalty: penalize high correlation to all-ones vector
    # In the Fourier domain over Z_3^n, the all-ones vector has a large DC component.
    # Vectors with many 1s have higher "energy" in the low-frequency spectrum,
    # which correlates with forming 3-term APs.
    
    # Balance score: minimize variance from ideal n/3 distribution
    target = n / 3.0
    var = (n0 - target)**2 + (n1 - target)**2 + (n2 - target)**2
    balance_score = -var
    
    # Midpoint risk penalty: vectors with many 1s are more likely to be midpoints
    # of 3-term APs (since (a+c)/2 = b when a+c ≡ 2b mod 3)
    midpoint_penalty = -(n1 * n1) / max(n, 1)
    
    # 0/2 imbalance penalty: asymmetric distribution of 0s and 2s increases AP risk
    zero_two_imbalance = (n0 - n2)**2 / max(n, 1)
    
    # Combine spectral and structural penalties
    # Scale to make structural terms comparable to balance term
    structural_score = balance_score + 0.5 * midpoint_penalty - 0.3 * zero_two_imbalance
    
    # Deterministic tie-breaker
    h = _better_hash(vector)
    hash_score = (h % 100000) / 100000.0
    
    # Return combined score
    return structural_score * 100.0 + hash_score
