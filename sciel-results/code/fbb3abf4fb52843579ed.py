import time
_start_time = time.time()
_TIMEOUT = 320.0 # Safety margin

def priority(vector, n):
    # Check timeout
    if time.time() - _start_time > _TIMEOUT:
        return -1e9
    
    # Structural Change: Weight-layered Pseudo-Random Priority
    # 1. Calculate Hamming Weight (count of non-zeros)
    # 2. Calculate a high-quality hash for tie-breaking within weight classes
    # This avoids the "balance" local optimum and explores weight shells systematically
    # while using randomization to break symmetries within shells.
    
    weight = 0
    h = 0x9e3779b97f4a7c15 # Initial seed for mixing
    
    for v in vector:
        if v != 0:
            weight += 1
        # Strong mixing for hash
        # Use a simple but effective 64-bit mix
        h ^= (v + 0x9e3779b97f4a7c15 + (h << 6) + (h >> 2)) & 0xFFFFFFFFFFFFFFFF
        h = (h * 0x45d9f3b) & 0xFFFFFFFFFFFFFFFF
        h = (h ^ (h >> 16))
        
    # Priority: Lower weight is higher priority.
    # Within same weight, higher hash is higher priority (arbitrary but deterministic).
    # Scale weight to dominate the hash part.
    # Max weight is n. n is likely < 100.
    # Hash is 0..2^64.
    # We want: Priority = -weight * BIG + normalized_hash
    
    # Normalize hash to [0, 1)
    hash_float = float(h) / (1 << 64)
    
    # We want higher priority for lower weight.
    # So -weight is the main term.
    # Add hash as a small perturbation to break ties.
    return -weight * 10000.0 + hash_float
