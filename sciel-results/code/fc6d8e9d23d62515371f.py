import zlib
import time

_START = time.time()
_DEADLINE = _START + 280.0  # safety margin before 330s kill

def priority(vector, n):
    """
    Improved structural priority for cap set construction.
    
    Key insight: ANY subset of {0,2}^n is a cap set.
    For vectors with 1s, we need to minimize conflict potential.
    
    Strategy:
    1. Vectors with no 1s get highest priority (always safe)
    2. Vectors with 1s: score based on conflict potential
       - Fewer 1s = lower conflict potential
       - Balanced distribution across coordinates reduces conflict
    3. Deterministic hash for tie-breaking
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        return 0.0
    
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    # Vectors with no 1s are always safe - give them priority > 1.0
    if c1 == 0:
        data = bytes(vector)
        crc = zlib.crc32(data)
        return 1.0 + float(crc) / 0xFFFFFFFF
    
    # For vectors with 1s, compute conflict potential
    # Vectors with many 1s are more likely to form forbidden triples
    # Also, vectors where 1s are concentrated in few coordinates may be worse
    
    if n > 0:
        # Base score: exponential decay on count of 1s
        one_score = 2.718281828459045 ** (-0.5 * c1 / n)
        
        # Bonus for vectors that also have some 2s (more diverse)
        # Vectors with only 0s and 1s can only conflict in limited ways
        diversity_bonus = min(c2 / max(n, 1), 1.0) * 0.1
        
        # Penalty for high concentration of 1s
        # If all 1s are in the first few coordinates, that's more dangerous
        one_positions = [i for i, v in enumerate(vector) if v == 1]
        if len(one_positions) > 0:
            # Measure spread: higher variance in positions = better
            mean_pos = sum(one_positions) / len(one_positions)
            if len(one_positions) > 1:
                variance = sum((p - mean_pos) ** 2 for p in one_positions) / len(one_positions)
                spread_bonus = min(variance / (n * n / 4), 1.0) * 0.05
            else:
                spread_bonus = 0.0
        else:
            spread_bonus = 0.0
    else:
        one_score = 1.0
        diversity_bonus = 0.0
        spread_bonus = 0.0
    
    # Hash for deterministic tie-breaking
    data = bytes(vector)
    crc = zlib.crc32(data)
    hash_score = float(crc) / 0xFFFFFFFF
    
    # Combine: exponential decay dominates, small bonuses for diversity and spread
    score = 0.85 * one_score + 0.05 * diversity_bonus + 0.05 * spread_bonus + 0.05 * hash_score
    
    return score
