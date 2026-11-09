"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

_T0 = time.time()
_DEADLINE = _T0 + 285.0  # 45s safety margin before 330s kill

def priority(vector, n):
    """
    Priority for a vector in {0,1,2}^n.
    
    Strategy:
    1. Balance: Prefer vectors with counts of 0, 1, 2 close to n/3.
    2. Diversity: Penalize vectors that are 'monochromatic' in too many coordinates.
    3. Deterministic tie-break to avoid symmetry issues.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a low priority to effectively stop adding vectors or just return a default
        # Actually, if we are past deadline, the evaluator might kill us anyway.
        # But returning a very low score ensures no more vectors are added if the loop checks.
        # However, the contract says "finish within 300s". 
        # If we are past deadline, we should probably just return quickly.
        # The greedy loop will likely stop or we will be killed.
        # Returning a very low value ensures this vector is not added.
        return -1e9

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
            
    # Target is n/3
    target = n / 3.0
    
    # Deviation from balance (L1 norm)
    deviation = abs(c0 - target) + abs(c1 - target) + abs(c2 - target)
    
    # We want higher priority for lower deviation
    score = -deviation
    
    # Deterministic tie-breaker
    # Use a more complex hash to break symmetries better
    # Simple weighted sum can be too correlated.
    # Let's use a polynomial hash.
    h = 0
    for v in vector:
        h = (h * 31 + v) & 0xFFFFFFFF
    score += (h / 4294967295.0) * 1e-6
    
    return score

# EVOLVE-BLOCK-END
