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
    1. Balance: Prefer vectors with counts of 0, 1, 2 close to n/3 using a smooth penalty.
    2. Mixed-coordinate reward: Prefer vectors with diverse coordinate values.
    3. Deterministic position-dependent hash tie-break.
    """
    # Check deadline
    if time.time() > _DEADLINE:
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
    
    # Smooth penalty: use L2 norm squared but with a slight softening
    # to avoid over-penalizing moderate imbalances
    quad_dev = (c0 - target) ** 2 + (c1 - target) ** 2 + (c2 - target) ** 2
    
    # We want higher priority for lower deviation
    score = -quad_dev
    
    # Reward vectors that have all three values present (more "central" in composition space)
    # This helps avoid corner subspaces
    if c0 > 0 and c1 > 0 and c2 > 0:
        score += 0.5 * min(c0, c1, c2) / n
    
    # Additional: penalize vectors that are too "pure" (many same values)
    max_count = max(c0, c1, c2)
    if max_count > 0.6 * n:
        score -= (max_count - 0.6 * n) ** 2 / n
    
    # Position-dependent hash for better symmetry breaking
    # Use a weighted sum with position-dependent coefficients
    h = 0
    for i, v in enumerate(vector):
        h = (h * 37 + v * (i + 1)) & 0xFFFFFFFF
    score += (h / 4294967295.0) * 1e-6
    
    return score

# EVOLVE-BLOCK-END
