import time
from collections import Counter

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.
    
    Strategy:
    1. Primary: Prioritize vectors with all 3 distinct symbols.
    2. Secondary: Prioritize vectors with higher 'Sum of Squares' (c1 + 4*c2).
       This breaks the symmetry of the ternary grid by favoring vectors with
       more 2s, preventing clustering in the {0,1}^n sub-cube.
    3. Tertiary: Fast structural hash for deterministic tie-breaking.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        # Past deadline: return a constant to signal "stop caring about ordering"
        return 0.0

    # Fast symbol counting using Counter (C-implemented)
    c = Counter(vector)
    c0 = c.get(0, 0)
    c1 = c.get(1, 0)
    c2 = c.get(2, 0)
    
    # Distinct count (3 is best)
    # We use a large gap to ensure distinctness is the dominant factor.
    if c0 > 0 and c1 > 0 and c2 > 0:
        distinct_score = 30000.0
    elif (c0 > 0 and c1 > 0) or (c0 > 0 and c2 > 0) or (c1 > 0 and c2 > 0):
        distinct_score = 20000.0
    else:
        distinct_score = 10000.0
        
    # Sum of Squares: c1*1^2 + c2*2^2 = c1 + 4*c2
    # This metric heavily favors 2s over 1s and 0s.
    # Range: 0 to 4n.
    # We scale this to be significant but secondary to distinctness.
    # If distinctness gap is 10000, and n is up to ~100, sum_sq is up to 400.
    # Scaling by 100 gives up to 40000, which might overwhelm distinctness.
    # Let's scale by 50. Max 4n*50 = 200n. If n=100, max 20000.
    # This allows a vector with 2 distinct symbols but very high weight to potentially
    # beat a vector with 3 distinct symbols but low weight?
    # The strategy "favors vectors with more 2s" suggests weight is important.
    # However, distinctness is structurally better for cap sets.
    # Let's keep distinctness dominant but allow weight to strongly influence ordering within distinctness classes.
    # Scale by 10: Max 40n. If n=100, 4000. Gap is 10000. Safe.
    sum_sq = c1 + 4.0 * c2
    weight_score = sum_sq * 10.0

    # Fast hash: simple polynomial rolling hash
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    
    hash_score = h / 0x7FFFFFFF
    
    return float(distinct_score + weight_score + hash_score)
