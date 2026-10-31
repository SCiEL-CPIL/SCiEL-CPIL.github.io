import time
import math

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s safety margin

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.
    
    Strategy:
    1. Primary: Entropy-based balance score - measures how uniformly the symbols
       are distributed. Higher entropy = more balanced = safer against affine lines.
    2. Secondary: Mod-3 coordinate-sum penalty - vectors with sum ≡ 0 mod 3 are
       more likely to complete affine lines with complementary pairs.
    3. Tertiary: Quadratic distinct symbol score - smooth reward for using all 3 symbols.
    4. Quaternary: Hamming weight relative to all-ones vector (count of 2s) to break
       symmetry and favor vectors "heavy" in the 2-coordinate, preventing clustering
       in the {0,1}^n sub-cube.
    5. Quinary: Fast structural hash for deterministic tie-breaking.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    # Fast symbol counting
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)

    # Factor 1: Entropy-based balance score
    # Shannon entropy of the symbol distribution: H = -sum(p_i * log(p_i))
    # Maximum entropy (log(3)) when all symbols equally present
    # Low entropy when distribution is skewed
    counts = [c0, c1, c2]
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / n
            entropy -= p * math.log(p)
    
    # Normalize entropy to [0, 1] range where 1 = perfectly balanced
    max_entropy = math.log(3)
    entropy_norm = entropy / max_entropy if max_entropy > 0 else 0.0
    
    # Higher entropy = better (more balanced) = higher priority
    entropy_score = 20000.0 * entropy_norm

    # Factor 2: Mod-3 sum penalty
    # In F_3^n, three points form an affine line iff their coordinate-wise sum ≡ 0 mod 3
    coord_sum = c1 + 2 * c2  # c0*0 + c1*1 + c2*2
    sum_mod = coord_sum % 3
    if sum_mod == 0:
        sum_score = -5000.0
    elif sum_mod == 1:
        sum_score = 2000.0
    else:
        sum_score = 1500.0

    # Factor 3: Quadratic distinct symbol score (smooth landscape)
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 4: Hamming weight relative to all-ones vector (count of 2s)
    # Vectors with more 2s are structurally "heavy" in the 2-coordinate,
    # which breaks the symmetry of the ternary grid and prevents the greedy
    # algorithm from clustering in the {0,1}^n sub-cube.
    # We use a smooth quadratic reward for c2/n to encourage uniform exploration.
    weight_ratio = c2 / n if n > 0 else 0.0
    weight_score = 5000.0 * weight_ratio * (1.0 - weight_ratio) * 4.0  # Peaks at c2 = n/2

    # Factor 5: Fast polynomial rolling hash for deterministic tie-breaking
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    # Combined priority: higher is added first
    total = entropy_score + sum_score + distinct_score + weight_score + hash_score
    return float(total)
