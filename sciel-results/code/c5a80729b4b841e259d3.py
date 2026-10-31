"""Cap set priority: centering heuristic + global line-avoidance via pair-sum tracking."""
import time

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s safety margin

# State shared across priority() calls within one evaluation
_state = {
    'n': None,
    'vectors': [],
    'pair_sum': {},
    'size': 0,
}


def _ensure_init(n):
    """Reset state when n changes (new evaluation)."""
    if _state['n'] != n:
        _state['n'] = n
        _state['vectors'] = []
        _state['pair_sum'] = {}
        _state['size'] = 0


def _update_pair_sum(vec, n):
    """Add contributions from pairs involving vec to pair_sum."""
    ps = _state['pair_sum']
    size = _state['size']
    for j in range(size):
        vj = _state['vectors'][j]
        key = tuple((vec[i] + vj[i]) % 3 for i in range(n))
        ps[key] = ps.get(key, 0) + 1
        key_rev = tuple((vj[i] + vec[i]) % 3 for i in range(n))
        ps[key_rev] = ps.get(key_rev, 0) + 1


def priority(vector, n):
    """
    Priority for greedy cap set construction in {0,1,2}^n.
    
    Combines:
    1. Centering score: product of distances to all-zeros and all-twos
       (vectors with many 1s and balanced 0s/2s are geometrically central
        and less likely to form APs with boundary vectors)
    2. Diversity bonus: reward for having all three coordinate values present
    3. Global line-avoidance: penalize vectors that would complete many
       affine lines with already-selected pairs (the actual cap-set constraint)
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    _ensure_init(n)

    # Fast symbol counting
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)

    # Factor 1: Centering score (product of distances to boundaries)
    dist_to_zero = c1 + 2 * c2
    dist_to_two = 2 * c0 + c1
    center_score = dist_to_zero * dist_to_two

    # Factor 2: Diversity bonus
    diversity_bonus = 0.0
    if c0 > 0 and c1 > 0 and c2 > 0:
        min_count = min(c0, c1, c2)
        diversity_bonus = min_count * 2.0
    elif c1 > 0 and (c0 > 0 or c2 > 0):
        diversity_bonus = 0.5 * min(c0 + c2, c1)

    # Factor 3: Global line-avoidance
    # A vector v creates a line with pair (a,b) iff a + b + v ≡ 0 mod 3
    # i.e. v ≡ -(a+b) ≡ (3 - (a+b)) mod 3 coordinate-wise.
    ps = _state['pair_sum']
    if ps:
        v_key = tuple((3 - vector[i]) % 3 for i in range(n))
        line_count = ps.get(v_key, 0)
        # Scale penalty with set size to keep magnitude reasonable relative to center_score
        # center_score can be up to ~n^2, so scale line penalty accordingly
        line_penalty = -line_count * (n / 2.0 + 1.0)
    else:
        line_penalty = 0.0

    # Combined priority: higher is added first
    total = center_score + diversity_bonus + line_penalty

    # Update state: add this vector
    vec_tuple = tuple(vector)
    _state['vectors'].append(vec_tuple)
    _update_pair_sum(vec_tuple, n)
    _state['size'] += 1

    return float(total)
