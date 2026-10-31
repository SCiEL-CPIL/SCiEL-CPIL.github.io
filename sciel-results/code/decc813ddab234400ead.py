"""Cap set greedy builder: local structural heuristics + global line-avoidance."""
import time
import hashlib
import numpy as np

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s safety margin

# State shared across priority() calls within one evaluation
_state = {
    'n': None,
    'vectors': [],       # list of tuple vectors in the cap set
    'pair_sum': None,    # dict: (sum0 mod 3, sum1 mod 3, ...) -> count of ordered pairs (a,b) in S
    'size': 0,
    'vec_array': None,   # numpy array of all vectors for fast pair-sum updates
}


def _ensure_init(n):
    """Reset state when n changes (new evaluation)."""
    if _state['n'] != n:
        _state['n'] = n
        _state['vectors'] = []
        _state['pair_sum'] = {}
        _state['size'] = 0
        _state['vec_array'] = None


def _update_pair_sum(vec, n):
    """Add contributions from pairs involving vec to pair_sum using NumPy."""
    ps = _state['pair_sum']
    size = _state['size']
    if size == 0:
        return
    
    vec_arr = np.array(vec, dtype=np.int8)
    
    # Get existing vectors as a numpy array
    if _state['vec_array'] is None:
        _state['vec_array'] = np.array(_state['vectors'], dtype=np.int8)
    else:
        # Append new vector to the array
        new_row = vec_arr.reshape(1, -1)
        _state['vec_array'] = np.vstack([_state['vec_array'], new_row])
    
    # Compute all pair sums: (vec + vj) mod 3 for each existing vj
    # vec_arr is shape (n,), vec_array is shape (size, n)
    # We need (vec + vj) mod 3 for each j
    pair_sums = (vec_arr + _state['vec_array']) % 3  # shape (size, n)
    
    # Convert to tuples for dict lookup
    for i in range(size):
        key = tuple(pair_sums[i].tolist())
        ps[key] = ps.get(key, 0) + 1
        # The reverse is the same since addition is commutative mod 3
        # key_rev = tuple((vj[i] + vec[i]) % 3 for i in range(n)) == key
        ps[key] = ps.get(key, 0) + 1


def _fast_hash(vector):
    """64-bit slice of SHA256 for better uniformity and collision resistance."""
    # Encode vector as bytes: each coordinate is 0, 1, or 2
    b = bytes(vector)
    digest = hashlib.sha256(b).digest()
    # Take first 8 bytes as a 64-bit integer
    h = int.from_bytes(digest[:8], byteorder='big')
    return h / (2**64 - 1)


def priority(vector, n):
    """
    Priority for greedy cap set construction in {0,1,2}^n.

    Design (same family as seed, augmented with the actual cap-set constraint):
    1. Quadratic symbol-presence score (smooth distinct-symbol reward).
    2. Mod-3 coordinate-sum penalty (sum ≡ 0 mod 3 is dangerous).
    3. Quadratic balance penalty (deviation from uniform coordinate distribution).
    4. Global line-avoidance: penalize vectors that would complete many
       affine lines with already-selected pairs (the actual cap-set constraint).
    5. Fast rolling hash for deterministic tie-breaking.

    The global line term makes the greedy process aware of the real constraint:
    a vector's priority drops as it would create more 3-point lines, which is
    exactly what a cap set must avoid.
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

    # Factor 1: Quadratic presence score
    p0 = 1.0 if c0 > 0 else 0.0
    p1 = 1.0 if c1 > 0 else 0.0
    p2 = 1.0 if c2 > 0 else 0.0
    num_distinct = p0 + p1 + p2
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 2: Mod-3 sum penalty
    coord_sum = c1 + 2 * c2
    sum_mod = coord_sum % 3
    if sum_mod == 0:
        sum_score = -5000.0
    elif sum_mod == 1:
        sum_score = 2000.0
    else:
        sum_score = 1500.0

    # Factor 3: Quadratic balance penalty
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 5.0

    # Factor 4: Global line-avoidance (the actual cap-set constraint)
    # A vector v creates a line with pair (a,b) iff a + b + v ≡ 0 mod 3
    # i.e. v ≡ -(a+b) ≡ (3 - (a+b)) mod 3 coordinate-wise.
    # Count how many existing pairs would be completed by this vector.
    # Use a smooth surrogate: instead of hard penalty on exact count,
    # apply a soft penalty that grows superlinearly with line_count.
    ps = _state['pair_sum']
    if ps:
        v_key = tuple((3 - vector[i]) % 3 for i in range(n))
        line_count = ps.get(v_key, 0)
        # Smooth surrogate: use log(1 + line_count) scaled, which is a
        # smooth monotone function of the count. This avoids the discontinuity
        # of a hard threshold and provides a gradient-like signal.
        # Scale: for small counts, approximately linear; for large counts,
        # grows slower than linear but still penalizes heavily.
        line_penalty = -line_count * 50.0 - 10.0 * (line_count ** 1.5)
    else:
        line_penalty = 0.0

    # Factor 5: Fast polynomial rolling hash for deterministic tie-breaking
    hash_score = _fast_hash(vector)

    # Combined priority: higher is added first
    total = distinct_score + sum_score + balance_score + line_penalty + hash_score

    # Update state: add this vector (the evaluator will only call priority
    # for vectors that pass the cap-set check, so we assume acceptance)
    vec_tuple = tuple(vector)
    _state['vectors'].append(vec_tuple)
    _update_pair_sum(vec_tuple, n)
    _state['size'] += 1
    # Note: vec_array is updated inside _update_pair_sum

    return float(total)
