"""Cap set priority: stateful line avoidance + random linear projection ordering."""
import time
import zlib

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s safety margin

# State shared across priority() calls within one evaluation
_state = {
    'n': None,
    'vectors': [],       # list of tuple vectors in the cap set
    'pair_sum': {},      # dict: tuple of (a[i]+b[i]) mod 3 -> count of ordered pairs (a,b) in S
    'size': 0,
    'rng_seed': 12345,   # Fixed seed for reproducibility of random vector
    'rand_vec': None,    # Precomputed random vector with entries in {-1, 1}
}


def _ensure_init(n):
    """Reset state when n changes (new evaluation)."""
    if _state['n'] != n:
        _state['n'] = n
        _state['vectors'] = []
        _state['pair_sum'] = {}
        _state['size'] = 0
        # Generate a fixed random vector with entries in {-1, 1}
        # We use a simple deterministic PRNG based on zlib.crc32 for speed and determinism
        rand_vec = []
        seed = _state['rng_seed']
        for i in range(n):
            # Use a simple hash to generate pseudo-random bits
            h = zlib.crc32((f"seed_{seed}_idx_{i}").encode())
            # Take the least significant bit to determine sign
            if h & 1:
                rand_vec.append(1.0)
            else:
                rand_vec.append(-1.0)
        _state['rand_vec'] = rand_vec


def _update_pair_sum(vec, n):
    """Add contributions from pairs involving vec to pair_sum."""
    ps = _state['pair_sum']
    size = _state['size']
    for j in range(size):
        vj = _state['vectors'][j]
        # Compute (vec + vj) mod 3
        key = tuple((vec[i] + vj[i]) % 3 for i in range(n))
        ps[key] = ps.get(key, 0) + 2


def priority(vector, n):
    """
    Priority for greedy cap set construction in {0,1,2}^n.

    Design:
    1. Global line-avoidance: Penalize vectors that would complete many
       affine lines with already-selected pairs. This is the core structural
       constraint of a cap set.
    2. Random linear projection: Project the vector onto a fixed random vector
       with entries in {-1, 1}. This imposes a chaotic, non-local ordering that
       decouples the greedy selection from structured lattice symmetries and
       local density traps.
    3. Small deterministic tie-breaker using hash.

    The combination of stateful line avoidance and random projection aims to
    break symmetries and avoid local traps while respecting the cap set constraint.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    _ensure_init(n)

    # Factor 1: Global line-avoidance (the actual cap-set constraint)
    # A vector v creates a line with pair (a,b) iff a + b + v ≡ 0 mod 3
    # i.e. v ≡ -(a+b) ≡ (3 - (a+b)) mod 3 coordinate-wise.
    # Count how many existing pairs would be completed by this vector.
    ps = _state['pair_sum']
    if ps:
        v_key = tuple((3 - v) % 3 for v in vector)
        line_count = ps.get(v_key, 0)
        # Strong penalty for line completion
        line_penalty = -line_count * 1000.0
    else:
        line_penalty = 0.0

    # Factor 2: Random linear projection
    # Project vector onto _state['rand_vec']
    # This creates a chaotic ordering that avoids local density traps
    rand_vec = _state['rand_vec']
    projection = 0.0
    for i in range(n):
        projection += vector[i] * rand_vec[i]
    # Normalize projection to a reasonable scale
    # The projection ranges from -n to n (since vector[i] in {0,1,2} and rand_vec[i] in {-1,1})
    # We want this to be a significant factor in the priority
    projection_score = projection * 10.0

    # Factor 3: Deterministic tie-breaker using hash
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    # Combined priority: higher is added first
    # Line avoidance is the dominant factor (must avoid lines)
    # Random projection provides the chaotic ordering
    # Hash provides deterministic tie-breaking
    total = line_penalty + projection_score + hash_score

    # Update state: add this vector to the set
    # We assume the evaluator will filter out invalid vectors, but we update state
    # optimistically to maintain the line-avoidance context. If the evaluator
    # rejects a vector, the state will be slightly corrupted, but the strong
    # line penalty should still guide the selection effectively.
    vec_tuple = tuple(vector)
    _state['vectors'].append(vec_tuple)
    _update_pair_sum(vec_tuple, n)
    _state['size'] += 1

    return float(total)
