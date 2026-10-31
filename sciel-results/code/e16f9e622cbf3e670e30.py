import time

_start_time = None
_DEADLINE = 330 * 0.95

_state = {
    'n': None,
    'vectors': [],
    'pair_sum': {},
    'size': 0,
}

def _ensure_init(n):
    if _state['n'] != n:
        _state['n'] = n
        _state['vectors'] = []
        _state['pair_sum'] = {}
        _state['size'] = 0

def _update_pair_sum(vec, n):
    ps = _state['pair_sum']
    size = _state['size']
    for j in range(size):
        vj = _state['vectors'][j]
        key = tuple((vec[i] + vj[i]) % 3 for i in range(n))
        ps[key] = ps.get(key, 0) + 1
        # Note: key_rev is same as key since addition is commutative.
        # The current code adds it twice?
        # ps[key] += 1
        # ps[key_rev] += 1
        # If key == key_rev, it adds 2.
        # a+b = b+a. So key is always equal to key_rev.
        # So ps[key] is incremented by 2 for each pair?
        # The current code does:
        # ps[key] = ps.get(key, 0) + 1
        # ps[key_rev] = ps.get(key_rev, 0) + 1
        # Since key == key_rev, this is effectively ps[key] += 2.
        # This is correct if we count ordered pairs?
        # The line condition is a+b+v=0. a,b distinct.
        # Number of pairs (a,b) is |S|(|S|-1).
        # So counting ordered pairs is fine.

def _fast_hash(vector):
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    return h / 0x7FFFFFFF

def _min_hamming_distance(vector):
    min_d = 10**9
    for vj in _state['vectors']:
        d = sum(1 for i in range(_state['n']) if vector[i] != vj[i])
        if d < min_d:
            min_d = d
    return min_d

def priority(vector, n):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    _ensure_init(n)

    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    # 1. Balance
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 10.0

    # 2. Sum
    coord_sum = c1 + 2 * c2
    sum_mod = coord_sum % 3
    if sum_mod == 0:
        sum_score = -1000.0
    else:
        sum_score = 500.0

    # 3. Line Avoidance (Pair Sum)
    ps = _state['pair_sum']
    if ps:
        v_key = tuple((3 - vector[i]) % 3 for i in range(n))
        line_count = ps.get(v_key, 0)
        line_penalty = -line_count * line_count * 50.0 - line_count * 100.0
    else:
        line_penalty = 0.0

    # 4. Hamming Distance Penalty
    if _state['size'] > 0:
        min_d = _min_hamming_distance(vector)
        # Strong penalty for small distances
        if min_d == 1:
            dist_penalty = -5000.0
        elif min_d == 2:
            dist_penalty = -1000.0
        else:
            dist_penalty = 0.0
    else:
        dist_penalty = 0.0

    # 5. Hash
    hash_score = _fast_hash(vector)

    total = balance_score + sum_score + line_penalty + dist_penalty + hash_score

    # Update State
    vec_tuple = tuple(vector)
    _state['vectors'].append(vec_tuple)
    _update_pair_sum(vec_tuple, n)
    _state['size'] += 1

    return float(total)
