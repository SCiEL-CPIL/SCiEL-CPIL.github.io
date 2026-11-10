import time

_N = 256
_KILL = 90.0
_MARGIN = 8.0

# Precomputation
_weights = [0.0] + [1.0 / i for i in range(1, _N + 1)]

_suffix_sum = [0.0] * (_N + 2)
for i in range(_N, 0, -1):
    _suffix_sum[i] = _suffix_sum[i + 1] + _weights[i]

# Precompute conflict masks for each k
# conflict_masks[k] = list of (mask_u, mask_v) such that {u, v, k} is a 3-AP
_conflict_masks = [[] for _ in range(_N + 1)]
for k in range(1, _N + 1):
    # Case 1: k is the last term. a, b, k.
    # a = k - 2d, b = k - d.
    for d in range(1, (k // 2) + 1):
        a = k - 2 * d
        b = k - d
        if a >= 1:
            _conflict_masks[k].append((1 << (a - 1), 1 << (b - 1)))
    
    # Case 2: k is the first term. k, b, c.
    # b = k + d, c = k + 2d.
    for d in range(1, (_N - k) // 2 + 1):
        b = k + d
        c = k + 2 * d
        if c <= _N:
            _conflict_masks[k].append((1 << (b - 1), 1 << (c - 1)))
            
    # Case 3: k is the middle term. a, k, c.
    # a = k - d, c = k + d.
    max_d = min(k - 1, _N - k)
    for d in range(1, max_d + 1):
        a = k - d
        c = k + d
        _conflict_masks[k].append((1 << (a - 1), 1 << (c - 1)))

def construct():
    start_time = time.time()
    deadline = start_time + _KILL - _MARGIN
    
    best_weight = 0.0
    best_mask = 0
    stop_search = False
    
    import sys
    sys.setrecursionlimit(3000)
    
    def can_add(k, mask):
        for mu, mv in _conflict_masks[k]:
            if (mask & mu) and (mask & mv):
                return False
        return True

    def solve(idx, mask, curr_weight):
        nonlocal best_weight, best_mask, stop_search
        
        if stop_search:
            return
        
        if time.time() > deadline:
            stop_search = True
            return
            
        if idx > _N:
            if curr_weight > best_weight:
                best_weight = curr_weight
                best_mask = mask
            return
            
        # Upper Bound
        ub = curr_weight + _suffix_sum[idx]
        if ub <= best_weight:
            return
            
        # Branch 1: Include idx
        if can_add(idx, mask):
            new_mask = mask | (1 << (idx - 1))
            solve(idx + 1, new_mask, curr_weight + _weights[idx])
            
            if stop_search:
                return
                
        # Branch 2: Exclude idx
        solve(idx + 1, mask, curr_weight)
        
    solve(1, 0, 0.0)
    
    if best_mask == 0:
        # Fallback to greedy
        mask = 0
        for i in range(1, _N + 1):
            if can_add(i, mask):
                mask |= (1 << (i - 1))
        best_mask = mask
        
    sol = [i + 1 for i in range(_N) if (best_mask >> i) & 1]
    
    return sol
