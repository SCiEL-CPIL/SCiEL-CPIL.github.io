import time
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _compute_lcm_range(n):
    result = 1
    for i in range(1, n + 1):
        result = _lcm(result, i)
    return result

def _precompute_masks_for_residue(r, n=12, max_L=144):
    """For residue r, precompute for each i in 1..n the bitmask of positions 1..max_L where (r+j)%i==0."""
    masks = [0] * (n + 1)
    for i in range(1, n + 1):
        mask = 0
        for j in range(1, max_L + 1):
            if (r + j) % i == 0:
                mask |= (1 << (j - 1))
        masks[i] = mask
    return masks

def _has_perfect_matching_masks(masks, L, n=12):
    """Check perfect matching using precomputed masks, considering only first L positions."""
    trunc_mask = (1 << L) - 1
    adj = [masks[i] & trunc_mask for i in range(n + 1)]
    
    for i in range(1, n + 1):
        if adj[i] == 0:
            return False
    
    match = [-1] * L
    
    def dfs(i, visited):
        mm = adj[i]
        while mm:
            low = mm & (-mm)
            j = low.bit_length() - 1
            mm ^= low
            if visited[j]:
                continue
            visited[j] = True
            if match[j] == -1 or dfs(match[j], visited):
                match[j] = i
                return True
        return False
    
    for i in range(1, n + 1):
        visited = [False] * L
        if not dfs(i, visited):
            return False
    return True

def _compute_f_from_masks(masks, n=12, max_L=144):
    """Binary search on L using precomputed masks."""
    lo, hi = 1, max_L
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_perfect_matching_masks(masks, mid, n):
            hi = mid
        else:
            lo = mid + 1
    return lo

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85

_n_pre = 12
_period_pre = _compute_lcm_range(_n_pre)
_max_L_pre = 144

_best_f_pre = 0
_best_residue_pre = 0

for _r in range(_period_pre):
    if time.time() > _DEADLINE:
        break
    _masks = _precompute_masks_for_residue(_r, _n_pre, _max_L_pre)
    _f_val = _compute_f_from_masks(_masks, _n_pre, _max_L_pre)
    if _f_val > _best_f_pre:
        _best_f_pre = _f_val
        _best_residue_pre = _r
        if _best_f_pre == _max_L_pre:
            break  # Cannot do better than 144

_PRECOMPUTED_M = int(_best_residue_pre) if 0 <= _best_residue_pre <= 1000000 else 0

def construct():
    return _PRECOMPUTED_M
