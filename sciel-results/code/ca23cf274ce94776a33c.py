"""Rewritten program: fast integer-based meet-in-the-middle for reciprocal subset-sum checks."""
import time
import math
import random

# Precompute LCM of 1..24
def _lcm(a, b):
    return a * b // math.gcd(a, b)

_LCM = 1
for i in range(1, 25):
    _LCM = _lcm(_LCM, i)

# v[i] = LCM / i
_V = [0] * 25
for i in range(1, 25):
    _V[i] = _LCM // i

_HALF_SIZE = 12
_LEFT = list(range(1, _HALF_SIZE + 1))   # 1..12
_RIGHT = list(range(_HALF_SIZE + 1, 25)) # 13..24

# Precompute LeftSumsByMask and RightSumsByMask
# LeftSumsByMask[mask] is the set of all subset sums of the elements in mask (from _LEFT)
_LeftSumsByMask = [set() for _ in range(4096)]
for mask in range(4096):
    sub = mask
    while sub:
        s = 0
        for i in range(12):
            if sub >> i & 1:
                s += _V[_LEFT[i]]
        _LeftSumsByMask[mask].add(s)
        sub = (sub - 1) & mask
    _LeftSumsByMask[mask].add(0)

_RightSumsByMask = [set() for _ in range(4096)]
for mask in range(4096):
    sub = mask
    while sub:
        s = 0
        for i in range(12):
            if sub >> i & 1:
                s += _V[_RIGHT[i]]
        _RightSumsByMask[mask].add(s)
        sub = (sub - 1) & mask
    _RightSumsByMask[mask].add(0)

def _check_safe_fast(a_idx, S_mask):
    """Check if element a_idx (0-based) in set S_mask is safe."""
    val = a_idx + 1
    target = _V[val]
    avail_mask = S_mask & ~(1 << a_idx)
    
    if avail_mask == 0:
        return True
    
    left_avail = avail_mask & 0xFFF
    right_avail = (avail_mask >> 12) & 0xFFF
    
    L_sums = _LeftSumsByMask[left_avail]
    R_sums = _RightSumsByMask[right_avail]
    
    # Check if target can be formed by sL + sR
    if len(L_sums) < len(R_sums):
        for sL in L_sums:
            needR = target - sL
            if needR in R_sums:
                if sL != 0 or needR != 0:
                    return False
    else:
        for sR in R_sums:
            needL = target - sR
            if needL in L_sums:
                if sR != 0 or needL != 0:
                    return False
                    
    return True

def construct():
    """Return a nonempty subset of [1,24] maximizing cardinality."""
    start = time.time()
    deadline = start + 80.0
    
    best_set = 0
    best_size = 0
    
    # Greedy construction with random restarts
    for attempt in range(5000):
        if time.time() > deadline:
            break
            
        elems = list(range(24))
        random.shuffle(elems)
        
        S = 0
        for idx in elems:
            trial = S | (1 << idx)
            ok = True
            for i in range(24):
                if trial >> i & 1:
                    if not _check_safe_fast(i, trial):
                        ok = False
                        break
            
            if ok:
                S = trial
        
        if S.bit_count() > best_size:
            best_size = S.bit_count()
            best_set = S

    if best_set != 0:
        result = [i + 1 for i in range(24) if best_set >> i & 1]
    else:
        result = [1]
        
    return result

if __name__ == "__main__":
    t0 = time.time()
    r = construct()
    t1 = time.time()
    print(f"Result: {r}, size={len(r)}, time={t1-t0:.2f}s")
