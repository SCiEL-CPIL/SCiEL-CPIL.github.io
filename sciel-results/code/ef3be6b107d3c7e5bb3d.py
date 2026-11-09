import time
from math import gcd
from functools import reduce

_DIVS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_MAXV = 1000000
_DEADLINE = 80.0

def _lcm(a, b):
    return a // gcd(a, b) * b

_LCM_ALL = reduce(_lcm, _DIVS)

def _check_feasibility(m, W):
    """
    Check if there exists a distinct assignment of offsets delta_i in [0, W]
    such that (m + delta_i) is divisible by _DIVS[i].
    Returns the list of deltas if found, else None.
    """
    # Precompute candidate offsets for each divisor
    candidates = []
    for d in _DIVS:
        # We need m + delta = 0 mod d
        # delta = 0 mod d - m mod d
        rem = m % d
        # Smallest delta >= 0 such that (m + delta) % d == 0
        # If rem == 0, delta = 0
        # Else delta = d - rem
        delta0 = (d - rem) % d
        
        cands = []
        delta = delta0
        while delta <= W:
            cands.append(delta)
            delta += d
        if not cands:
            return None
        candidates.append(cands)
    
    # Now we need to pick one delta from each candidates[i] such that all are distinct.
    # Backtracking search
    n = len(_DIVS)
    assigned = [False] * (W + 1)
    res = [0] * n
    
    def backtrack(i):
        if i == n:
            return True
        for d in candidates[i]:
            if not assigned[d]:
                assigned[d] = True
                res[i] = d
                if backtrack(i + 1):
                    return True
                assigned[d] = False
        return False
    
    if backtrack(0):
        return res
    return None

def construct():
    t0 = time.time()
    
    # Lower bound for W is 9 (10 distinct integers)
    # Upper bound: we can go up to 1000 to be safe, but it will likely be much smaller.
    for W in range(9, 500):
        if time.time() - t0 > _DEADLINE:
            break
        
        # We only need to check m in [1, LCM_ALL]
        # Because the pattern of residues repeats every LCM_ALL.
        # If a solution exists for m, it exists for m + k*LCM_ALL.
        # We want the smallest values, so we prefer small m.
        # However, the values are m + delta.
        # We just need ANY valid m. The smallest W is the goal.
        # Once we find a valid m for a W, we can construct the values.
        
        found = False
        best_m = None
        best_deltas = None
        
        for m in range(1, _LCM_ALL + 1):
            if time.time() - t0 > _DEADLINE:
                break
            # Optimization: ensure m + W <= MAXV is possible?
            # The values are m + delta. Max value is m + W.
            # We need m + W <= MAXV.
            # Since m can be small, this is usually fine.
            # But if m is close to LCM, m+W might be large.
            # Actually, we can just pick the smallest m that works.
            
            deltas = _check_feasibility(m, W)
            if deltas is not None:
                best_m = m
                best_deltas = deltas
                found = True
                break
        
        if found:
            # Construct the final values
            vals = [best_m + d for d in best_deltas]
            # Verify constraints
            # 1. Distinct: ensured by _check_feasibility
            # 2. Positive: m >= 1, d >= 0, so vals >= 1
            # 3. <= 1e6: We should check this.
            if max(vals) <= _MAXV:
                return [int(x) for x in vals]
            # If max > MAXV, we might need a smaller m?
            # But we scanned m from 1 upwards. So the first m found is the smallest.
            # If even the smallest m leads to max > MAXV, then this W is invalid?
            # No, a larger m would make values even larger.
            # So if smallest m fails MAXV, no m works for this W?
            # Wait, if m is small, vals are small.
            # So if smallest m fails, larger m will definitely fail.
            # Thus, if found is True but max > MAXV, we should continue to next W?
            # Actually, if m=1 works for feasibility but 1+W > MAXV, then no m works for this W
            # because any other m >= 1 will result in max_val >= m+W >= 1+W > MAXV.
            # So we can just break out of the m loop and continue to next W.
            # But my code breaks on first found m.
            # So I should check MAXV before breaking.
            
            # Let's refine: inside the m loop, if deltas found, check max.
            # If max <= MAXV, return.
            # Else, continue searching m? No, larger m is worse for MAXV.
            # So if smallest m fails MAXV, no m works.
            # So we can break the m loop and continue W loop.
            # But my current code breaks and returns if max <= MAXV.
            # If max > MAXV, it falls through.
            # I should handle this explicitly.
            
            # Let's re-structure slightly to be safe.
            pass # The logic above is handled by the return. 
                 # If return is not executed, it means max > MAXV.
                 # In that case, we should not try larger m for this W.
                 # So we should break the m loop.
                 # But the code above breaks immediately after finding deltas.
                 # So it will check max. If fail, it exits m loop (due to break in if found block? No, break is inside if deltas is not None).
                 # Wait, the break is inside `if deltas is not None`.
                 # So it breaks the m loop.
                 # Then it checks `if found`.
                 # Then it constructs vals.
                 # Then it checks `if max(vals) <= _MAXV`.
                 # If true, returns.
                 # If false, it falls through to the next W iteration?
                 # No, the `for W` loop continues.
                 # This is correct.
    
    # Fallback
    return list(_DIVS)
