# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _smooth_numbers(limit):
    s = set()
    a = 1
    while a <= limit:
        b = a
        while b <= limit:
            c = b
            while c <= limit:
                s.add(c)
                c *= 5
            b *= 3
        a *= 2
    return sorted(s)

_SMOOTH = _smooth_numbers(128)
_SMOOTH_SET = set(_SMOOTH)
_N = len(_SMOOTH)
_SMOOTH_IDX = {v: i for i, v in enumerate(_SMOOTH)}

# Precompute incompatibility: for each index, set of incompatible indices
_incompat = [set() for _ in range(_N)]
for i in range(_N):
    for j in range(i + 1, _N):
        a, b = _SMOOTH[i], _SMOOTH[j]
        if a % b == 0 or b % a == 0:
            _incompat[i].add(j)
            _incompat[j].add(i)

def _find_min_antichain(target, deadline):
    """Find minimum number of distinct smooth numbers forming antichain summing to target."""
    # k=1
    if target in _SMOOTH_SET:
        return [target]
    
    # k=2
    if time.time() < deadline:
        for i in range(_N):
            a = _SMOOTH[i]
            if a >= target:
                break
            b = target - a
            if b in _SMOOTH_SET and b != a:
                j = _SMOOTH_IDX[b]
                if j not in _incompat[i]:
                    return [a, b]
    
    # k=3
    if time.time() < deadline:
        # Backtracking for k=3
        for i in range(_N):
            a = _SMOOTH[i]
            if a >= target:
                break
            rem1 = target - a
            if rem1 <= 0:
                continue
            for j in range(i + 1, _N):
                b = _SMOOTH[j]
                if b >= rem1:
                    break
                if j in _incompat[i]:
                    continue
                c = rem1 - b
                if c <= b:
                    continue
                if c in _SMOOTH_SET and c != a and c != b:
                    k_idx = _SMOOTH_IDX[c]
                    if k_idx not in _incompat[i] and k_idx not in _incompat[j]:
                        return [a, b, c]
    
    # k=4+ via backtracking
    if time.time() < deadline:
        found = [None]
        def backtrack(start, remaining, count, chosen):
            if time.time() > deadline:
                return
            if count == 4:
                if remaining == 0:
                    found[0] = list(chosen)
                    return
                return
            if remaining < 0:
                return
            # Prune: need (4-count) numbers summing to remaining, each >= _SMOOTH[start]
            # Simple bound
            for idx in range(start, _N):
                if time.time() > deadline:
                    return
                v = _SMOOTH[idx]
                if v > remaining:
                    break
                # Check compatibility
                compat = True
                for ci in chosen:
                    if idx in _incompat[ci]:
                        compat = False
                        break
                if not compat:
                    continue
                chosen.append(idx)
                backtrack(idx + 1, remaining - v, count + 1, chosen)
                chosen.pop()
                if found[0] is not None:
                    return
        backtrack(0, target, 0, [])
        if found[0] is not None:
            return [_SMOOTH[i] for i in found[0]]
    
    # Fallback: try up to 6 with simple search
    if time.time() < deadline:
        for k in range(4, 7):
            # Simple combination search with pruning
            smooth_leq = [v for v in _SMOOTH if v <= target]
            for terms in combinations(smooth_leq, k):
                if sum(terms) != target:
                    continue
                # Check antichain
                valid = True
                for p in range(len(terms)):
                    for q in range(p+1, len(terms)):
                        if terms[p] % terms[q] == 0 or terms[q] % terms[p] == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    return list(terms)
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    results = []
    for n in range(1, 129):
        if time.time() > deadline:
            # Emergency fallback
            for m in range(n, 129):
                ans = _find_min_antichain(m, deadline)
                if ans is None:
                    ans = [m] if m in _SMOOTH_SET else [1, m-1]
                results.append(ans)
            return results
        
        ans = _find_min_antichain(n, deadline)
        if ans is None:
            # Last resort fallback
            if n in _SMOOTH_SET:
                ans = [n]
            else:
                ans = [1, n-1] if (n-1) in _SMOOTH_SET and n-1 != 1 else [n]
        results.append(ans)
    
    return results
# EVOLVE-BLOCK-END
