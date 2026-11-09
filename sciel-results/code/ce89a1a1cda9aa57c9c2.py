"""Precompute optimal antichain representations for all targets 1-128."""
import sys
from functools import lru_cache

def _is_smooth(x):
    for p in (2, 3, 5):
        while x % p == 0:
            x //= p
    return x == 1

def _build_smooth_list(limit):
    return [x for x in range(1, limit + 1) if _is_smooth(x)]

def _check_antichain(terms):
    """Check that no term divides another in the list."""
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            a, b = terms[i], terms[j]
            if a < b and b % a == 0:
                return False
            if b < a and a % b == 0:
                return False
    return True

def _find_min_rep(n, smooth):
    """Find minimum-cardinality antichain representation for target n."""
    # Try from 1 term up to 30
    for count in range(1, 31):
        # Use DFS with pruning
        candidates = [x for x in smooth if x <= n]
        if not candidates:
            continue
        
        # Sort descending for better pruning
        candidates.sort(reverse=True)
        
        found = None
        
        def dfs(start, remaining, chosen):
            nonlocal found
            if found is not None:
                return
            if remaining == 0:
                if len(chosen) == count:
                    found = list(chosen)
                return
            if len(chosen) == count:
                return
            # Pruning: if remaining < 0, return
            if remaining < 0:
                return
            # Pruning: not enough elements left
            if start + (count - len(chosen)) > len(candidates):
                return
            # Pruning: even taking all remaining largest can't reach
            max_possible = sum(candidates[start:start + (count - len(chosen))])
            if max_possible < remaining:
                return
            
            for i in range(start, len(candidates)):
                val = candidates[i]
                if val > remaining:
                    continue
                # Check antichain with existing chosen
                valid = True
                for c in chosen:
                    if c < val and val % c == 0:
                        valid = False
                        break
                    if val < c and c % val == 0:
                        valid = False
                        break
                if not valid:
                    continue
                chosen.append(val)
                dfs(i + 1, remaining - val, chosen)
                chosen.pop()
                if found is not None:
                    return
        
        dfs(0, n, [])
        if found is not None:
            return found
    return None

def _precompute():
    smooth = _build_smooth_list(128)
    results = [None] * 129  # index 1..128
    for n in range(1, 129):
        rep = _find_min_rep(n, smooth)
        if rep is None:
            # Fallback: try harder with more terms
            rep = _find_min_rep(n, smooth)
        results[n] = rep
    return results

_PRECOMPUTED = None

def construct():
    global _PRECOMPUTED
    if _PRECOMPUTED is None:
        _PRECOMPUTED = _precompute()
    return [_PRECOMPUTED[n] for n in range(1, 129)]
