"""Backtracking with forward checking for truncated perfect difference sets."""
# EVOLVE-BLOCK-START
import time
import random
import sys

_DEADLINE = 145.0
_MARGIN = 5.0

def _valid(s):
    """Check Sidon + covers 1..30 + has 0 + size <= 32 + max <= 10^6."""
    if 0 not in s:
        return False
    n = len(s)
    if n > 32 or n < 2:
        return False
    if max(s) > 10**6:
        return False
    seen = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = s[j] - s[i]
            if d in seen:
                return False
            seen.add(d)
    for d in range(1, 31):
        if d not in seen:
            return False
    return True

def _backtrack_search(max_elem, deadline, rng, min_size=12, max_size=32):
    """
    Backtracking search for a Sidon set containing 0, all diffs 1..30,
    with max element <= max_elem.
    
    We build the set incrementally. At each step we decide which element to add next.
    The key insight: we need differences 1..30 to appear. We can think of this as
    placing elements such that their pairwise differences cover 1..30.
    
    Strategy: Use a recursive backtracking where we maintain:
    - current set S (sorted, contains 0)
    - used_diffs: set of differences already present
    - need_diffs: set of differences still needed (subset of 1..30)
    
    We try adding elements in a smart order.
    """
    best = [None]
    
    def _search(S, used_diffs, need_diffs, depth):
        if time.time() > deadline - _MARGIN:
            return False
        
        # Check if we've covered all needed differences
        if not need_diffs:
            # We have a valid solution
            if best[0] is None or max(S) < max(best[0]):
                best[0] = list(S)
            return True
        
        # If we've hit max size, we can't add more
        if len(S) >= max_size:
            return False
        
        # Pruning: if the remaining elements can't cover the needed diffs
        # This is a rough bound - skip for now
        
        # Determine which differences we still need
        # Try to add a new element x
        # x should be > S[-1] (we build in increasing order)
        # x must create new differences that are in need_diffs or not conflicting
        
        last = S[-1] if S else 0
        # Try values from last+1 up to max_elem
        # But we can be smarter: focus on values that create needed differences
        
        # Collect candidate x values
        candidates = set()
        for d in need_diffs:
            # If we add x = s + d for some s in S, we get difference d
            for s in S:
                x = s + d
                if x > last and x <= max_elem:
                    candidates.add(x)
                # Also x = s - d but that would be < s, skip since we build increasing
            # x = d (if 0 is in S, which it is)
            if d > last and d <= max_elem:
                candidates.add(d)
        
        # Also try some random candidates for diversity
        if rng.random() < 0.3:
            for _ in range(5):
                x = rng.randint(last + 1, max_elem)
                candidates.add(x)
        
        # Sort candidates: prefer those that cover more needed differences
        def _score_candidate(x):
            if x in S:
                return -1
            new_diffs = set()
            for s in S:
                d = abs(x - s)
                if d not in used_diffs:
                    new_diffs.add(d)
            # Score: number of needed diffs this covers
            return len(new_diffs & need_diffs)
        
        candidates = sorted(candidates, key=_score_candidate, reverse=True)
        
        for x in candidates:
            if time.time() > deadline - _MARGIN:
                return False
            
            if x in S:
                continue
            if x <= last:
                continue
            if x > max_elem:
                continue
            
            # Check if adding x creates any duplicate differences
            valid_add = True
            new_diffs = set()
            for s in S:
                d = x - s
                if d in used_diffs:
                    valid_add = False
                    break
                new_diffs.add(d)
            
            if not valid_add:
                continue
            
            # Update state
            S.append(x)
            new_used = used_diffs | new_diffs
            new_need = need_diffs - new_diffs
            
            if _search(S, new_used, new_need, depth + 1):
                S.pop()
                return True
            
            S.pop()
        
        return False
    
    S = [0]
    used_diffs = set()
    need_diffs = set(range(1, 31))
    
    _search(S, used_diffs, need_diffs, 0)
    
    return best[0]

def _local_shrink(s, rng, deadline, max_elem):
    """Try to reduce the maximum element via local moves."""
    s = sorted(set(s))
    if not s or 0 not in s:
        return s, max(s) if s else 0
    best_mx = max(s)
    
    for _ in range(3000):
        if time.time() > deadline - _MARGIN:
            break
        
        idx = rng.randint(1, len(s) - 1)
        old = s[idx]
        
        # Try to move this element to a smaller position
        new = rng.randint(1, min(old - 1, max_elem))
        if new in s:
            continue
        
        s[idx] = new
        s = sorted(set(s))
        
        if _valid(s) and max(s) < best_mx:
            best_mx = max(s)
        else:
            s[idx] = old
            s = sorted(set(s))
    
    return s, best_mx

def _greedy_construct():
    """Simple greedy construction as fallback."""
    a = [0]
    used_diffs = set()
    
    for d in range(1, 31):
        placed = False
        for x in range(1, 200):
            if x in a:
                continue
            # Check if adding x keeps Sidon property
            valid = True
            for s in a:
                dd = abs(x - s)
                if dd in used_diffs:
                    valid = False
                    break
            if valid:
                a.append(x)
                a.sort()
                for s in a[:-1]:
                    used_diffs.add(x - s)
                placed = True
                break
        if not placed:
            # Try adding two elements
            for y in range(1, 500):
                for z in range(y + 1, y + 100):
                    if y in a or z in a:
                        continue
                    valid = True
                    for s in a:
                        for t in [y, z]:
                            dd = abs(t - s)
                            if dd in used_diffs:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                    # Also check y-z
                    if abs(y - z) in used_diffs:
                        continue
                    a.extend([y, z])
                    a.sort()
                    for i in range(len(a)):
                        for j in range(i + 1, len(a)):
                            used_diffs.add(a[j] - a[i])
                    placed = True
                    break
                if placed:
                    break
    
    return a

def construct():
    start = time.time()
    deadline = start + _DEADLINE
    rng = random.Random(42)
    
    best = None
    best_max = 10**6
    
    # First, get a valid solution via greedy
    greedy_sol = _greedy_construct()
    if _valid(greedy_sol):
        best = list(greedy_sol)
        best_max = max(greedy_sol)
    
    # Try backtracking with various max_elem bounds
    # Start from a reasonable bound and go down
    if best is not None:
        # Try to improve via backtracking with tighter bounds
        for max_elem in range(max(best_max - 10, 35), 30, -1):
            if time.time() > deadline - _MARGIN:
                break
            
            sol = _backtrack_search(max_elem, deadline, rng)
            if sol is not None and _valid(sol):
                if max(sol) < best_max:
                    best = list(sol)
                    best_max = max(sol)
            
            # If we found a solution at this bound, we can stop
            # (we're going downward, so first found is best)
            if sol is not None:
                break
    
    # If no solution found yet, try wider backtracking
    if best is None:
        for max_elem in [100, 80, 60, 50]:
            if time.time() > deadline - _MARGIN:
                break
            sol = _backtrack_search(max_elem, deadline, rng)
            if sol is not None and _valid(sol):
                best = list(sol)
                best_max = max(sol)
                break
    
    # Local shrink to try to reduce further
    if best is not None and time.time() < deadline - _MARGIN:
        for _ in range(10):
            if time.time() > deadline - _MARGIN:
                break
            shrunken, mx = _local_shrink(list(best), rng, deadline, best_max - 1)
            if _valid(shrunken) and mx < best_max:
                best = list(shrunken)
                best_max = mx
    
    # Try more aggressive backtracking with smaller bounds if we have time
    if best is not None and time.time() < deadline - _MARGIN:
        for max_elem in range(max(best_max - 1, 30), 29, -1):
            if time.time() > deadline - _MARGIN:
                break
            sol = _backtrack_search(max_elem, deadline, rng)
            if sol is not None and _valid(sol):
                if max(sol) < best_max:
                    best = list(sol)
                    best_max = max(sol)
                break
    
    if best is None:
        # Ultimate fallback
        return [0, 1, 4, 10, 12, 17, 20, 23, 25, 29]
    
    return best
# EVOLVE-BLOCK-END
