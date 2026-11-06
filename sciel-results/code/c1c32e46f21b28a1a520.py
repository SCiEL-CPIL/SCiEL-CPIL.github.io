"""Constraint-driven constructive search for truncated perfect difference sets."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

_DEADLINE = 145.0
_MARGIN = 5.0

def _diffs(s):
    """Return list of all positive pairwise differences of sorted set s."""
    n = len(s)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            out.append(s[j] - s[i])
    return out

def _valid(s):
    """Check Sidon + covers 1..30 + has 0 + size <= 32 + max <= 10^6."""
    if 0 not in s:
        return False
    n = len(s)
    if n > 32:
        return False
    if n < 2:
        return False
    if max(s) > 10**6:
        return False
    seen = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = s[j] - s[i]
            if d <= 0:
                return False
            if d in seen:
                return False
            seen.add(d)
    for d in range(1, 31):
        if d not in seen:
            return False
    return True

def _energy(s):
    """Composite energy: collision penalty + uncovered penalty + diameter penalty."""
    n = len(s)
    seen = set()
    collisions = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = s[j] - s[i]
            if d <= 0:
                collisions += 10
            elif d in seen:
                collisions += 5
            else:
                seen.add(d)
    uncovered = sum(1 for d in range(1, 31) if d not in seen)
    diameter = max(s) if s else 0
    return 100 * collisions + 20 * uncovered + diameter

def _try_add(s, x):
    """Try to add x to sorted set s. Return new sorted list if valid, else None."""
    if x in s or x <= 0:
        return None
    trial = sorted(s + [x])
    seen = set()
    for i in range(len(trial)):
        for j in range(i + 1, len(trial)):
            d = trial[j] - trial[i]
            if d in seen:
                return None
            seen.add(d)
    return trial

def _greedy_construct(target_D, rng, seed_size=8):
    """Greedy constructive: build set covering 1..30 within target_D."""
    a = [0]
    needed = set(range(1, 31))
    
    # Start with a few random elements to break symmetry
    for _ in range(seed_size - 1):
        x = rng.randint(1, target_D)
        new_a = _try_add(a, x)
        if new_a is not None:
            a = new_a
            # Update needed
            seen = set()
            for i in range(len(a)):
                for j in range(i + 1, len(a)):
                    seen.add(a[j] - a[i])
            needed = needed - seen
            if not needed:
                break
    
    for _ in range(300):  # max additions
        if time.time() > _DEADLINE - _MARGIN:
            break
        if not needed:
            break
        
        # Find best candidate x that covers most needed differences without collision
        best_x = None
        best_gain = -1
        
        # Try candidates in random order
        candidates = list(range(1, target_D + 1))
        rng.shuffle(candidates)
        
        for x in candidates[:100]:
            if x in a:
                continue
            trial = _try_add(a, x)
            if trial is None:
                continue
            # Count new coverage
            seen = set()
            for i in range(len(trial)):
                for j in range(i + 1, len(trial)):
                    seen.add(trial[j] - trial[i])
            new_cov = len(needed & seen)
            if new_cov > best_gain:
                best_gain = new_cov
                best_x = x
                if best_gain >= 3:  # Early exit if good enough
                    break
        
        if best_x is not None and best_gain > 0:
            a = _try_add(a, best_x)
            # Update needed
            seen = set()
            for i in range(len(a)):
                for j in range(i + 1, len(a)):
                    seen.add(a[j] - a[i])
            needed = needed - seen
        else:
            break
    
    return a

def _local_refine(s, rng, deadline, best_max, max_iter=500):
    """Local refinement: try to reduce diameter while maintaining validity."""
    s = list(s)
    cur_max = max(s)
    T = 30.0
    cooling = 0.995
    
    for _ in range(max_iter):
        if time.time() > deadline - _MARGIN:
            break
        
        # Pick a non-zero element to perturb
        idx = rng.randint(1, len(s) - 1)
        old_val = s[idx]
        
        # Perturb: move by random delta, biased toward smaller values
        if rng.random() < 0.7:
            delta = rng.randint(-15, 10)
            new_val = old_val + delta
        else:
            new_val = rng.randint(1, old_val - 1)
        
        if new_val <= 0 or new_val in s:
            continue
        
        s[idx] = new_val
        s.sort()
        new_max = max(s)
        
        # Accept if valid and better, or with some probability
        if _valid(s):
            if new_max < cur_max:
                cur_max = new_max
                if cur_max < best_max:
                    best_max = cur_max
            elif rng.random() < 0.3:  # Accept equal max with some probability
                pass
            else:
                s[idx] = old_val
                s.sort()
        else:
            # Reject invalid moves
            s[idx] = old_val
            s.sort()
        
        T *= cooling
    
    return s, best_max

def construct():
    start = time.time()
    deadline = start + _DEADLINE
    rng = random.Random(42)

    best = None
    best_max = 10**6

    # Phase 1: Multiple randomized greedy constructions with varying D
    for D in [60, 80, 100, 120, 150, 200]:
        if time.time() > deadline - _MARGIN:
            break
        for attempt in range(50):
            if time.time() > deadline - _MARGIN:
                break
            s = _greedy_construct(D, rng)
            if len(s) >= 2 and _valid(s):
                m = max(s)
                if m < best_max:
                    best = list(s)
                    best_max = m
                # Local refinement
                refined, best_max = _local_refine(list(s), rng, deadline, best_max, max_iter=200)
                if _valid(refined):
                    m2 = max(refined)
                    if m2 < best_max:
                        best = list(refined)
                        best_max = m2

    # Phase 2: If we have a good solution, try to improve it further
    if best is not None:
        if time.time() < deadline - _MARGIN:
            # More aggressive local search
            refined, best_max = _local_refine(list(best), rng, deadline, best_max, max_iter=3000)
            if _valid(refined):
                m = max(refined)
                if m < best_max:
                    best = list(refined)
                    best_max = m

        # Element removal
        if time.time() < deadline - _MARGIN:
            s = list(best)
            changed = True
            while changed and time.time() < deadline - _MARGIN:
                changed = False
                for i in range(len(s) - 1, 0, -1):
                    if s[i] == 0:
                        continue
                    trial = [x for x in s if x != s[i]]
                    if len(trial) >= 2 and _valid(trial):
                        s = trial
                        changed = True
                        break
            if _valid(s):
                m = max(s)
                if m < best_max:
                    best = list(s)
                    best_max = m

        # Try to reduce diameter by moving largest element down
        if time.time() < deadline - _MARGIN:
            for _ in range(2000):
                if time.time() > deadline - _MARGIN:
                    break
                s = list(best)
                idx = len(s) - 1  # largest element
                old = s[idx]
                new = rng.randint(1, old - 1)
                if new in set(s):
                    continue
                s[idx] = new
                s.sort()
                if _valid(s):
                    m = max(s)
                    if m < best_max:
                        best = list(s)
                        best_max = m

    # Fallback: known small Sidon set if valid
    if best is None:
        fallback = [0, 1, 3, 4, 9, 11, 12, 17, 19, 20, 25, 27]
        if _valid(fallback):
            return fallback
        # Ultimate fallback: greedy from seed
        a = [0]
        for d in range(1, 31):
            if d in _diffs(sorted(a)):
                continue
            placed = False
            for x in sorted(set(v + dd for v in a for dd in (d, -d) if v + dd >= 0) - set(a)):
                trial = sorted(a + [x])
                if len(set(_diffs(trial))) == len(_diffs(trial)):
                    a = trial
                    placed = True
                    break
            if not placed:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if len(set(_diffs(trial))) == len(_diffs(trial)):
                        a = trial
                        break
                    y += 1
        return a

    return best
# EVOLVE-BLOCK-END
