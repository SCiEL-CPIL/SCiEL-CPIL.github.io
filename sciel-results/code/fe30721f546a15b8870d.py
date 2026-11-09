import random
import time

N = 256
DEADLINE = 55.0  # safety margin under 60s

def _is_sidon(s):
    s = sorted(s)
    seen = set()
    for i in range(len(s)):
        for j in range(i, len(s)):
            v = s[i] + s[j]
            if v in seen:
                return False
            seen.add(v)
    return True

def _is_maximal(s, N=N):
    # For each missing x, check if s ∪ {x} is still Sidon. If any is, not maximal.
    sset = set(s)
    # Precompute existing sums
    sums = set()
    ok = True
    for i, a in enumerate(s):
        for b in s[i:]:
            sums.add(a + b)
    for x in range(1, N + 1):
        if x in sset:
            continue
        # new sums: x + a for a in s, and 2x
        for a in s:
            if x + a in sums:
                break
        else:
            if 2 * x not in sums:
                return False  # x can be added, not maximal
    return True

def _greedy_random(order):
    selected = []
    sums = set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def _local_search(s, deadline):
    # Removals
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(len(s)):
            cand = s[:i] + s[i+1:]
            if _is_maximal(cand):
                s = cand
                improved = True
                break
    # Swaps: remove one, add one missing
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        missing = [x for x in range(1, N + 1) if x not in set(s)]
        random.shuffle(missing)
        for i in range(len(s)):
            base = s[:i] + s[i+1:]
            for x in missing:
                cand = base + [x]
                if _is_sidon(cand) and _is_maximal(cand):
                    s = cand
                    improved = True
                    break
            if improved:
                break
    return s

def _refine(s, deadline):
    s = _local_search(s, deadline)
    # final removal pass
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(len(s)):
            cand = s[:i] + s[i+1:]
            if _is_maximal(cand):
                s = cand
                improved = True
                break
    return s

def construct():
    start = time.time()
    deadline = start + DEADLINE
    best = None
    best_len = 10**9

    rng = random.Random(12345)
    # Initial: greedy ascending (known valid)
    base = _greedy_random(list(range(1, N + 1)))
    base = _refine(base, deadline)
    if _is_sidon(base) and _is_maximal(base):
        best = base
        best_len = len(base)

    # Randomized restarts
    for trial in range(2000):
        if time.time() > deadline:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        cand = _greedy_random(order)
        if len(cand) >= best_len:
            continue
        cand = _refine(cand, deadline)
        if _is_sidon(cand) and _is_maximal(cand) and len(cand) < best_len:
            best = cand
            best_len = len(cand)
            if best_len <= 11:
                break

    if best is None:
        # fallback
        best = _greedy_random(list(range(1, N + 1)))
    return best
