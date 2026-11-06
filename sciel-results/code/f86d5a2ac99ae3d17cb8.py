# EVOLVE-BLOCK-START
import time

_N = 512
_TARGET = 48
_DEADLINE = 50.0  # safety margin vs 90s kill

def _valid_insert(selected, x):
    # Check that adding x keeps the set valid (x distinct, in range).
    if x in selected:
        return False
    if x < 1 or x > _N:
        return False
    others = [v for v in selected if v != x]
    n = len(others)
    # For each element a in selected, x is "other", so need: no subset of
    # (others) of size >=2 sums to k*a. Also for a == x: no subset of others
    # of size >=2 sums to k*x.
    for a in selected:
        kmax = (_N * 47) // a  # k*a <= 47*512 (max sum of 47 others)
        # Build bitset DP over 'others'
        bits = 1  # bit s set => subset sum s reachable
        for v in others:
            bits |= (bits << v)
        # Now check any k in [2, kmax] with k*a reachable AND achieved by >=2 elements.
        # We also need subset size >= 2. Track min elements to reach each sum? 
        # Simpler: also track that the sum is reachable using at least 2 elements.
        # We'll do a second DP tracking min-count, but cheaper: check reachability
        # of k*a, and if reachable, ensure it's not just a single element equal to k*a.
        # Since all others <= 512 and k*a may be >512, a single element can't reach it
        # unless k*a <= 512. Handle both: compute reachable sums with >=2 elements.
        # Use DP of min number of elements to reach sum s (small ints).
        INF = 10**9
        maxsum = 47 * _N
        # min elements array
        cnt = [INF] * (maxsum + 1)
        cnt[0] = 0
        for v in others:
            for s in range(maxsum, v - 1, -1):
                if cnt[s - v] + 1 < cnt[s]:
                    cnt[s] = cnt[s - v] + 1
        for k in range(2, kmax + 1):
            target = k * a
            if target <= maxsum and cnt[target] >= 2:
                return False
    return True

def _is_valid(selected):
    # Full validity of 'selected' (no element is mean of subset of >=2 others).
    n = len(selected)
    if n < 3:
        return True
    maxsum = (n - 1) * _N
    for a in selected:
        others = [v for v in selected if v != a]
        INF = 10**9
        cnt = [INF] * (maxsum + 1)
        cnt[0] = 0
        for v in others:
            for s in range(maxsum, v - 1, -1):
                if cnt[s - v] + 1 < cnt[s]:
                    cnt[s] = cnt[s - v] + 1
        kmax = maxsum // a
        for k in range(2, kmax + 1):
            target = k * a
            if cnt[target] >= 2:
                return False
    return True

def _greedy_seed():
    selected = []
    for x in range(_N, 0, -1):
        if _valid_insert(selected, x):
            selected.append(x)
            if len(selected) >= _TARGET:
                break
    return selected

def construct():
    t0 = time.time()
    best = _greedy_seed()
    if len(best) >= _TARGET:
        return best
    best_set = set(best)
    # Local search: try to increase size via single insertions, and improve via swaps.
    # Keep best by (len, -max) so we prefer larger then smaller max (headroom).
    while time.time() - t0 < _DEADLINE:
        improved = False
        cur = best
        cur_set = best_set
        # 1) Try single insertions (any x not in cur)
        for x in range(_N, 0, -1):
            if x in cur_set:
                continue
            if _valid_insert(cur, x):
                cand = cur + [x]
                # prefer replacing to keep max lower? Just accept if improves objective
                if len(cand) > len(cur):
                    cur = cand
                    cur_set = set(cand)
                    improved = True
                    break
        if improved:
            best = cur
            best_set = cur_set
            if len(best) >= _TARGET:
                break
            continue
        # 2) Try swaps: remove one, add one (keep size, lower max for headroom)
        swapped = False
        cur_max = max(cur)
        for i in range(len(cur)):
            removed = cur[i]
            rest = cur[:i] + cur[i+1:]
            for x in range(removed - 1, 0, -1):
                if x in cur_set:
                    continue
                if x >= cur_max:
                    continue
                if _valid_insert(rest, x):
                    cand = rest + [x]
                    cur = cand
                    cur_set = set(cand)
                    swapped = True
                    break
            if swapped:
                break
        if swapped:
            best = cur
            best_set = cur_set
            if len(best) >= _TARGET:
                break
            continue
        # No improving move found; stop.
        break
    return best
# EVOLVE-BLOCK-END
