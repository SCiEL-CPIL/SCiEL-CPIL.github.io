import time

P = 27720

def _has_match(base, L, r):
    # base = m+1, r[i] = (-base) % (i+1) for i in 0..11
    # label i (1..12) -> r index i-1
    match = [-1] * L
    # build cand per label on the fly
    def dfs(li, seen):
        step = li + 1
        start = r[li]
        for j in range(start, L, step):
            if seen[j]:
                continue
            seen[j] = True
            if match[j] == -1 or dfs(match[j] - 1, seen):
                match[j] = li + 1
                return True
        return False
    for li in range(12):
        seen = [False] * L
        if not dfs(li, seen):
            return False
    return True

def _f_for_m(m):
    base = m + 1
    r = [(-base) % (i + 1) for i in range(12)]
    lo, hi = 12, 144
    # binary search smallest L with match
    ans = 144
    while lo <= hi:
        mid = (lo + hi) // 2
        if _has_match(base, mid, r):
            ans = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return ans

def construct():
    t0 = time.time()
    deadline = t0 + 70.0
    best_m = 12
    best_f = _f_for_m(12)
    for m in range(0, P):
        if time.time() > deadline:
            break
        f = _f_for_m(m)
        if f > best_f:
            best_f = f
            best_m = m
    return int(best_m)
