import time

P = 27720

def _has_match(r, L):
    # r[i] = (-base) % (i+1) for i in 0..11
    # label i (1..12) -> r index i-1
    match = [-1] * L
    for li in range(12):
        seen = [False] * L
        step = li + 1
        start = r[li]
        # iterative DFS to avoid recursion overhead
        stack = []
        # We'll use a standard augmenting path approach
        def try_augment(li_idx, seen_arr):
            # returns True if we can find an augmenting path for li_idx
            step_val = li_idx + 1
            start_val = r[li_idx]
            for j in range(start_val, L, step_val):
                if seen_arr[j]:
                    continue
                seen_arr[j] = True
                if match[j] == -1:
                    match[j] = li_idx + 1
                    return True
                prev_li = match[j] - 1
                if try_augment(prev_li, seen_arr):
                    match[j] = li_idx + 1
                    return True
            return False
        if not try_augment(li, seen):
            return False
    return True

def _f_for_m(m):
    base = m + 1
    r = [(-base) % (i + 1) for i in range(12)]
    lo, hi = 12, 144
    ans = 144
    while lo <= hi:
        mid = (lo + hi) // 2
        if _has_match(r, mid):
            ans = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return ans

def construct():
    t0 = time.time()
    deadline = t0 + 85.0
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
