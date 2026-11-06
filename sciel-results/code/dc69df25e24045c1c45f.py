import time
import random
import numpy as np

N = 128

# Precompute all nonconstant 4-term progressions: (a, a+d, a+2d, a+3d) with 0<=a, a+3d<=127, d>=1
PROGS = []
for d in range(1, N // 3 + 1):
    for a in range(0, N - 3 * d):
        PROGS.append((a, a + d, a + 2 * d, a + 3 * d))
PROGS = np.array(PROGS, dtype=np.int32)

def count_colors(arr):
    return len(set(arr.tolist()))

def valid(arr):
    # arr is 0-indexed length N; check every progression uses >=3 distinct colors
    cols = arr[PROGS]  # shape (M,4)
    # distinct count per row >= 3
    c0, c1, c2, c3 = cols[:, 0], cols[:, 1], cols[:, 2], cols[:, 3]
    distinct = 1
    distinct += (c1 != c0).astype(np.int32)
    distinct += (c2 != c0).astype(np.int32) + (c2 != c1).astype(np.int32)
    distinct += (c3 != c0).astype(np.int32) + (c3 != c1).astype(np.int32) + (c3 != c2).astype(np.int32)
    return bool((distinct >= 3).all())

def greedy_construct(k, rng):
    # Build a valid coloring using at most k colors (labels 0..k-1). Returns array or None.
    arr = np.zeros(N, dtype=np.int32)
    for i in range(N):
        found = False
        order = list(range(k))
        rng.shuffle(order)
        for c in order:
            arr[i] = c
            if valid(arr):
                found = True
                break
        if not found:
            return None
    return arr

def anneal_reduce(arr, k, rng, deadline):
    # Try to reduce color usage below k while staying valid, using simulated annealing.
    # Objective: minimize number of distinct colors used, with validity as hard constraint.
    best = arr.copy()
    best_cnt = count_colors(best)
    cur = arr.copy()
    cur_cnt = count_colors(cur)
    T = 1.0
    T_min = 0.001
    step = 0.999
    while time.time() < deadline and T > T_min:
        # pick a random position and a random color in 0..k-1
        i = rng.randrange(N)
        c = rng.randrange(k)
        if cur[i] == c:
            T *= step
            continue
        old = cur[i]
        cur[i] = c
        if valid(cur):
            new_cnt = count_colors(cur)
            delta = new_cnt - cur_cnt
            if delta <= 0 or rng.random() < np.exp(-delta / T):
                cur_cnt = new_cnt
                if new_cnt < best_cnt:
                    best = cur.copy()
                    best_cnt = new_cnt
            else:
                cur[i] = old
        else:
            # reject invalid moves (hard constraint)
            cur[i] = old
        T *= step
    return best, best_cnt

def construct():
    deadline = time.time() + 80.0  # safety margin under 90s
    rng = random.Random(12345)
    np_rng = np.random.default_rng(12345)

    # Start with a small k and increase until greedy finds a valid coloring.
    base = None
    for k in range(3, 12):
        if time.time() > deadline - 5:
            break
        for attempt in range(5):
            arr = greedy_construct(k, rng)
            if arr is not None:
                base = arr
                break
        if base is not None:
            break
    if base is None:
        # fallback: greedy with many colors (seed-like)
        base = np.zeros(N, dtype=np.int32)
        for i in range(N):
            for c in range(128):
                base[i] = c
                if valid(base):
                    break

    best = base.copy()
    best_cnt = count_colors(best)

    # Try to reduce colors via annealing, starting from current best count.
    start_k = best_cnt
    # anneal with palette size = start_k (we can only use colors present or fewer)
    res, cnt = anneal_reduce(base, start_k, rng, deadline)
    if cnt < best_cnt:
        best = res
        best_cnt = cnt

    # If we still have time, try a few random restarts with smaller palettes
    while time.time() < deadline - 3:
        # try greedy with k = best_cnt - 1 (fewer colors)
        k_try = best_cnt - 1
        if k_try < 3:
            break
        arr = greedy_construct(k_try, rng)
        if arr is not None:
            best = arr
            best_cnt = k_try
            # then anneal to reduce further
            res, cnt = anneal_reduce(arr, k_try, rng, deadline)
            if cnt < best_cnt:
                best = res
                best_cnt = cnt

    # Ensure labels are within 0..127 and return as list of ints
    return [int(x) for x in best.tolist()]
