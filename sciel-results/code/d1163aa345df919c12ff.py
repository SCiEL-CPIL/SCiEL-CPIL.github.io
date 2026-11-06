"""Integer sum-product extremal sets: minimize max(|A+A|, |A*A|) for 20 distinct ints <= 10000."""
import time
import random
import math
import numpy as np

N = 20
MAXV = 10000
DEADLINE = time.time() + 84.0   # safety margin under the 90 s wall-clock kill

def _objective(vals):
    # vals: np.array of N distinct ints
    s = vals[:, None] + vals[None, :]
    p = vals[:, None] * vals[None, :]
    return int(max(np.unique(s).size, np.unique(p).size))

def _greedy_init(rng, start):
    # Build a set greedily minimizing incremental collision pressure (sum + product).
    used_s = set()
    used_p = set()
    A = []
    cand = list(range(start, MAXV + 1))
    rng.shuffle(cand)
    for c in cand:
        if len(A) >= N:
            break
        # incremental collisions if we add c
        inc = 0
        for a in A:
            if (a + c) in used_s: inc += 2
            if (a * c) in used_p: inc += 2
        # self pair
        if (2 * c) in used_s: inc += 1
        if (c * c) in used_p: inc += 1
        # prefer smaller values (keeps products in range & reduces spread)
        key = (inc, c)
        A.append(c)
        for a in A:
            used_s.add(a + c)
            used_p.add(a * c)
    # ensure exactly N
    while len(A) < N:
        A.append(MAXV - (N - len(A)))
    return np.array(sorted(A[:N]), dtype=np.int64)

def _geometric_seed(rng, base, step):
    # geometric-ish progression, distinct, within range
    A = set()
    x = base
    while len(A) < N and x <= MAXV:
        A.add(int(x))
        x *= step
    A = sorted(A)
    i = 1
    while len(A) < N:
        A.append(A[-1] + i)
        i += 1
    return np.array(A[:N], dtype=np.int64)

def _local_search(A, rng, budget):
    A = A.copy()
    cur = _objective(A)
    best = A.copy()
    best_val = cur
    t0 = time.time()
    # candidate neighbor pool: small offsets around each element + a few randoms
    while time.time() < min(budget, DEADLINE - 2.0):
        i = rng.randrange(N)
        if rng.random() < 0.5:
            delta = rng.choice([-2,-1,1,2,3,5,8,13])
            nv = int(A[i]) + delta
        else:
            nv = rng.randint(1, MAXV)
        if nv < 1 or nv > MAXV:
            continue
        if nv in A:
            continue
        A[i] = nv
        nv_obj = _objective(A)
        if nv_obj < best_val:
            best_val = nv_obj
            best = A.copy()
            cur = nv_obj
        else:
            A[i] = best[i]  # revert (we track best separately)
            # keep A == best for consistency
            A = best.copy()
            cur = best_val
    return best, best_val

def construct():
    t0 = time.time()
    rng = random.Random(12345)
    best = None
    best_val = 10**9

    seeds = []
    # structured seeds
    seeds.append(_greedy_init(rng, 1))
    seeds.append(_geometric_seed(rng, 2, 1.5))
    seeds.append(_geometric_seed(rng, 3, 1.4))
    seeds.append(_geometric_seed(rng, 5, 1.3))
    # a few random
    for _ in range(4):
        seeds.append(np.array(sorted(random.sample(range(1, MAXV + 1), N)), dtype=np.int64))

    for s in seeds:
        if time.time() > DEADLINE - 6.0:
            break
        s = np.array(sorted(set(int(x) for x in s)), dtype=np.int64)
        # fix size if needed
        if len(s) < N:
            extra = [x for x in range(1, MAXV+1) if x not in set(s.tolist())]
            s = np.concatenate([s, np.array(extra[:N-len(s)], dtype=np.int64)])
        s = np.array(sorted(set(int(x) for x in s[:N])), dtype=np.int64)
        v = _objective(s)
        if v < best_val:
            best_val = v
            best = s.copy()

    # local search from the best seed and a couple of other good ones
    order = sorted(range(len(seeds)), key=lambda i: _objective(seeds[i]))
    for idx in order[:4]:
        if time.time() > DEADLINE - 6.0:
            break
        b, bv = _local_search(seeds[idx], rng, DEADLINE - 6.0)
        if bv < best_val:
            best_val = bv
            best = b.copy()

    # final polish: tiny random restarts of local search from best
    while time.time() < DEADLINE - 3.0:
        cand = best.copy()
        for _ in range(3):
            i = rng.randrange(N)
            cand[i] = max(1, min(MAXV, int(cand[i]) + rng.randint(-3, 3)))
        cand = np.array(sorted(set(int(x) for x in cand)), dtype=np.int64)
        if len(cand) < N:
            extra = [x for x in range(1, MAXV+1) if x not in set(cand.tolist())]
            cand = np.concatenate([cand, np.array(extra[:N-len(cand)], dtype=np.int64)])
            cand = np.array(sorted(set(int(x) for x in cand[:N])), dtype=np.int64)
        b, bv = _local_search(cand, rng, DEADLINE - 3.0)
        if bv < best_val:
            best_val = bv
            best = b.copy()

    # safety: return best found
    if best is None:
        return list(range(1, N + 1))
    return [int(x) for x in best]

# EVOLVE-BLOCK-START
# (construct defined above; keep entry point name)
# EVOLVE-BLOCK-END
