"""Improved program: fast scoring + base-B candidate family + stronger local search."""
import time
import random
import numpy as np
from math import log

# EVOLVE-BLOCK-START

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1.0
    n = len(U)
    if n < 2:
        return -1.0
    a = np.array(U, dtype=np.int64)
    s = np.add.outer(a, a)
    d = np.subtract.outer(a, a)
    ns = len(np.unique(s))
    nd = len(np.unique(d))
    if ns <= 0 or nd <= 0:
        return -1.0
    ratio = nd / ns
    if ratio <= 1:
        return -1.0
    D = 2 * int(U[-1]) + 1
    if D <= 1:
        return -1.0
    return 1.0 + log(ratio) / log(D)

def generate_candidates(deadline):
    cands = []
    def add(U):
        if time.time() > deadline - 15:
            return
        U = sorted(set(int(x) for x in U))
        if 0 not in U or len(U) < 2:
            return
        cands.append(U)

    # 1. Pure base-B (binary-like) sets  {0, 1, B, B^2, ...}
    for B in range(2, 40):
        for K in range(3, 16):
            U = [0]
            v = 1
            for _ in range(1, K):
                U.append(v)
                v *= B
            add(U)

    # 2. Base-B with a leading coefficient a (a*0, a, aB, ...)
    for B in range(2, 12):
        for K in range(3, 10):
            for a in range(1, 4):
                U = [0]
                v = a
                for _ in range(1, K):
                    U.append(v)
                    v *= B
                add(U)

    # 3. Mixed base: digit d_k in [1, B_k]
    random.seed(1234)
    for _ in range(250):
        K = random.randint(4, 10)
        U = [0]
        v = 1
        for _ in range(K - 1):
            B = random.randint(2, 7)
            d = random.randint(1, B)
            U.append(d * v)
            v *= B
        add(U)

    # 4. Randomized geometric (smooth base)
    for _ in range(250):
        B = random.uniform(1.3, 6.0)
        K = random.randint(4, 14)
        U = [0]
        v = 1.0
        for _ in range(1, K):
            U.append(int(round(v)))
            v *= B
        add(U)

    # 5. Fibonacci-like
    for K in range(3, 12):
        U = [0, 1]
        while len(U) < K:
            U.append(U[-1] + U[-2])
        add(U)

    return cands

def local_search(start_set, deadline, max_iter=3000):
    current = sorted(set(int(x) for x in start_set))
    if 0 not in current:
        current = sorted(set(current + [0]))
    current_score = score_set(current)
    if current_score < 0:
        return current, -1.0
    best_set = current[:]
    best_score = current_score

    rng = random.Random(99)

    def try_move(new_set, allow_worse=False):
        nonlocal current, current_score, best_set, best_score
        new_set = sorted(set(int(x) for x in new_set))
        if 0 not in new_set or len(new_set) < 2:
            return False
        new_score = score_set(new_set)
        if new_score <= -0.5:
            return False
        if new_score > current_score:
            current = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = current[:]
            return True
        if allow_worse:
            # smooth surrogate: accept small expected worsening
            if new_score > current_score - 0.005:
                current = new_set
                current_score = new_score
                return True
        return False

    for _ in range(max_iter):
        if time.time() > deadline - 8:
            break
        moved = False
        cur = current
        m = len(cur)

        # Move 1: shift one element (small or proportional)
        if m > 1:
            i = rng.randint(1, m - 1)
            old = cur[i]
            if old > 1:
                deltas = rng.sample([1, 2, 3, 5, -1, -2, -3, -5,
                                     max(1, old // 2), -max(1, old // 2),
                                     old * 2, old + rng.randint(1, 10)], 4)
            else:
                deltas = [1, 2, 3, 4, 5]
            for delta in deltas:
                nv = old + delta
                if nv > 0:
                    ns = cur[:]
                    ns[i] = nv
                    if try_move(ns):
                        moved = True
                        break
        if moved:
            continue

        # Move 2: add a new element
        if m < 22:
            mx = cur[-1]
            pool = [mx + rng.randint(1, 10), mx * 2, mx + 1, mx + 2]
            for x in rng.sample(cur, min(4, m)):
                pool.extend([x + 1, x - 1, x * 2, x + rng.randint(1, 5)])
            pool.append(rng.randint(1, mx))
            pool.append(mx + rng.randint(5, 30))
            for nv in rng.sample([p for p in pool if p > 0 and p not in cur],
                                 min(6, len([p for p in pool if p > 0 and p not in cur]))):
                if try_move(cur + [nv]):
                    moved = True
                    break
        if moved:
            continue

        # Move 3: remove an element
        if m > 4:
            i = rng.randint(1, m - 1)
            ns = [x for j, x in enumerate(cur) if j != i]
            if try_move(ns):
                moved = True
                continue

        # Move 4: smooth-surrogate step (accept small worsening)
        if rng.random() < 0.15 and m > 1:
            i = rng.randint(1, m - 1)
            nv = cur[i] + rng.randint(-4, 4)
            if nv > 0:
                ns = cur[:]
                ns[i] = nv
                try_move(ns, allow_worse=True)

    return best_set, best_score

def construct():
    deadline = time.time() + 140  # safety margin under 150s wall-clock

    candidates = generate_candidates(deadline)

    # Fallback
    best_set = [0, 1, 2, 4, 8, 16]
    best_score = score_set(best_set)

    scored = []
    for c in candidates:
        s = score_set(c)
        if s > best_score:
            best_score = s
            best_set = c[:]
        if s > 0:
            scored.append((s, c))

    scored.sort(key=lambda x: x[0], reverse=True)

    seen = set()
    top_starts = []
    for s, c in scored:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            top_starts.append(c)
        if len(top_starts) >= 8:
            break
    if tuple(best_set) not in [tuple(x) for x in top_starts]:
        top_starts.append(best_set)

    for start in top_starts:
        if time.time() > deadline - 10:
            break
        lb, ls = local_search(start, deadline, max_iter=600)
        if ls > best_score:
            best_score = ls
            best_set = lb[:]

    # Final polish pass on the current best with tighter budget
    if time.time() < deadline - 12:
        lb, ls = local_search(best_set, deadline, max_iter=400)
        if ls > best_score:
            best_score = ls
            best_set = lb[:]

    return best_set

# EVOLVE-BLOCK-END
