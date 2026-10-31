"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 142  # safety margin before 150s kill

    def compute_score(U):
        U = np.array(sorted(set(U)), dtype=np.int64)
        if U.size == 0 or U[0] != 0:
            return -1.0
        max_u = int(U[-1])
        if max_u == 0:
            return 0.0
        s = np.unique(U[:, None] + U[None, :]).size
        d = np.unique(U[:, None] - U[None, :]).size
        if s == 0:
            return -1.0
        base = 2 * max_u + 1
        if base <= 1:
            return 0.0
        ratio = d / s
        if ratio <= 1:
            return 1.0
        return 1.0 + math.log(ratio) / math.log(base)

    best_score = compute_score([0, 1, 3])
    best_set = [0, 1, 3]

    def try_set(cand):
        nonlocal best_score, best_set
        if not cand or cand[0] != 0:
            return
        if cand[-1] == 0:
            return
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
        return s

    def greedy_build(seed, max_steps=400):
        """Add the element that most improves the score, iteratively."""
        cur = list(seed)
        cur_score = compute_score(cur)
        max_u = max(cur)
        for _ in range(max_steps):
            if time.time() > deadline - 4:
                break
            # candidate pool: nearby + spread + beyond current max
            pool = set()
            for x in cur:
                for dx in range(-3, 4):
                    if x + dx > 0:
                        pool.add(x + dx)
            pool.update(range(1, max_u + 1))
            pool.update(range(max_u + 1, max_u + 40))
            best_el, best_s = None, cur_score
            for el in pool:
                if el in cur:
                    continue
                cand = sorted(cur + [el])
                s = compute_score(cand)
                if s > best_s + 1e-12:
                    best_s, best_el = s, el
            if best_el is None:
                break
            cur = sorted(cur + [best_el])
            cur_score = best_s
            max_u = max(cur)
        return cur, cur_score

    # ---- Phase A: diverse greedy multi-starts ----
    random.seed(2024)
    starts = []
    # geometric seeds
    for r in [2, 3, 4, 5, 6]:
        g = [0, 1]
        while g[-1] * r <= 2000:
            g.append(g[-1] * r)
        starts.append(g[:len(g)])
    # small dense + a spread point
    for a in [2, 3, 4, 5, 6]:
        for b in [20, 50, 100, 200, 500, 1000]:
            starts.append(sorted(list(range(0, a + 1)) + [b]))
    # small random seeds
    for _ in range(30):
        n = random.randint(3, 7)
        mx = random.randint(10, 300)
        inner = random.sample(range(1, mx), min(n - 1, mx - 1))
        starts.append(sorted([0] + inner + [mx]))
    # arithmetic
    for d in [2, 3, 5, 7]:
        starts.append(list(range(0, 300, d)))

    for seed in starts:
        if time.time() > deadline - 20:
            break
        cur, cs = greedy_build(seed)
        try_set(cur)

    # ---- Phase B: simulated annealing from the best + a few good starts ----
    def anneal(start, steps, t0, t1):
        cur = list(start)
        cur_score = compute_score(cur)
        best_local = list(cur)
        best_local_score = cur_score
        n = len(cur)
        for it in range(steps):
            if time.time() > deadline - 5:
                break
            t = t0 * (t1 / t0) ** (it / max(1, steps - 1))
            r = random.random()
            cand = cur
            if n > 2 and r < 0.35:
                # remove
                idx = random.randint(1, n - 1)
                cand = [x for j, x in enumerate(cur) if j != idx]
            elif r < 0.70:
                # swap
                idx = random.randint(1, n - 1)
                new_el = random.randint(1, max(cur) + 30)
                if new_el not in cur:
                    cand = list(cur)
                    cand[idx] = new_el
                    cand = sorted(set(cand))
                    if cand[0] != 0:
                        continue
            else:
                # add
                new_el = random.randint(1, max(cur) + 30)
                if new_el not in cur:
                    cand = sorted(cur + [new_el])
            if len(cand) < 2:
                continue
            s = compute_score(cand)
            if s >= cur_score or random.random() < math.exp((s - cur_score) / max(t, 1e-6)):
                cur = cand
                cur_score = s
                if cur_score > best_local_score:
                    best_local_score = cur_score
                    best_local = list(cur)
        return best_local, best_local_score

    # bank of good starting points
    bank = [list(best_set)]
    random.seed(7)
    for _ in range(8):
        n = random.randint(3, 8)
        mx = random.randint(10, 500)
        inner = random.sample(range(1, mx), min(n - 1, mx - 1))
        bank.append(sorted([0] + inner + [mx]))

    for st in bank:
        if time.time() > deadline - 6:
            break
        bl, bs = anneal(st, steps=1500, t0=0.05, t1=0.002)
        try_set(bl)

    # ---- Phase C: final greedy polish on the global best ----
    if time.time() < deadline - 3:
        cur, cs = greedy_build(list(best_set), max_steps=200)
        try_set(cur)

    return list(best_set)
# EVOLVE-BLOCK-END
