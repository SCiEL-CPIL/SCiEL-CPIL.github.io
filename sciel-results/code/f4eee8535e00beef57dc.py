import time
import random
import numpy as np
from math import lcm

def construct():
    start = time.time()
    deadline = start + 85.0

    def ok():
        return time.time() < deadline

    denominators = list(range(2, 49))
    n = len(denominators)
    recip = [1.0 / d for d in denominators]

    # exact integer weights: sum(weights) < L  <=>  reciprocal sum < 1
    L = 1
    for d in denominators:
        L = lcm(L, d)
    weights = [L // d for d in denominators]

    def exact_sum(idx_list):
        return sum(weights[i] for i in idx_list)

    # ---- Meet-in-the-middle ----
    mid = n // 2
    left = list(range(0, mid))
    right = list(range(mid, n))
    kl = len(left)
    kr = len(right)

    def gen_sums(items):
        k = len(items)
        nmask = 1 << k
        vals = np.array([recip[i] for i in items], dtype=np.float64)
        sums = np.zeros(nmask, dtype=np.float64)
        for i in range(k):
            period = 1 << (i + 1)
            half = 1 << i
            sums[half::period] += vals[i]
        return sums

    sumsL = gen_sums(left)
    if not ok():
        return [0]
    sumsR = gen_sums(right)
    if not ok():
        return [0]

    # keep only right sums < 1.0 (left sums are >= 0, total must be < 1)
    keepR = sumsR < 1.0
    sR = sumsR[keepR]
    mR = np.nonzero(keepR)[0]
    order = np.argsort(sR)
    sR_sorted = sR[order]
    mR_sorted = mR[order]

    # for every left sum, best right sum with total strictly < 1.0
    target = 1.0 - sumsL
    idx = np.searchsorted(sR_sorted, target, side='left') - 1
    valid = idx >= 0
    total = np.full(len(sumsL), -1.0)
    total[valid] = sumsL[valid] + sR_sorted[idx[valid]]
    best_pos = int(np.argmax(total))
    maskL = best_pos
    maskR = int(mR_sorted[idx[best_pos]])

    chosen = []
    for i in range(kl):
        if maskL & (1 << i):
            chosen.append(left[i])
    for i in range(kr):
        if maskR & (1 << i):
            chosen.append(right[i])
    if not chosen:
        chosen = [0]

    # exact safety: ensure strictly below 1
    if exact_sum(chosen) >= L:
        if len(chosen) > 1:
            chosen.remove(min(chosen, key=lambda i: weights[i]))
        if not chosen or exact_sum(chosen) >= L:
            chosen = [0]

    best_set = set(chosen)
    best_sum = exact_sum(chosen)

    # ---- deterministic exact polish: 1-add then 1-swap, repeat ----
    def polish():
        nonlocal best_set, best_sum
        improved = True
        while improved and ok():
            improved = False
            for i in range(n):
                if i not in best_set and best_sum + weights[i] < L:
                    best_set.add(i)
                    best_sum += weights[i]
                    improved = True
            for rem in list(best_set):
                s_without = best_sum - weights[rem]
                for add in range(n):
                    if add in best_set:
                        continue
                    ns = s_without + weights[add]
                    if ns < L and ns > best_sum:
                        best_set.discard(rem)
                        best_set.add(add)
                        best_sum = ns
                        improved = True
                        break
                if improved:
                    break

    polish()

    # ---- randomized exact local search with remaining time ----
    if ok():
        rng = random.Random(12345)
        for _ in range(300000):
            if not ok():
                break
            cur_set = set(best_set)
            cur_sum = best_sum
            move = rng.random()
            if move < 0.34:
                avail = [i for i in range(n) if i not in cur_set]
                if not avail:
                    continue
                a = rng.choice(avail)
                ns = cur_sum + weights[a]
                if ns < L:
                    cur_set.add(a)
                    cur_sum = ns
            elif move < 0.67:
                if len(cur_set) <= 1:
                    continue
                r = rng.choice(list(cur_set))
                cur_set.discard(r)
                cur_sum -= weights[r]
            else:
                if not cur_set:
                    continue
                r = rng.choice(list(cur_set))
                avail = [i for i in range(n) if i not in cur_set]
                if not avail:
                    continue
                a = rng.choice(avail)
                ns = cur_sum - weights[r] + weights[a]
                if ns < L:
                    cur_set.discard(r)
                    cur_set.add(a)
                    cur_sum = ns
            if cur_sum > best_sum:
                best_set = cur_set
                best_sum = cur_sum
        polish()

    best_indices = list(best_set)
    if not best_indices or exact_sum(best_indices) >= L:
        best_indices = [0]
    return best_indices
