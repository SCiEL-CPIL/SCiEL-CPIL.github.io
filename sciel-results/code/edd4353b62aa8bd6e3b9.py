import time
import random
from math import isqrt

def _build_spf(limit):
    spf = list(range(limit + 1))
    spf[0] = 0
    spf[1] = 1
    for i in range(2, isqrt(limit) + 1):
        if spf[i] == i:
            step = i
            start = i * i
            for j in range(start, limit + 1, step):
                if spf[j] == j:
                    spf[j] = i
    return spf

def _factor_set(n, spf):
    s = set()
    while n > 1:
        p = spf[n]
        s.add(p)
        while n % p == 0:
            n //= p
    return s

def _initial_score(arr, spf):
    s = set()
    n = len(arr)
    for i in range(n):
        ai = arr[i]
        for j in range(i + 1, n):
            s |= _factor_set(ai + arr[j], spf)
    return s

def _delta_score_remove(arr, i, cur, spf):
    n = len(arr)
    ai = arr[i]
    new = set(cur)
    for j in range(n):
        if j == i:
            continue
        s = _factor_set(ai + arr[j], spf)
        for p in s:
            new.discard(p)
    return new

def _delta_score_add(arr, i, val, cur, spf):
    new = set(cur)
    for j in range(len(arr)):
        if j == i:
            continue
        new |= _factor_set(val + arr[j], spf)
    return new

def _hill_climb(arr, spf, deadline, rng):
    cur = list(arr)
    cur_score = _initial_score(cur, spf)
    n = len(cur)
    while time.time() < deadline:
        improved = False
        for i in range(n):
            old = cur[i]
            old_score = cur_score
            rem = _delta_score_remove(cur, i, old_score, spf)
            best_v = None
            best_s = None
            best_len = len(old_score)
            for v in range(2, 10001, 2):
                if v == old:
                    continue
                if v in cur:
                    continue
                ns = _delta_score_add(cur, i, v, rem, spf)
                ln = len(ns)
                if ln < best_len:
                    best_len = ln
                    best_v = v
                    best_s = ns
                if ln < 2:
                    break
            if best_v is not None:
                cur[i] = best_v
                cur_score = best_s
                improved = True
        if not improved:
            break
    return cur, cur_score

def _anneal(arr, spf, deadline, rng):
    cur = list(arr)
    cur_score = _initial_score(cur, spf)
    n = len(cur)
    best = list(cur)
    best_score = set(cur_score)
    t = 1.0
    cooling = 0.999
    while time.time() < deadline and t > 0.001:
        i = rng.randrange(n)
        old = cur[i]
        old_score = cur_score
        rem = _delta_score_remove(cur, i, old_score, spf)
        v = rng.randrange(1, 5001) * 2
        if v == old or v in cur:
            t *= cooling
            continue
        ns = _delta_score_add(cur, i, v, rem, spf)
        delta = len(ns) - len(old_score)
        if delta <= 0 or rng.random() < pow(2.718281828459045, -delta / t):
            cur[i] = v
            cur_score = ns
            if len(cur_score) < len(best_score):
                best = list(cur)
                best_score = set(cur_score)
        t *= cooling
    return best, best_score

def construct():
    t0 = time.time()
    deadline = t0 + 75.0
    rng = random.Random(12345)

    limit = 20000
    spf = _build_spf(limit)

    # Deterministic base: 12 even numbers with small pairwise sums
    base = list(range(2, 25, 2))  # 2..24
    base_score = _initial_score(base, spf)

    # Hill climb from base
    hc, hc_score = _hill_climb(base, spf, min(deadline, t0 + 25.0), rng)

    best = hc
    best_score = hc_score

    # Try a few more random even starting points + hill climb
    remaining = deadline - time.time()
    attempts = 0
    while time.time() < deadline - 10.0 and attempts < 6:
        attempts += 1
        start = list(rng.sample(range(2, 10001, 2), 12))
        sub_deadline = min(deadline - 10.0, time.time() + max(2.0, remaining / (6 - attempts)))
        cand, cs = _hill_climb(start, spf, sub_deadline, rng)
        if len(cs) < len(best_score):
            best = cand
            best_score = cs

    # Simulated annealing from best
    if time.time() < deadline - 5.0:
        sa, sa_score = _anneal(best, spf, deadline - 2.0, rng)
        if len(sa_score) < len(best_score):
            best = sa
            best_score = sa_score

    # Final safety
    result = list(best)
    # Ensure validity
    assert len(set(result)) == 12
    assert all(1 <= x <= 10000 for x in result)
    return result
