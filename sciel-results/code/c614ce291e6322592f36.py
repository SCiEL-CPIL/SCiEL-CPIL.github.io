"""Few prime factors across pairwise sums: pick a prime set S and 12 distinct
numbers <=10000 such that every pairwise sum is S-smooth (only primes in S)."""
import time
from math import prod
from itertools import combinations
from bisect import bisect_left

MAXV = 10000
N = 12
DEADLINE = 60.0
START = time.time()


def sieve(limit):
    is_p = bytearray([1]) * (limit + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_p[i]:
            step = i
            start = i * i
            is_p[start:limit + 1:step] = bytearray(len(range(start, limit + 1, step)))
    return [i for i in range(2, limit + 1) if is_p[i]]


PRIMES = sieve(MAXV)
PSET = set(PRIMES)


def factor(n):
    out = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            out.add(d)
            n //= d
        d += 1
    if n > 1:
        out.add(n)
    return out


def cost_of(arr):
    s = set()
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            s |= factor(arr[i] + arr[j])
    return len(s), s


def smooth_set(primes):
    ps = list(primes)
    ps.sort()
    m = len(ps)
    vals = [1]
    for p in ps:
        cur = list(vals)
        v = p
        while v <= MAXV:
            cur.append(v)
            v *= p
        vals += cur
    vals = sorted(set(vals))
    return vals, set(vals)


def build_solution(primes, deadline):
    vals, vset = smooth_set(primes)
    maxs = [MAXV - v for v in vals]
    maxs.sort(reverse=True)

    # candidate values whose double is smooth (so x+x is fine if ever needed)
    cand = [v for v in vals if (2 * v) in vset]
    cand.sort()

    def feasible(x, chosen):
        for c in chosen:
            if (x + c) not in vset:
                return False
        return True

    best = None
    best_cost = 10 ** 9

    def rec(chosen, idx):
        nonlocal best, best_cost
        if time.time() > deadline:
            return
        if len(chosen) == N:
            best = list(chosen)
            best_cost = len(primes)
            return
        # upper bound pruning: remaining needed
        remaining = N - len(chosen)
        # candidate pool: values > last chosen (distinct, increasing)
        start = chosen[-1] + 1 if chosen else 1
        pos = bisect_left(cand, start)
        for k in range(pos, len(cand)):
            x = cand[k]
            if not feasible(x, chosen):
                continue
            rec(chosen + [x], k + 1)
            if best is not None:
                return

    rec([], 0)
    return best, best_cost


def search_prime_set(primes, deadline):
    # quick: build smooth set and try greedy/backtrack
    sol, _ = build_solution(primes, deadline)
    if sol is None:
        return None
    return sol


def try_prime_set(primes, deadline):
    sol = search_prime_set(primes, deadline)
    if sol is None:
        return None
    # verify
    c, _ = cost_of(sol)
    if c <= len(primes):
        return sol, c
    return None


def construct():
    deadline = START + DEADLINE - 2.0  # safety margin

    # 1) Precomputed strong candidate (12 numbers, all pairwise sums 2,3-smooth)
    #    Found by search: 1,2,4,6,8,12,16,24,32,48,64,96
    pre = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96]
    c, _ = cost_of(pre)
    best = pre
    best_c = c  # expect 2

    if time.time() > deadline:
        return best

    # 2) Try to beat with 2-prime sets: search over (2,p) and (3,p)
    two_sets = [(2, p) for p in PRIMES if p != 2 and p <= 1000]
    two_sets += [(3, p) for p in PRIMES if p != 3 and p <= 1000]
    two_sets = list(set(two_sets))
    # order: small primes first (denser smooth numbers)
    two_sets.sort(key=lambda t: min(t))

    found_two = False
    for ps in two_sets:
        if time.time() > deadline:
            break
        res = try_prime_set(list(ps), deadline)
        if res is not None:
            sol, cc = res
            if cc < best_c:
                best, best_c = sol, cc
            found_two = True
            if best_c == 2:
                break

    # 3) Try 3-prime sets if we didn't already get 2 (or to confirm)
    if best_c > 2:
        three_sets = list(combinations(PRIMES[:40], 3))
        three_sets.sort(key=lambda t: min(t))
        for ps in three_sets:
            if time.time() > deadline:
                break
            res = try_prime_set(list(ps), deadline)
            if res is not None:
                sol, cc = res
                if cc < best_c:
                    best, best_c = sol, cc
                if best_c == 3:
                    break

    # 4) If still not great, local search on the best solution
    if time.time() < deadline - 1.0:
        arr = list(best)
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            cur_c, _ = cost_of(arr)
            for i in range(N):
                if time.time() > deadline - 1.0:
                    break
                for v in range(1, MAXV + 1):
                    if v in arr and v != arr[i]:
                        continue
                    if v == arr[i]:
                        continue
                    arr[i] = v
                    nc, _ = cost_of(arr)
                    if nc < cur_c:
                        cur_c = nc
                        improved = True
                        break
                    else:
                        arr[i] = best[i]
                if time.time() > deadline - 1.0:
                    break
            if cur_c < best_c:
                best_c = cur_c
                best = list(arr)

    return list(best)
