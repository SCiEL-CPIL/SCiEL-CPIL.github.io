import time
import random
from math import isqrt

_N = 256
_MAXP = _N * _N          # 65536, max possible b*c
_SAFETY = 8.0            # wall-clock safety margin before the 90s kill

_START = time.time()


def _elapsed():
    return time.time() - _START


def _deadline():
    return _START + 90.0 - _SAFETY


def _precompute(n):
    """divs[m] = list of d in [2,n] with d | m, for m up to n*n.
    Then  a | (b*c)  <=>  a in divs[b*c]  (O(1) via set membership)."""
    max_prod = n * n
    divs = [set() for _ in range(max_prod + 1)]
    for d in range(2, n + 1):
        for m in range(d, max_prod + 1, d):
            divs[m].add(d)
    return divs


def construct():
    n = _N
    divs = _precompute(n)
    uni = list(range(2, n + 1))

    def can_add(x, cur, curset):
        """True iff adding x keeps the set valid.
        New triples are exactly those containing x: {x,a,b} with a,b distinct in cur.
        Need: a ∤ x*b, b ∤ x*a, x ∤ a*b  for all distinct a,b in cur."""
        L = len(cur)
        for i in range(L):
            a = cur[i]
            for j in range(i + 1, L):
                b = cur[j]
                if a in divs[x * b] or b in divs[x * a] or x in divs[a * b]:
                    return False
        return True

    def is_valid_set(S):
        L = len(S)
        for i in range(L):
            a = S[i]
            for j in range(i + 1, L):
                b = S[j]
                for k in range(j + 1, L):
                    c = S[k]
                    if a in divs[b * c] or b in divs[a * c] or c in divs[a * b]:
                        return False
        return True

    # ---- Greedy initial solution: primes first (they are the 'safest' divisors),
    #      then composites in increasing order.
    def greedy():
        def is_prime(m):
            if m < 2:
                return False
            if m < 4:
                return True
            if m % 2 == 0 or m % 3 == 0:
                return False
            i = 5
            while i * i <= m:
                if m % i == 0 or m % (i + 2) == 0:
                    return False
                i += 6
            return True

        order = [a for a in uni if is_prime(a)] + \
                [a for a in uni if not is_prime(a)]
        S = []
        Sset = set()
        for x in order:
            if can_add(x, S, Sset):
                S.append(x)
                Sset.add(x)
        return S

    best = greedy()
    best_set = set(best)

    # ---- Local search: repeatedly add any missing element that fits;
    #      with perturbation restarts to escape local optima.
    rng = random.Random(12345)

    def local_search(cur, curset, missing):
        nonlocal best, best_set
        improved = True
        while improved and _elapsed() < _deadline():
            improved = False
            rng.shuffle(missing)
            for x in missing:
                if _elapsed() > _deadline():
                    return
                if x in curset:
                    continue
                if can_add(x, cur, curset):
                    cur.append(x)
                    curset.add(x)
                    missing.remove(x)
                    improved = True
                    if len(cur) > len(best):
                        best = cur[:]
                        best_set = curset.copy()
        return cur, curset, missing

    cur = best[:]
    curset = set(cur)
    missing = [x for x in uni if x not in curset]
    local_search(cur, curset, missing)

    # Perturbation restarts: remove a few random members, then re-fill.
    restart = 0
    while _elapsed() < _deadline():
        restart += 1
        cur = best[:]
        curset = set(cur)
        missing = [x for x in uni if x not in curset]
        if len(cur) > 5:
            for _ in range(rng.randint(1, 4)):
                r = rng.choice(cur)
                cur.remove(r)
                curset.discard(r)
                missing.append(r)
        local_search(cur, curset, missing)

    result = best

    # Final safety: verify, and fall back to a clean greedy build if broken.
    if not is_valid_set(result):
        result = []
        rs = set()
        for x in uni:
            if can_add(x, result, rs):
                result.append(x)
                rs.add(x)

    return [int(x) for x in result]
