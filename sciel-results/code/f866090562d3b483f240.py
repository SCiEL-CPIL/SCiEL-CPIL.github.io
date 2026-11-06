"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

N = 511
LIMIT = 512


def _sieve_primes(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin under 150s kill

    primes = _sieve_primes(N)
    pindex = {p: i for i, p in enumerate(primes)}

    # mult_mask[p]: bitmask of all multiples of p in [1..N]
    mult_mask = {}
    for p in primes:
        m = 0
        for b in range(p, N + 1, p):
            m |= (1 << b)
        mult_mask[p] = m

    # conflict[a]: bitmask of all b != a sharing a prime factor with a
    conflict = [0] * (N + 1)
    for a in range(2, N + 1):
        x = a
        cm = 0
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                cm |= mult_mask[p]
                while x % p == 0:
                    x //= p
        if x > 1:
            cm |= mult_mask[x]
        conflict[a] = cm & ~(1 << a)

    # weight[a] = 1/(512 - a)
    w = [0.0] * (N + 1)
    for a in range(1, N + 1):
        w[a] = 1.0 / (LIMIT - a)

    # ---- Greedy seed: order by weight descending (= a descending) ----
    S = set()
    S_mask = 0
    for a in range(N, 0, -1):
        if (conflict[a] & S_mask) == 0:
            S_mask |= (1 << a)
            S.add(a)

    def score():
        return sum(w[a] for a in S)

    best = score()

    def try_additions():
        nonlocal S, S_mask, best
        improved = True
        while improved:
            improved = False
            for a in range(N, 0, -1):
                if not ((S_mask >> a) & 1) and (conflict[a] & S_mask) == 0:
                    S_mask |= (1 << a)
                    S.add(a)
                    improved = True
        cur = score()
        if cur > best:
            best = cur

    def try_single_swaps():
        nonlocal S, S_mask, best
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in list(S):
                if time.time() >= deadline:
                    return
                xm = 1 << x
                if not (S_mask & xm):
                    continue
                rest = S_mask ^ xm
                # best replacement a not in S, coprime to rest, w[a] > w[x]
                best_a = -1
                best_w = w[x]
                for a in range(N, 0, -1):
                    if (S_mask >> a) & 1:
                        continue
                    if w[a] <= best_w:
                        continue
                    if (conflict[a] & rest) == 0:
                        best_a = a
                        break
                if best_a >= 0:
                    S_mask = rest | (1 << best_a)
                    S.discard(x)
                    S.add(best_a)
                    improved = True
        cur = score()
        if cur > best:
            best = cur

    def try_multi_swaps():
        """Replace a set R of low-weight elements with a single high-weight a
        that is coprime to everything else. Escapes single-swap local optima."""
        nonlocal S, S_mask, best
        improved = True
        while improved and time.time() < deadline:
            improved = False
            s_list = list(S)
            n = len(s_list)
            # For each candidate a (high weight first), find a small set R of
            # currently-chosen elements whose union of conflicts is exactly
            # covered by a's conflicts, and where sum of w[R] < w[a].
            # Practical approach: try R = single element first (already done),
            # then R = pairs of low-weight elements.
            # We try: for each a not in S, find elements in S that conflict
            # with a. If that conflict set C is non-empty and small, removing
            # C and adding a improves if w[a] > sum(w[c] for c in C).
            for a in range(N, 0, -1):
                if time.time() >= deadline:
                    return
                if (S_mask >> a) & 1:
                    continue
                # which chosen elements conflict with a?
                C_mask = conflict[a] & S_mask
                if C_mask == 0:
                    # a is coprime to all chosen -> just add (handled by additions)
                    S_mask |= (1 << a)
                    S.add(a)
                    improved = True
                    continue
                # C is the set of chosen elements that conflict with a.
                # If we remove C and add a, we gain w[a] - sum(w[c] for c in C).
                C = []
                m = C_mask
                while m:
                    lb = m & (-m)
                    c = lb.bit_length() - 1
                    C.append(c)
                    m ^= lb
                if len(C) > 3:
                    continue  # too risky / too costly to remove
                gain = w[a] - sum(w[c] for c in C)
                if gain > 1e-12:
                    for c in C:
                        S_mask ^= (1 << c)
                        S.discard(c)
                    S_mask |= (1 << a)
                    S.add(a)
                    improved = True
                    break  # restart scan after a change
        cur = score()
        if cur > best:
            best = cur

    # Alternate passes until no improvement or deadline
    try_additions()
    try_single_swaps()
    while time.time() < deadline:
        before = score()
        try_multi_swaps()
        try_single_swaps()
        try_additions()
        after = score()
        if after <= before + 1e-12:
            # try one more multi-swap round in case of cycling
            before2 = score()
            try_multi_swaps()
            after2 = score()
            if after2 <= before2 + 1e-12:
                break

    return list(S)
# EVOLVE-BLOCK-END
