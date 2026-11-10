import time
import math
import random
import numpy as np

def _primes_upto(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(math.isqrt(n)) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(n + 1) if sieve[i]]

def _coverage(residues, primes, M):
    counts = np.zeros(M + 1, dtype=np.int32)
    for i, p in enumerate(primes):
        r = residues[i]
        start = r if r > 0 else p
        if start > M:
            continue
        counts[start::p] += 1
    return counts

def _prefix_L(counts):
    # counts[0] unused; need counts[x] >= 2 for x=1..L
    bad = np.nonzero(counts[1:] < 2)[0]
    if bad.size == 0:
        return int(len(counts) - 1)
    return int(bad[0])  # first bad index (1-based)

def construct():
    start_time = time.time()
    TIME_LIMIT = 140.0

    primes = _primes_upto(1000)
    n = len(primes)
    M = 20000

    def deadline_ok():
        return (time.time() - start_time) < TIME_LIMIT

    def greedy_init(seed):
        rng = random.Random(seed)
        residues = [0] * n
        counts = np.zeros(M + 1, dtype=np.int32)
        for i in range(n):
            p = primes[i]
            r = rng.randrange(p)
            residues[i] = r
            start = r if r > 0 else p
            if start <= M:
                counts[start::p] += 1
        L = _prefix_L(counts)
        # Greedy: extend by fixing first bad point with a safe prime
        for _ in range(3000):
            if L >= M - 1:
                break
            x = L + 1
            if counts[x] >= 2:
                L += 1
                continue
            # find a prime that doesn't already cover x and whose change keeps prefix safe
            improved = False
            for i in range(n):
                p = primes[i]
                r_old = residues[i]
                r_new = x % p
                if r_new == r_old:
                    continue
                # check safety on prefix: positions with x%p==r_old must have count>=3
                if r_old == 0:
                    first = p
                else:
                    first = r_old
                # scan prefix positions
                ok = True
                xx = first
                while xx <= L:
                    if counts[xx] < 3:
                        ok = False
                        break
                    xx += p
                if not ok:
                    continue
                # apply
                if r_old == 0:
                    f_old = p
                else:
                    f_old = r_old
                counts[f_old::p] -= 1
                if r_new == 0:
                    f_new = p
                else:
                    f_new = r_new
                counts[f_new::p] += 1
                residues[i] = r_new
                improved = True
                break
            if not improved:
                break
        return residues, L

    best_res = None
    best_L = -1
    # Try several greedy seeds
    for s in range(12):
        if not deadline_ok():
            break
        res, L = greedy_init(1000 + s)
        if L > best_L:
            best_L = L
            best_res = list(res)

    # Local search + simulated annealing from best
    def local_search(residues, L):
        counts = _coverage(residues, primes, M)
        L = _prefix_L(counts)
        # safety margin: only do moves if prefix is "safe" (all >=3) so L never drops
        # but we allow moves that keep L non-decreasing by checking after.
        improved = True
        while improved and deadline_ok():
            improved = False
            if L >= M - 1:
                break
            x = L + 1
            if counts[x] >= 2:
                L += 1
                continue
            # try each prime
            best_gain = 0
            best_i = -1
            best_r = -1
            for i in range(n):
                p = primes[i]
                r_old = residues[i]
                r_new = x % p
                if r_new == r_old:
                    continue
                # delta: positions old -> -1, new -> +1
                # safety: any prefix pos with old residue must stay >=2 after -1 => currently >=3
                if r_old == 0:
                    f_old = p
                else:
                    f_old = r_old
                ok = True
                xx = f_old
                while xx <= L:
                    if counts[xx] < 3:
                        ok = False
                        break
                    xx += p
                if not ok:
                    continue
                # compute new L after applying (cheap: apply, measure, revert if not better)
                if r_new == 0:
                    f_new = p
                else:
                    f_new = r_new
                counts[f_old::p] -= 1
                counts[f_new::p] += 1
                residues[i] = r_new
                newL = _prefix_L(counts)
                if newL > L:
                    L = newL
                    improved = True
                    break
                else:
                    # revert
                    counts[f_new::p] -= 1
                    counts[f_old::p] += 1
                    residues[i] = r_old
            # end while
        return residues, L

    if best_res is not None and deadline_ok():
        best_res, best_L = local_search(best_res, best_L)

    # Multi-start simulated annealing to push further
    def anneal(residues, L, iters):
        counts = _coverage(residues, primes, M)
        L = _prefix_L(counts)
        cur = L
        rng = random.Random(42)
        T = 3.0
        for it in range(iters):
            if not deadline_ok():
                break
            if L >= M - 1:
                break
            i = rng.randrange(n)
            p = primes[i]
            r_old = residues[i]
            r_new = rng.randrange(p)
            if r_new == r_old:
                continue
            if r_old == 0:
                f_old = p
            else:
                f_old = r_old
            if r_new == 0:
                f_new = p
            else:
                f_new = r_new
            counts[f_old::p] -= 1
            counts[f_new::p] += 1
            residues[i] = r_new
            newL = _prefix_L(counts)
            delta = newL - L
            if delta >= 0:
                L = newL
                cur = L
            else:
                # accept with prob exp(delta/T)
                if rng.random() < math.exp(delta / T):
                    L = newL
                else:
                    # revert
                    counts[f_new::p] -= 1
                    counts[f_old::p] += 1
                    residues[i] = r_old
            T = max(0.05, T * 0.999)
        return residues, L

    if best_res is not None and deadline_ok():
        # a few anneal passes from current best
        for k in range(4):
            if not deadline_ok():
                break
            res, L = anneal(list(best_res), best_L, 20000)
            if L > best_L:
                best_L = L
                best_res = res

    # Final local search
    if best_res is not None and deadline_ok():
        best_res, best_L = local_search(best_res, best_L)

    # Ensure valid Python ints
    best_res = [int(r) % primes[i] for i, r in enumerate(best_res)]
    return best_res

if __name__ == "__main__":
    r = construct()
    print(len(r))
