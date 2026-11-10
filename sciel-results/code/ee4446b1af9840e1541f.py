"""Find a, b <= 1e7 with sigma(a)=sigma(b) minimizing |a/b - 3/2|, using sieve + fiber two-pointer."""
import time

MAX = 10000000
LIMIT = 72.0          # seconds; evaluator kills at 90s, keep a wide safety margin
T2 = 2                 # target ratio 3/2 -> numerator 3, denominator 2
T3 = 3


def _sieve_sigma(n):
    sigma = [1] * (n + 1)
    for p in range(2, n + 1):
        if sigma[p] == 1:          # p is prime (unvisited)
            # add p^k terms to every multiple of p
            pk = p
            while pk <= n:
                term = pk * (p + 1) - p
                for m in range(pk, n + 1, pk):
                    sigma[m] += term
                pk *= p
    return sigma


def construct():
    t0 = time.time()
    sigma = _sieve_sigma(MAX)

    # Group indices by sigma value (fiber).
    # Use dict of lists; sigma values can be large but counts are manageable.
    fibers = {}
    for i in range(1, MAX + 1):
        s = sigma[i]
        if s in fibers:
            fibers[s].append(i)
        else:
            fibers[s] = [i]

    best_a, best_b = None, None
    best_err = float('inf')

    for s, lst in fibers.items():
        if time.time() - t0 > LIMIT - 15.0:
            break
        if len(lst) < 2:
            continue
        # Two-pointer: for each a, find b closest to (2/3)*a.
        # lst is sorted ascending.
        n = len(lst)
        j = 0
        for i in range(n):
            a = lst[i]
            # target b = (2/3)*a
            target = (2.0 * a) / 3.0
            # advance j to first index with lst[j] >= target
            while j < n and lst[j] < target:
                j += 1
            # consider j and j-1
            for idx in (j, j - 1):
                if 0 <= idx < n:
                    b = lst[idx]
                    if b == a:
                        continue
                    # exact rational error |a/b - 3/2| = |2a - 3b| / (2b)
                    num = 2 * a - 3 * b
                    if num < 0:
                        num = -num
                    if num == 0:
                        continue
                    den = 2 * b
                    # compare num/den < best_err
                    if num / den < best_err:
                        best_err = num / den
                        best_a, best_b = a, b

    if best_a is None:
        # fallback (should not happen)
        return [11, 6]
    # ensure types are plain Python ints in a list
    return [int(best_a), int(best_b)]
