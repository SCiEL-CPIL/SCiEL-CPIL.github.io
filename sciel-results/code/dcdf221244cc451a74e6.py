import time
import math
import random
import numpy as np

def _fft_sumdiff_count(u, M):
    """Return (|U+U|, |U-U|) for set U of nonneg ints with max M, via FFT."""
    n = max(M + 1, 2)
    size = 1
    while size < 2 * n:
        size <<= 1
    a = np.zeros(size)
    a[u] = 1.0
    f = np.fft.rfft(a)
    af = f * f
    sm = np.fft.irfft(af, n=size)
    sm = np.rint(sm)
    sum_count = int(np.count_nonzero(sm[:2 * M + 1] > 0.5))
    # difference: U - U  <=>  convolution of U with reverse(U)
    br = np.zeros(size)
    br[:n] = a[:n][::-1]
    g = np.fft.rfft(br)
    cg = af * g
    df = np.fft.irfft(cg, n=size)
    df = np.rint(df)
    # index (n-1) is the zero difference (all pairs). support in [0, 2n-2].
    diff_count = int(np.count_nonzero(df[:2 * n - 1] > 0.5))
    return sum_count, diff_count


def _score_of(u, M):
    """Score = 1 + log(|U-U|/|U+U|)/log(2M+1)."""
    sum_count, diff_count = _fft_sumdiff_count(u, M)
    if sum_count <= 0 or diff_count <= 0:
        return -1e18
    return 1.0 + math.log(diff_count / sum_count) / math.log(2 * M + 1)


def construct():
    t0 = time.time()
    deadline = t0 + 115.0

    rng = random.Random(1234567)
    best_u = None
    best_M = 0
    best_score = -1e18

    def consider(u, M):
        nonlocal best_u, best_M, best_score
        s = _score_of(u, M)
        if s > best_score:
            best_score = s
            best_M = M
            best_u = list(u)
        return s

    # seed with a few small known-good sets
    for seed in ([0, 1, 3], [0, 1, 2, 4, 8, 16, 32, 64, 128, 256],
                 [0, 1, 3, 9, 27, 81, 243, 729, 2187]):
        M = max(seed)
        consider(np.array(seed, dtype=np.int64), M)

    # sweep over (M, k) targets, running local search with a time budget
    # larger M and larger k push the exponent up.
    Ms = [128, 256, 512, 1024, 2048, 4096]
    k_fracs = [0.02, 0.03, 0.05, 0.08, 0.12, 0.2]

    for M in Ms:
        if time.time() > deadline - 5:
            break
        for kf in k_fracs:
            if time.time() > deadline - 3:
                break
            k = max(2, int(round(M * kf)))
            k = min(k, M)
            # initialize: 0 plus k-1 random distinct in [1, M]
            u = np.zeros(M + 1, dtype=np.int64)
            u[0] = 1
            chosen = set(rng.sample(range(1, M + 1), k - 1))
            for c in chosen:
                u[c] = 1
            cur = consider(u, M)
            # local search: bit-flips (toggle) and swaps (move one element)
            iters = 4000
            for it in range(iters):
                if time.time() > deadline - 3:
                    break
                op = rng.random()
                if op < 0.5:
                    # toggle a random position (keep 0 fixed)
                    pos = rng.randint(1, M)
                    u[pos] = 1 - u[pos]
                else:
                    # swap: remove a present element, add an absent one
                    present = np.nonzero(u)[0]
                    present = present[present != 0]
                    if len(present) == 0:
                        continue
                    rem = int(rng.choice(present))
                    absent = np.nonzero(u == 0)[0]
                    if len(absent) == 0:
                        continue
                    add = int(rng.choice(absent))
                    u[rem] = 0
                    u[add] = 1
                new = _score_of(u, M)
                if new > cur:
                    cur = new
                    if cur > best_score:
                        best_score = cur
                        best_M = M
                        best_u = list(np.nonzero(u)[0])
                else:
                    # revert
                    if op < 0.5:
                        u[pos] = 1 - u[pos]
                    else:
                        u[rem] = 1
                        u[add] = 0

    if best_u is None:
        return [0, 1, 3]
    return sorted(best_u)
