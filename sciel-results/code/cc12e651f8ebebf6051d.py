"""Bootstrap spawn: MODEL-AND-OPTIMIZE approach for divisor-sum fiber ratio."""
# EVOLVE-BLOCK-START
import time
import bisect
import numpy as np
from fractions import Fraction

MAXV = 10000000
WALL_LIMIT = 90.0
SAFETY = 0.82

def construct():
    t0 = time.time()
    hard_stop = t0 + WALL_LIMIT * SAFETY

    def time_left():
        return hard_stop - time.time()

    try:
        N = MAXV

        # ---------- Linear Sieve for sigma (using lists for speed) ----------
        lp = [0] * (N + 1)
        sigma_list = [1] * (N + 1)
        pk = [1] * (N + 1)
        sigma_pk = [1] * (N + 1)
        
        lp[1] = 1
        sigma_list[1] = 1
        pk[1] = 1
        sigma_pk[1] = 1
        
        primes = []
        for i in range(2, N + 1):
            if lp[i] == 0:
                lp[i] = i
                primes.append(i)
                pk[i] = i
                sigma_pk[i] = 1 + i
                sigma_list[i] = 1 + i
            for p in primes:
                if p > lp[i] or i * p > N:
                    break
                lp[i * p] = p
                if p == lp[i]:
                    pk[i * p] = pk[i] * p
                    sigma_pk[i * p] = sigma_pk[i] + pk[i * p]
                    sigma_list[i * p] = sigma_list[i // pk[i]] * sigma_pk[i * p]
                    break
                else:
                    pk[i * p] = p
                    sigma_pk[i * p] = 1 + p
                    sigma_list[i * p] = sigma_list[i] * sigma_pk[i * p]
            if (i & 0x3FFFFF) == 0 and time_left() < 30.0:
                return [11, 6]

        if time_left() < 15.0:
            return [11, 6]

        sigma = np.array(sigma_list, dtype=np.int64)

        # ---------- Group indices by sigma value ----------
        idx = np.arange(1, N + 1, dtype=np.int32)
        vals = sigma[1:]
        order = np.argsort(vals, kind='stable')
        svals = vals[order]
        M = order.size
        change = np.empty(M, dtype=bool)
        change[0] = True
        change[1:] = svals[1:] != svals[:-1]
        starts = np.nonzero(change)[0]
        ends = np.append(starts[1:], M)

        best_val = None
        best_pair = None
        max_b = (2 * N) // 3

        for gi in range(starts.size):
            s0 = int(starts[gi]); e0 = int(ends[gi])
            length = e0 - s0
            if length >= 2:
                gn = np.sort(idx[order[s0:e0]]).tolist()
                L = len(gn)
                # For each b, find a closest to 1.5*b
                for bi in range(L):
                    b = gn[bi]
                    if b > max_b:
                        break
                    target = 1.5 * b
                    j = bisect.bisect_left(gn, target)
                    # Check j and j-1
                    for jj in (j, j - 1):
                        if 0 <= jj < L:
                            a = gn[jj]
                            if a != b and 1 <= a <= N:
                                num = abs(2 * a - 3 * b)
                                if num > 0:
                                    val = num / (2 * b)
                                    if best_val is None or val < best_val:
                                        best_val = val
                                        best_pair = (a, b)
            if (gi & 0x3FFF) == 0 and time_left() < 5.0:
                break
            if time_left() < 5.0:
                break

        if best_pair is None:
            return [11, 6]
        a, b = int(best_pair[0]), int(best_pair[1])
        if not (1 <= a <= N and 1 <= b <= N):
            return [11, 6]
        if 2 * a == 3 * b:
            return [11, 6]
        return [a, b]
    except Exception:
        return [11, 6]
# EVOLVE-BLOCK-END
