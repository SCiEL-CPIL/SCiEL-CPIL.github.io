import time
import random
from fractions import Fraction


def construct():
    start = time.time()
    deadline = 80.0  # safety margin under 90s wall-clock

    denominators = list(range(2, 49))
    n = len(denominators)  # 47
    LCM = 735134400  # lcm(2..48)
    vals = [LCM // d for d in denominators]

    def time_ok():
        return (time.time() - start) < deadline

    # ---- Exact meet-in-the-middle: best (largest < LCM) sum for each size k ----
    def mitm_best(k):
        if k < 0 or k > n:
            return -1
        if k == 0:
            return 0
        half = n // 2
        A = vals[:half]
        B = vals[half:]
        m1 = len(A)
        m2 = len(B)

        def gen(arr, m, target_k):
            res = {0: [0] * (target_k + 1)}
            for i in range(m):
                v = arr[i]
                for kk in range(i, -1, -1):
                    prev = res.get(kk)
                    if prev is None:
                        continue
                    nk = kk + 1
                    if nk > target_k:
                        continue
                    lst = res.get(nk)
                    if lst is None:
                        res[nk] = [prev[kk] + v]
                    else:
                        lst.append(prev[kk] + v)
            return res

        dA = gen(A, m1, k)
        dB = gen(B, m2, k)

        best = -1
        for ka in range(0, k + 1):
            kb = k - ka
            la = dA.get(ka)
            lb = dB.get(kb)
            if not la or not lb:
                continue
            lb.sort()
            i = 0
            lenb = len(lb)
            for a in la:
                need = LCM - 1 - a
                if i >= lenb:
                    break
                if lb[i] > need:
                    continue
                # binary search largest index with lb[idx] <= need
                lo = i
                hi = lenb
                while lo < hi:
                    mid = (lo + hi) // 2
                    if lb[mid] <= need:
                        lo = mid + 1
                    else:
                        hi = mid
                idx = lo - 1
                if idx >= 0:
                    s = a + lb[idx]
                    if s > best:
                        best = s
        return best

    best_sum = -1
    best_k = -1
    for k in range(0, n + 1):
        if not time_ok():
            break
        s = mitm_best(k)
        if s > best_sum:
            best_sum = s
            best_k = k
        if best_sum == LCM - 1:
            break  # cannot do better

    if best_sum >= 0:
        deficit = LCM - best_sum
        assert deficit > 0, "strictly below 1 violated"
        return best_k, deficit

    # ---- Fallback: greedy + local search (should not be reached) ----
    def deficit_of(indices):
        s = 0
        for i in indices:
            s += vals[i]
        if s >= LCM:
            return None
        return LCM - s

    best_def = LCM
    best_idx = []
    cur = []
    cur_sum = 0
    for i in range(n):
        if cur_sum + vals[i] < LCM:
            cur.append(i)
            cur_sum += vals[i]
    d = deficit_of(cur)
    if d is not None and d < best_def:
        best_def = d
        best_idx = list(cur)

    for _ in range(200):
        if not time_ok():
            break
        improved = True
        while improved and time_ok():
            improved = False
            cs = set(cur)
            # add
            for i in range(n):
                if i in cs:
                    continue
                if cur_sum + vals[i] < LCM:
                    nd = LCM - (cur_sum + vals[i])
                    if nd < LCM - cur_sum:
                        cur.append(i)
                        cur_sum += vals[i]
                        cs.add(i)
                        if cur_sum > best_def and (LCM - cur_sum) < best_def:
                            best_def = LCM - cur_sum
                            best_idx = list(cur)
                        improved = True
                        break
            if improved:
                continue
            # swap
            for i in range(n):
                if i not in cs:
                    continue
                for j in range(n):
                    if j in cs:
                        continue
                    ns = cur_sum - vals[i] + vals[j]
                    if ns < LCM and (LCM - ns) < (LCM - cur_sum):
                        cur = [j if x == i else x for x in cur]
                        cur_sum = ns
                        cs.remove(i)
                        cs.add(j)
                        if (LCM - cur_sum) < best_def:
                            best_def = LCM - cur_sum
                            best_idx = list(cur)
                        improved = True
                        break
                if improved:
                    break

    if best_idx:
        return len(best_idx), best_def
    return 0, LCM
