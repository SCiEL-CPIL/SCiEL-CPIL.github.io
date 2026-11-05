import time
import heapq
import numpy as np
from fractions import Fraction

def construct():
    start = time.time()
    deadline = start + 80

    denominators = list(range(2, 49))
    n = len(denominators)  # 47
    mid = 23
    A_dens = denominators[:mid]   # 2..24
    B_dens = denominators[mid:]   # 25..48

    def build_sums(dens):
        sums = np.zeros(1, dtype=np.float64)
        for d in dens:
            w = 1.0 / d
            sums = np.concatenate((sums, sums + w))
        return sums

    A_sums = build_sums(A_dens)   # 2^23
    B_sums = build_sums(B_dens)   # 2^24

    order = np.argsort(A_sums, kind='stable')
    A_sorted = A_sums[order]
    del A_sums

    K = 600
    heap = []
    chunk = 1 << 20
    nb = len(B_sums)
    for s in range(0, nb, chunk):
        Bc = B_sums[s:s+chunk]
        targets = 1.0 - Bc
        a_idxs = np.searchsorted(A_sorted, targets, side='right') - 1
        valid = a_idxs >= 0
        if not valid.any():
            continue
        a_sel = a_idxs[valid]
        b_sel = np.nonzero(valid)[0] + s
        totals = A_sorted[a_sel] + B_sums[b_sel]
        ok = totals < 1.0
        if not ok.any():
            continue
        a_sel = a_sel[ok]; b_sel = b_sel[ok]; totals = totals[ok]
        a_masks = order[a_sel]
        m = len(totals)
        if m <= K:
            top = np.argsort(totals)[::-1]
        else:
            top = np.argpartition(totals, -K)[-K:]
        for j in top:
            heapq.heappush(heap, (float(totals[j]), int(a_masks[j]), int(b_sel[j])))
            if len(heap) > K:
                heapq.heappop(heap)
        if time.time() > deadline - 5:
            break

    def mask_indices(mask, offset):
        res = []
        i = 0
        while mask:
            if mask & 1:
                res.append(offset + i)
            mask >>= 1
            i += 1
        return res

    best = None
    for t, am, bm in sorted(heap, key=lambda x: x[0], reverse=True):
        idxs = mask_indices(am, 0) + mask_indices(bm, mid)
        if not idxs:
            continue
        s = sum(Fraction(1, denominators[i]) for i in idxs)
        if s < 1:
            best = (1 - s, idxs)
            break

    if best is None:
        return [0]

    deficit, sel = best
    sel = list(sel)

    # Light exact local search (safety net)
    if time.time() < deadline - 3:
        sel_set = set(sel)
        unselected = [i for i in range(n) if i not in sel_set]
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for rem in list(sel):
                new_total = sum(Fraction(1, denominators[i]) for i in sel) - Fraction(1, denominators[rem])
                for add in unselected:
                    cand = new_total + Fraction(1, denominators[add])
                    if cand < 1:
                        d = 1 - cand
                        if d < deficit:
                            deficit = d
                            sel.remove(rem)
                            sel.append(add)
                            sel_set.discard(rem)
                            sel_set.add(add)
                            unselected.remove(add)
                            unselected.append(rem)
                            improved = True
                            break
                if time.time() > deadline - 3:
                    break
            if time.time() > deadline - 3:
                break
        # try single adds
        if time.time() < deadline - 3:
            for add in unselected:
                cand = sum(Fraction(1, denominators[i]) for i in sel) + Fraction(1, denominators[add])
                if cand < 1:
                    d = 1 - cand
                    if d < deficit:
                        deficit = d
                        sel.append(add)
                        sel_set.add(add)
                        unselected.remove(add)
                        improved = True

    # final verify
    final = sum(Fraction(1, denominators[i]) for i in sel)
    if sel and final < 1:
        return sel
    return [0]
