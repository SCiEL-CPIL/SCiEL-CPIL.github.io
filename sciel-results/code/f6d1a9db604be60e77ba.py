"""Divisibility-antichain smooth-number representations for targets 1..128."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    t0 = time.time()
    DEADLINE = t0 + 82.0   # wall-clock guard (evaluator kills at 90s)
    N = 128

    # ---- smooth numbers 2^a*3^b*5^c <= 128 ----
    smooth = []
    a = 1
    while a <= N:
        b = a
        while b <= N:
            c = b
            while c <= N:
                smooth.append(c)
                c *= 5
            b *= 3
        a *= 2
    smooth = sorted(set(smooth))
    S = len(smooth)

    # ---- pairwise "divides" mask: divmask[i] has bit j set iff smooth[i] | smooth[j] ----
    divmask = [0] * S
    for i in range(S):
        ai = smooth[i]
        m = 0
        for j in range(S):
            if smooth[j] % ai == 0:
                m |= (1 << j)
        divmask[i] = m

    # ---- candidate subsets by total sum (only subsets that sum to a target are stored) ----
    by_sum = {}
    for k in range(1, 31):
        for idxs in combinations(range(S), k):
            s = 0
            for i in idxs:
                s += smooth[i]
            if s > N:
                continue
            mask = 0
            for i in idxs:
                mask |= (1 << i)
            by_sum.setdefault(s, []).append((k, mask))
    if time.time() > DEADLINE - 4.0:
        # cannot finish; fall through to a safe fallback below
        pass

    # ---- solve each target, minimum term count first ----
    answers = [None] * N
    for n in range(1, N + 1):
        if time.time() > DEADLINE - 3.0:
            break
        cands = by_sum.get(n)
        if not cands:
            continue
        cands.sort(key=lambda x: x[0])
        for k, mask in cands:
            ok = True
            mm = mask
            while mm:
                lsb = mm & (-mm)
                i = lsb.bit_length() - 1
                if (mm & divmask[i]) != lsb:
                    ok = False
                    break
                mm ^= lsb
            if ok:
                terms = [smooth[i] for i in range(S) if (mask >> i) & 1]
                answers[n - 1] = terms
                break

    # ---- safety fallback for any target still missing ----
    if time.time() > DEADLINE - 2.0:
        return answers

    for n in range(1, N + 1):
        if answers[n - 1] is not None:
            continue
        # simple exhaustive fallback (bounded)
        pool = [x for x in smooth if x <= n]
        found = None
        for k in range(1, min(30, len(pool)) + 1):
            for idxs in combinations(range(len(pool)), k):
                if sum(pool[i] for i in idxs) != n:
                    continue
                ok = True
                for a in range(k):
                    for b in range(k):
                        if a != b and pool[idxs[b]] % pool[idxs[a]] == 0:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    found = [pool[i] for i in idxs]
                    break
            if found is not None:
                break
        if found is None:
            found = [n] if n in pool else [1]
        answers[n - 1] = found

    return answers
# EVOLVE-BLOCK-END
