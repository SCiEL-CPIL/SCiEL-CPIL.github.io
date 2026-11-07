import time
import random

def _is_valid(Sset):
    # Full check: no distinct a,b,c in S with a+b,a+c,b+c all in S.
    elems = sorted(Sset)
    n = len(elems)
    for i in range(n):
        a = elems[i]
        for j in range(i + 1, n):
            b = elems[j]
            ab = a + b
            if ab not in Sset:
                continue
            cnt = 0
            for k in range(j + 1, n):
                c = elems[k]
                if (a + c) in Sset and (b + c) in Sset:
                    cnt += 1
                    if cnt > 2:
                        return False
    return True

def construct():
    start = time.time()
    deadline = start + 82.0   # safety margin under 90s wall-clock

    # Start from the strong base: {64..128}, size 65.
    S = set(range(64, 129))
    best = set(S)

    # Greedily add each small element 1..63 if it keeps validity.
    for cand in range(1, 64):
        if cand in S:
            continue
        trial = S | {cand}
        if _is_valid(trial):
            S = trial
    if len(S) > len(best):
        best = set(S)

    # Randomized local search with fast incremental validity.
    # For adding x to S, only triples where x is a member need checking.
    # Represent the "badness" incrementally by counting, for each existing
    # pair (a,b) in S with a+b in S, how many c in S satisfy c+a,c+b in S.
    # A triple is forbidden iff some pair has >=3 such c.

    def _add_ok(Sset, x):
        # Check if adding x (not in Sset) creates a forbidden triple.
        # Only triples containing x matter.
        # Case A: x is the largest (c): need pair (a,b) in Sset with
        #   a+b in Sset and a+x in Sset and b+x in Sset.
        #   But that would make (a,b,x) a triple only if a+b in Sset AND
        #   a+x in Sset AND b+x in Sset. Since a+x and b+x must be in Sset
        #   already (x not added yet), they must already be in Sset.
        # Case B: x is the middle (b): need a<x<c, a+x in Sset, b+x=x+x? no.
        #   triple (a,x,c): a+x in Sset, a+c in Sset, x+c in Sset.
        # Case C: x is the smallest (a): triple (x,b,c): x+b in Sset, x+c in Sset, b+c in Sset.
        # Simplest: just check all triples containing x by scanning.
        elems = sorted(Sset)
        n = len(elems)
        # x as c: pairs (a,b) in Sset
        for i in range(n):
            a = elems[i]
            ax = a + x
            if ax not in Sset:
                continue
            for j in range(i + 1, n):
                b = elems[j]
                if a + b in Sset and b + x in Sset:
                    return False
        # x as b (middle): a < x < c
        for i in range(n):
            a = elems[i]
            if a >= x:
                break
            ax = a + x
            if ax not in Sset:
                continue
            for k in range(n):
                c = elems[k]
                if c <= x:
                    continue
                if a + c in Sset and x + c in Sset:
                    return False
        # x as a (smallest): x < b < c
        for j in range(n):
            b = elems[j]
            if b <= x:
                continue
            xb = x + b
            if xb not in Sset:
                continue
            for k in range(j + 1, n):
                c = elems[k]
                if x + c in Sset and b + c in Sset:
                    return False
        return True

    # Local search: try single additions repeatedly (randomized order),
    # and 2-for-1 swaps when stuck.
    rng = random.Random(12345)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        cands = list(range(1, 64))
        rng.shuffle(cands)
        for x in cands:
            if x in S:
                continue
            if _add_ok(S, x):
                S = S | {x}
                improved = True
                if len(S) > len(best):
                    best = set(S)
                if time.time() > deadline:
                    break
        # 2-for-1: add two small elements
        if time.time() < deadline:
            cands2 = [x for x in range(1, 64) if x not in S]
            rng.shuffle(cands2)
            for idx in range(len(cands2)):
                if time.time() > deadline:
                    break
                x = cands2[idx]
                if _add_ok(S, x):
                    S = S | {x}
                    if len(S) > len(best):
                        best = set(S)
                    improved = True

    if len(S) > len(best):
        best = set(S)

    # Safety: ensure best is valid; if somehow not, fall back to base.
    if not _is_valid(best):
        best = set(range(64, 129))

    result = sorted(best)
    result = [int(x) for x in result]
    return result
