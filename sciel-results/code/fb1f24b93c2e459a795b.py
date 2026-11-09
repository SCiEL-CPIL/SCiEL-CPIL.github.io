import time

N = 128
DEADLINE = time.time() + 80


def is_valid(S):
    Sset = set(S)
    L = sorted(S)
    n = len(L)
    for i in range(n):
        a = L[i]
        for j in range(i + 1, n):
            b = L[j]
            if (a + b) not in Sset:
                continue
            for k in range(j + 1, n):
                c = L[k]
                if (a + c) in Sset and (b + c) in Sset:
                    return False
    return True


def can_add(Sset, x):
    for p in Sset:
        if p == x:
            continue
        xp = x + p
        if xp not in Sset and xp != x:
            continue
        for q in Sset:
            if q == x or q == p:
                continue
            xq = x + q
            if xq not in Sset and xq != x:
                continue
            pq = p + q
            if (pq in Sset) or (pq == x):
                return False
    return True


def construct():
    bestset = set(range(N // 2 + 1, N + 1))
    # greedy add small elements
    for x in range(1, N // 2 + 1):
        if time.time() > DEADLINE:
            break
        if x not in bestset and can_add(bestset, x):
            bestset.add(x)
    # local search
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        curset = set(bestset)
        # adds
        for x in range(1, N + 1):
            if time.time() > DEADLINE:
                break
            if x in curset:
                continue
            if can_add(curset, x):
                curset.add(x)
                improved = True
        # swaps
        for r in list(curset):
            if time.time() > DEADLINE:
                break
            curset.remove(r)
            added = False
            for a in range(1, N + 1):
                if a in curset:
                    continue
                if can_add(curset, a):
                    curset.add(a)
                    added = True
                    improved = True
                    break
            if not added:
                curset.add(r)
        bestset = curset
    result = [int(v) for v in bestset]
    if not is_valid(result):
        return list(range(N // 2 + 1, N + 1))
    return result
