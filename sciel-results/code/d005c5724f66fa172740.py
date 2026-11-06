import time

def construct():
    start = time.time()
    HARD = 78.0            # hard deadline (safety margin before 90s kill)
    n = 256
    MAXSIZE = 64

    def deadline_left():
        return HARD - (time.time() - start)

    def is_sidon(S):
        seen = set()
        L = len(S)
        for i in range(L):
            si = S[i]
            for j in range(i, L):
                s = si + S[j]
                if s in seen:
                    return False
                seen.add(s)
        return True

    def diffset(S):
        d = set()
        L = len(S)
        for i in range(L):
            si = S[i]
            for j in range(i + 1, L):
                d.add(S[i] - S[j])
        return d

    def repair(S):
        """Remove minimal elements to make S a Sidon set."""
        S = sorted(S)
        changed = True
        while changed and len(S) > 1:
            changed = False
            seen = {}
            for i in range(len(S)):
                si = S[i]
                for j in range(i, len(S)):
                    s = si + S[j]
                    if s in seen:
                        # conflict between (i,j) and (seen[s])
                        a, b = seen[s]
                        if i == a and j == b:
                            continue
                        # pick a removal candidate among the four involved
                        cands = [a, b, i, j]
                        best = None
                        bestcnt = -1
                        for c in set(cands):
                            cnt = 0
                            sc = S[c]
                            for k in range(len(S)):
                                if k == c:
                                    continue
                                if (sc + S[k]) in seen or (2 * sc) == (sc + S[k]):
                                    cnt += 1
                            if cnt > bestcnt:
                                bestcnt = cnt
                                best = c
                        if best is None:
                            best = a
                        S.pop(best)
                        changed = True
                        break
                    seen[s] = (i, j)
            if changed:
                break
        return S

    def greedy(S, order):
        """Grow S (Sidon) by adding elements from order in sequence."""
        S = list(S)
        seen = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                seen.add(S[i] + S[j])
        for x in order:
            if x in S:
                continue
            if len(S) >= MAXSIZE:
                break
            ok = True
            ns = set()
            for s in S:
                v = x + s
                if v in seen or v in ns:
                    ok = False
                    break
                ns.add(v)
            if ok and (2 * x) in seen:
                ok = False
            if ok:
                S.append(x)
                seen.update(ns)
                seen.add(2 * x)
        return S

    def local_search(S, order):
        """Improve Sidon set size via element swaps."""
        S = list(S)
        seen = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                seen.add(S[i] + S[j])
        improved = True
        guard = 0
        while improved and len(S) < MAXSIZE and guard < 400:
            guard += 1
            improved = False
            for i in range(len(S)):
                old = S[i]
                # remove old
                for j in range(len(S)):
                    if j == i:
                        continue
                    seen.discard(old + S[j])
                seen.discard(2 * old)
                T = S[:i] + S[i + 1:]
                tseen = set(seen)
                # try to add a new element from order
                for x in order:
                    if x in T:
                        continue
                    ok = True
                    ns = set()
                    for s in T:
                        v = x + s
                        if v in tseen or v in ns:
                            ok = False
                            break
                        ns.add(v)
                    if ok and (2 * x) in tseen:
                        ok = False
                    if ok:
                        S = T + [x]
                        seen = tseen | ns | {2 * x}
                        improved = True
                        break
                if improved:
                    break
                # restore old
                for j in range(len(T)):
                    seen.add(old + T[j])
                seen.add(2 * old)
        return S

    def build_one(order):
        S = []
        S = greedy(S, order)
        S = repair(S)
        S = greedy(S, order)
        S = local_search(S, order)
        S = greedy(S, order)
        if len(S) > MAXSIZE:
            S = S[:MAXSIZE]
        return S

    # Sign-split orders:
    # A: ascending  (differences positive)
    # B: descending (differences negative)
    orders = [
        list(range(1, n + 1)),
        list(range(n, 0, -1)),
    ]
    # a couple of extra orderings to try for variety
    extra = []
    # odd/even splits
    extra.append([x for x in range(1, n + 1, 2)] + [x for x in range(2, n + 1, 2)])
    extra.append([x for x in range(2, n + 1, 2)] + [x for x in range(1, n + 1, 2)])
    # block interleaving
    block = []
    for i in range(128):
        block.append(1 + i)
        block.append(n - i)
    extra.append(block)

    bestA, bestB = [1], [2]
    best_total = 2

    # Build A from ascending, B from descending (primary)
    A = build_one(orders[0])
    if deadline_left() < 20:
        B = [2]
        return [A, B] if len(A) + 1 >= best_total else [bestA, bestB]
    B = build_one(orders[1])

    total = len(A) + len(B)
    if total > best_total:
        best_total = total
        bestA, bestB = A, B

    # Try extra orderings if time remains
    for o in extra:
        if deadline_left() < 15:
            break
        C = build_one(o)
        # pair C with the complement-direction order
        comp = orders[1] if o is orders[0] else orders[0]
        if deadline_left() < 10:
            break
        D = build_one(comp)
        t = len(C) + len(D)
        if t > best_total:
            best_total = t
            bestA, bestB = C, D

    # Ensure both nonempty
    if not bestA:
        bestA = [1]
    if not bestB:
        bestB = [2]

    # Final safety: cap sizes
    if len(bestA) > MAXSIZE:
        bestA = bestA[:MAXSIZE]
    if len(bestB) > MAXSIZE:
        bestB = bestB[:MAXSIZE]

    return [bestA, bestB]
