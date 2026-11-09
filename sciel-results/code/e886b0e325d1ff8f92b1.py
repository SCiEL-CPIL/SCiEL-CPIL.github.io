import time
import random

def construct():
    _start = time.time()
    _deadline = _start + 55.0

    def _is_done():
        return time.time() - _start > _deadline

    def _compute_sums_by_k(A):
        n = len(A)
        skb = [set() for _ in range(n + 1)]
        skb[0] = {0}
        for a in A:
            for k in range(n - 1, -1, -1):
                if skb[k]:
                    skb[k + 1].update(s + a for s in skb[k])
        return skb

    def _is_valid(skb):
        n = len(skb) - 1
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                if skb[i] & skb[j]:
                    return False
        return True

    def _try_add(A, skb, candidate):
        n = len(A)
        new_skb = [set(s) for s in skb]
        for k in range(n, -1, -1):
            if new_skb[k]:
                new_skb[k + 1].update(s + candidate for s in new_skb[k])
        nn = n + 1
        for i in range(nn + 1):
            for j in range(i + 1, nn + 1):
                if new_skb[i] & new_skb[j]:
                    return False, None
        return True, new_skb

    best = None

    # Greedy small to large
    A = []
    skb = [{0}]
    for c in range(1, 257):
        if len(A) >= 60:
            break
        ok, new_skb = _try_add(A, skb, c)
        if ok:
            A.append(c)
            skb = new_skb
    if best is None or len(A) > len(best):
        best = list(A)

    if _is_done() or len(best) >= 60:
        return [int(x) for x in best]

    # Greedy large to small
    A = []
    skb = [{0}]
    for c in range(256, 0, -1):
        if len(A) >= 60:
            break
        ok, new_skb = _try_add(A, skb, c)
        if ok:
            A.append(c)
            skb = new_skb
    if len(A) > len(best):
        best = list(A)

    if _is_done() or len(best) >= 60:
        return [int(x) for x in best]

    # Random restarts
    random.seed(777)
    for _ in range(1000):
        if _is_done() or len(best) >= 60:
            break
        seed_size = random.randint(3, 15)
        pool = list(range(1, 257))
        random.shuffle(pool)
        A = pool[:seed_size]
        skb = _compute_sums_by_k(A)
        if not _is_valid(skb):
            continue
        for c in range(1, 257):
            if len(A) >= 60:
                break
            if c in A:
                continue
            ok, new_skb = _try_add(A, skb, c)
            if ok:
                A.append(c)
                skb = new_skb
        if len(A) > len(best):
            best = list(A)

    if best is None:
        best = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    return [int(x) for x in best]
