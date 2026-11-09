import time

def _smooth_set(limit, primes):
    s = set()
    for n in range(2, limit + 1):
        m = n
        for p in primes:
            while m % p == 0:
                m //= p
        if m == 1:
            s.add(n)
    return s

def _find_clique(cands, smooth, size, deadline):
    n = len(cands)
    adj = [set() for _ in range(n)]
    for i in range(n):
        ai = cands[i]
        for j in range(i + 1, n):
            if (ai + cands[j]) in smooth:
                adj[i].add(j)
                adj[j].add(i)
    order = sorted(range(n), key=lambda i: len(adj[i]), reverse=True)
    best = []
    def bt(current, pool):
        nonlocal best
        if time.time() > deadline:
            return
        if len(current) == size:
            best = list(current)
            return
        if len(pool) < size - len(current):
            return
        for k, idx in enumerate(pool):
            if time.time() > deadline:
                return
            if not all(idx in adj[c] for c in current):
                continue
            new_pool = [x for x in pool[k + 1:] if x in adj[idx]]
            current.append(idx)
            bt(current, new_pool)
            current.pop()
            if best:
                return
    bt([], order)
    return best

def construct():
    t0 = time.time()
    deadline = t0 + 75.0
    prime_sets = [
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13],
        [2, 3, 5, 7, 11, 13, 17],
        [2, 3, 5, 7, 11, 13, 17, 19],
    ]
    for ps in prime_sets:
        if time.time() > deadline - 8.0:
            break
        for N in [300, 600, 1200, 2400]:
            if time.time() > deadline - 8.0:
                break
            smooth = _smooth_set(2 * N, ps)
            cands = list(range(1, N + 1))
            res = _find_clique(cands, smooth, 12, deadline - 8.0)
            if res:
                out = [cands[i] for i in res]
                if len(out) == 12 and len(set(out)) == 12:
                    return out
    return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
