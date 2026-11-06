def construct():
    import time
    deadline = time.time() + 110.0
    N = 511

    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    pindex = {p: i for i, p in enumerate(primes)}

    mult_mask = {}
    for p in primes:
        m = 0
        for b in range(p, N + 1, p):
            m |= (1 << b)
        mult_mask[p] = m

    pmask = [0] * (N + 1)
    conflict = [0] * (N + 1)
    for a in range(2, N + 1):
        x = a
        pm = 0
        cm = 0
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                pm |= (1 << pindex[p])
                cm |= mult_mask[p]
                while x % p == 0:
                    x //= p
        if x > 1:
            pm |= (1 << pindex[x])
            cm |= mult_mask[x]
        pmask[a] = pm
        conflict[a] = cm

    w = [0.0] * (N + 1)
    for a in range(1, N + 1):
        w[a] = 1.0 / (512 - a)

    S = set()
    S_mask = 0
    for a in range(N, 0, -1):
        if (conflict[a] & S_mask) == 0:
            S_mask |= (1 << a)
            S.add(a)

    def score():
        return sum(w[a] for a in S)

    best = score()

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # additions
        for a in range(N, 0, -1):
            if not (S_mask >> a) & 1 and (conflict[a] & S_mask) == 0:
                S_mask |= (1 << a)
                S.add(a)
                improved = True
        # swaps
        for x in list(S):
            if time.time() >= deadline:
                break
            xm = 1 << x
            if not (S_mask & xm):
                continue
            rest = S_mask ^ xm
            for a in range(N, 0, -1):
                if (S_mask >> a) & 1:
                    continue
                if w[a] <= w[x]:
                    continue
                if (conflict[a] & rest) == 0:
                    S_mask = rest | (1 << a)
                    S.discard(x)
                    S.add(a)
                    improved = True
                    break
        cur = score()
        if cur > best:
            best = cur

    return list(S)
