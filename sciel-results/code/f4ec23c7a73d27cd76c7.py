"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time, random
    from fractions import Fraction

    t0 = time.time()
    deadline = t0 + 82.0

    denoms = list(range(2, 49))
    n = len(denoms)
    inv = [1.0 / d for d in denoms]
    rng = random.Random(20240517)

    best_def = float('inf')
    best_mask = 0

    def fsum(mask):
        s = 0.0
        m = mask
        i = 0
        while m:
            if m & 1:
                s += inv[i]
            i += 1
            m >>= 1
        return s

    def update(mask, s):
        nonlocal best_def, best_mask
        if s < 1.0:
            d = 1.0 - s
            if d < best_def:
                best_def = d
                best_mask = mask

    def local_search(mask, end_time):
        nonlocal best_def, best_mask
        cur = mask
        s = fsum(cur)
        if s >= 1.0:
            return
        update(cur, s)
        while time.time() < end_time:
            d = 1.0 - s
            best_move = None
            bd = d
            for i in range(n):
                if not (cur >> i) & 1:
                    ns = s + inv[i]
                    if ns < 1.0:
                        nd = 1.0 - ns
                        if nd < bd - 1e-16:
                            bd = nd
                            best_move = (cur | (1 << i), ns)
            for i in range(n):
                if (cur >> i) & 1:
                    base = s - inv[i]
                    for j in range(n):
                        if not (cur >> j) & 1:
                            ns = base + inv[j]
                            if ns < 1.0:
                                nd = 1.0 - ns
                                if nd < bd - 1e-16:
                                    bd = nd
                                    best_move = ((cur & ~(1 << i)) | (1 << j), ns)
            if best_move:
                cur, s = best_move
                update(cur, s)
            else:
                kicked = False
                for _ in range(30):
                    if rng.random() < 0.5:
                        i = rng.randrange(n)
                        if not (cur >> i) & 1:
                            ns = s + inv[i]
                            if ns < 1.0:
                                cur |= (1 << i)
                                s = ns
                                kicked = True
                                break
                    else:
                        bits = [i for i in range(n) if (cur >> i) & 1]
                        if bits:
                            i = rng.choice(bits)
                            cur &= ~(1 << i)
                            s -= inv[i]
                            kicked = True
                            break
                if not kicked:
                    break

    # initial random fill to get below 1
    start = 0
    s = 0.0
    order = list(range(n))
    rng.shuffle(order)
    for i in order:
        if s + inv[i] < 1.0:
            start |= (1 << i)
            s += inv[i]
    local_search(start, t0 + 25.0)

    # random restarts to escape local minima
    while time.time() < deadline - 6.0:
        mask = 0
        for i in range(n):
            if rng.random() < 0.5:
                mask |= (1 << i)
        if fsum(mask) < 1.0:
            local_search(mask, min(time.time() + 3.0, deadline - 6.0))

    def exact_sum(mask):
        t = Fraction(0, 1)
        m = mask
        i = 0
        while m:
            if m & 1:
                t += Fraction(1, denoms[i])
            i += 1
            m >>= 1
        return t

    # ensure best_mask is strictly below 1 exactly (float may be optimistic)
    cur = best_mask
    while True:
        s = exact_sum(cur)
        if s < 1:
            break
        rem = None
        for i in range(n):
            if (cur >> i) & 1:
                if rem is None or denoms[i] < denoms[rem]:
                    rem = i
        if rem is None:
            break
        cur &= ~(1 << rem)
    if cur == 0:
        cur = 1

    # exact local polish with Fractions
    it = 0
    improved = True
    while improved and it < 300 and time.time() < deadline - 1.0:
        improved = False
        it += 1
        s = exact_sum(cur)
        d = 1 - s
        best_move = None
        bd = d
        for i in range(n):
            if not (cur >> i) & 1:
                ns = s + Fraction(1, denoms[i])
                if ns < 1:
                    nd = 1 - ns
                    if nd < bd:
                        bd = nd
                        best_move = cur | (1 << i)
        for i in range(n):
            if (cur >> i) & 1:
                base = s - Fraction(1, denoms[i])
                for j in range(n):
                    if not (cur >> j) & 1:
                        ns = base + Fraction(1, denoms[j])
                        if ns < 1:
                            nd = 1 - ns
                            if nd < bd:
                                bd = nd
                                best_move = (cur & ~(1 << i)) | (1 << j)
        if best_move is not None:
            cur = best_move
            improved = True

    # final safety net
    s = exact_sum(cur)
    if s >= 1:
        cur = 0
        s = Fraction(0, 1)
        for i in range(n):
            if s + Fraction(1, denoms[i]) < 1:
                cur |= (1 << i)
                s += Fraction(1, denoms[i])

    result = [i for i in range(n) if (cur >> i) & 1]
    if not result:
        result = [0]
    return result
# EVOLVE-BLOCK-END
