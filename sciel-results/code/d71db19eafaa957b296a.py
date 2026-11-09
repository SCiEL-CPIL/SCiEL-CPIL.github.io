import time
import random
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140.0  # safety margin under 150s

    N = 512
    # weights[a] = 1/(512-a)
    weights = [0.0] * N
    for a in range(1, N):
        weights[a] = 1.0 / (N - a)

    # Prime factors of each a in 1..511
    def prime_factors(n):
        fs = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                fs.add(d)
                n //= d
            d += 1
        if n > 1:
            fs.add(n)
        return fs

    factors = [set() for _ in range(N)]
    for a in range(1, N):
        factors[a] = prime_factors(a)

    # conflict[a] = bitmask of b (1..511) with gcd(a,b)>1
    conflict = [0] * N
    for a in range(1, N):
        mask = 0
        for b in range(a + 1, N):
            if factors[a] & factors[b]:
                mask |= (1 << b)
        conflict[a] = mask
        conflict[b] = 0  # placeholder, set below
    # rebuild symmetrically cleanly
    conflict = [0] * N
    for a in range(1, N):
        for b in range(a + 1, N):
            if factors[a] & factors[b]:
                conflict[a] |= (1 << b)
                conflict[b] |= (1 << a)

    def score_mask(mask):
        s = 0.0
        m = mask
        while m:
            lsb = m & (-m)
            a = lsb.bit_length() - 1
            s += weights[a]
            m ^= lsb
        return s

    # Greedy seed
    def greedy(order):
        mask = 0
        for a in order:
            if (conflict[a] & mask) == 0:
                mask |= (1 << a)
        return mask

    # Local search: best-improvement over moves
    def local_search(mask, deadline):
        best = mask
        best_score = score_mask(mask)
        while time.time() < deadline:
            improved = False
            # Build candidate set of unchosen nodes
            # chosen list
            chosen = []
            m = mask
            while m:
                lsb = m & (-m)
                a = lsb.bit_length() - 1
                chosen.append(a)
                m ^= lsb
            chosen_set = set(chosen)
            unchosen = [a for a in range(1, N) if a not in chosen_set]

            # ADD move: add best x compatible
            add_gain = 0.0
            add_x = -1
            for x in unchosen:
                if (conflict[x] & mask) == 0:
                    if weights[x] > add_gain:
                        add_gain = weights[x]
                        add_x = x
            if add_x > 0:
                mask |= (1 << add_x)
                best = mask
                best_score += add_gain
                improved = True
                continue

            # 1-for-1 swap
            swap_gain = 0.0
            swap_a = -1
            swap_x = -1
            for a in chosen:
                base = mask ^ (1 << a)
                for x in unchosen:
                    if (conflict[x] & base) == 0:
                        g = weights[x] - weights[a]
                        if g > swap_gain:
                            swap_gain = g
                            swap_a = a
                            swap_x = x
            if swap_x > 0:
                mask = (mask ^ (1 << swap_a)) | (1 << swap_x)
                best = mask
                best_score += swap_gain
                improved = True
                continue

            # 1-for-2: remove a, add x,y
            p12_gain = 0.0
            p12_a = -1
            p12_x = -1
            p12_y = -1
            for a in chosen:
                base = mask ^ (1 << a)
                # candidates compatible with base
                cands = [x for x in unchosen if (conflict[x] & base) == 0]
                if len(cands) < 2:
                    continue
                cands.sort(key=lambda t: weights[t], reverse=True)
                cands = cands[:60]  # limit
                for i in range(len(cands)):
                    x = cands[i]
                    for j in range(i + 1, len(cands)):
                        y = cands[j]
                        if (conflict[x] & (1 << y)) == 0:
                            g = weights[x] + weights[y] - weights[a]
                            if g > p12_gain:
                                p12_gain = g
                                p12_a = a
                                p12_x = x
                                p12_y = y
            if p12_x > 0:
                mask = (mask ^ (1 << p12_a)) | (1 << p12_x) | (1 << p12_y)
                best = mask
                best_score += p12_gain
                improved = True
                continue

            # 2-for-1: remove a,b, add x
            p21_gain = 0.0
            p21_a = -1
            p21_b = -1
            p21_x = -1
            nc = len(chosen)
            for i in range(nc):
                a = chosen[i]
                for j in range(i + 1, nc):
                    b = chosen[j]
                    base = mask ^ (1 << a) ^ (1 << b)
                    for x in unchosen:
                        if (conflict[x] & base) == 0:
                            g = weights[x] - weights[a] - weights[b]
                            if g > p21_gain:
                                p21_gain = g
                                p21_a = a
                                p21_b = b
                                p21_x = x
            if p21_x > 0:
                mask = (mask ^ (1 << p21_a)) ^ (1 << p21_b) | (1 << p21_x)
                best = mask
                best_score += p21_gain
                improved = True
                continue

            # 2-for-2: remove a,b, add x,y
            p22_gain = 0.0
            p22_a = -1
            p22_b = -1
            p22_x = -1
            p22_y = -1
            for i in range(nc):
                a = chosen[i]
                for j in range(i + 1, nc):
                    b = chosen[j]
                    base = mask ^ (1 << a) ^ (1 << b)
                    cands = [x for x in unchosen if (conflict[x] & base) == 0]
                    if len(cands) < 2:
                        continue
                    cands.sort(key=lambda t: weights[t], reverse=True)
                    cands = cands[:40]
                    for k in range(len(cands)):
                        x = cands[k]
                        for l in range(k + 1, len(cands)):
                            y = cands[l]
                            if (conflict[x] & (1 << y)) == 0:
                                g = weights[x] + weights[y] - weights[a] - weights[b]
                                if g > p22_gain:
                                    p22_gain = g
                                    p22_a = a
                                    p22_b = b
                                    p22_x = x
                                    p22_y = y
            if p22_x > 0:
                mask = (mask ^ (1 << p22_a)) ^ (1 << p22_b) | (1 << p22_x) | (1 << p22_y)
                best = mask
                best_score += p22_gain
                improved = True
                continue

            if not improved:
                break

        return best, best_score

    # Multiple seeds
    desc_order = list(range(N - 1, 0, -1))
    best_mask = greedy(desc_order)
    best_score = score_mask(best_mask)

    deadline_inner = DEADLINE - 5.0
    best_mask, best_score = local_search(best_mask, deadline_inner)

    # Try random seeds if time remains
    rng = random.Random(12345)
    n_seeds = 8
    for s in range(n_seeds):
        if time.time() > deadline_inner:
            break
        order = list(range(1, N))
        # random weighted shuffle: bias toward high weight
        order.sort(key=lambda a: -weights[a] + rng.random() * 0.5)
        m = greedy(order)
        m, sc = local_search(m, deadline_inner)
        if sc > best_score:
            best_score = sc
            best_mask = m

    # Final: ensure we return a valid list
    result = []
    m = best_mask
    while m:
        lsb = m & (-m)
        a = lsb.bit_length() - 1
        result.append(a)
        m ^= lsb

    return result
