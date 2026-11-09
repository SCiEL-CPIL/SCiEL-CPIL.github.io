import time
import random
from math import gcd
from fractions import Fraction

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

    def exact_score(mask):
        """Compute exact rational score for a mask."""
        s = Fraction(0)
        m = mask
        while m:
            lsb = m & (-m)
            a = lsb.bit_length() - 1
            s += Fraction(1, N - a)
            m ^= lsb
        return s

    # Local search: best-improvement over moves
    def local_search(mask, deadline):
        best = mask
        best_score = score_mask(mask)
        while time.time() < deadline:
            improved = False
            chosen = []
            m = mask
            while m:
                lsb = m & (-m)
                a = lsb.bit_length() - 1
                chosen.append(a)
                m ^= lsb
            chosen_set = set(chosen)
            unchosen = [a for a in range(1, N) if a not in chosen_set]
            # Pre-sort unchosen by weight descending for faster candidate selection
            unchosen_sorted = sorted(unchosen, key=lambda t: weights[t], reverse=True)

            # ADD move: add best x compatible
            add_gain = 0.0
            add_x = -1
            for x in unchosen_sorted:
                if (conflict[x] & mask) == 0:
                    if weights[x] > add_gain:
                        add_gain = weights[x]
                        add_x = x
                        break  # already sorted, first compatible is best
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
                for x in unchosen_sorted:
                    if (conflict[x] & base) == 0:
                        g = weights[x] - weights[a]
                        if g > swap_gain:
                            swap_gain = g
                            swap_a = a
                            swap_x = x
                        break  # sorted, first compatible gives max for this a
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
                cands = [x for x in unchosen_sorted if (conflict[x] & base) == 0]
                if len(cands) < 2:
                    continue
                cands = cands[:50]
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

            # 1-for-3: remove a, add x,y,z
            p13_gain = 0.0
            p13_a = -1
            p13_x = -1
            p13_y = -1
            p13_z = -1
            for a in chosen:
                base = mask ^ (1 << a)
                cands = [x for x in unchosen_sorted if (conflict[x] & base) == 0]
                if len(cands) < 3:
                    continue
                cands = cands[:35]
                for i in range(len(cands)):
                    x = cands[i]
                    for j in range(i + 1, len(cands)):
                        y = cands[j]
                        if (conflict[x] & (1 << y)) != 0:
                            continue
                        for k in range(j + 1, len(cands)):
                            z = cands[k]
                            if (conflict[x] & (1 << z)) == 0 and (conflict[y] & (1 << z)) == 0:
                                g = weights[x] + weights[y] + weights[z] - weights[a]
                                if g > p13_gain:
                                    p13_gain = g
                                    p13_a = a
                                    p13_x = x
                                    p13_y = y
                                    p13_z = z
            if p13_x > 0:
                mask = (mask ^ (1 << p13_a)) | (1 << p13_x) | (1 << p13_y) | (1 << p13_z)
                best = mask
                best_score += p13_gain
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
                    for x in unchosen_sorted:
                        if (conflict[x] & base) == 0:
                            g = weights[x] - weights[a] - weights[b]
                            if g > p21_gain:
                                p21_gain = g
                                p21_a = a
                                p21_b = b
                                p21_x = x
                            break  # sorted, first compatible is best for this (a,b)
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
                    cands = [x for x in unchosen_sorted if (conflict[x] & base) == 0]
                    if len(cands) < 2:
                        continue
                    cands = cands[:35]
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

    # Final high-precision polish: use exact rational arithmetic to verify
    # that no single-swap or add/remove move can improve the score.
    def polish(mask, deadline):
        """Verify local optimality with exact rational scores."""
        chosen = []
        m = mask
        while m:
            lsb = m & (-m)
            a = lsb.bit_length() - 1
            chosen.append(a)
            m ^= lsb
        chosen_set = set(chosen)
        unchosen = [a for a in range(1, N) if a not in chosen_set]
        unchosen_sorted = sorted(unchosen, key=lambda t: weights[t], reverse=True)

        improved = True
        while improved and time.time() < deadline:
            improved = False

            # Check ADD: can we add any unchosen element?
            for x in unchosen_sorted:
                if (conflict[x] & mask) == 0:
                    # Adding x always increases score (weight > 0)
                    mask |= (1 << x)
                    chosen.append(x)
                    chosen_set.add(x)
                    unchosen.remove(x)
                    unchosen_sorted.remove(x)
                    improved = True
                    break

            if improved:
                continue

            # Check 1-for-1 swap using exact rational
            best_gain = Fraction(0)
            best_a = -1
            best_x = -1
            for a in chosen:
                base = mask ^ (1 << a)
                for x in unchosen_sorted:
                    if (conflict[x] & base) == 0:
                        g = Fraction(1, N - x) - Fraction(1, N - a)
                        if g > best_gain:
                            best_gain = g
                            best_a = a
                            best_x = x
                        break  # sorted by weight, first compatible is best for this a

            if best_x > 0:
                mask = (mask ^ (1 << best_a)) | (1 << best_x)
                chosen.remove(best_a)
                chosen_set.discard(best_a)
                chosen.append(best_x)
                chosen_set.add(best_x)
                unchosen.remove(best_x)
                unchosen_sorted.remove(best_x)
                unchosen.append(best_a)
                unchosen_sorted.append(best_a)
                improved = True
                continue

            # Check 1-for-2 using exact rational
            best_gain = Fraction(0)
            best_a = -1
            best_x = -1
            best_y = -1
            for a in chosen:
                base = mask ^ (1 << a)
                cands = [x for x in unchosen_sorted if (conflict[x] & base) == 0]
                if len(cands) < 2:
                    continue
                cands = cands[:40]
                for i in range(len(cands)):
                    x = cands[i]
                    for j in range(i + 1, len(cands)):
                        y = cands[j]
                        if (conflict[x] & (1 << y)) == 0:
                            g = Fraction(1, N - x) + Fraction(1, N - y) - Fraction(1, N - a)
                            if g > best_gain:
                                best_gain = g
                                best_a = a
                                best_x = x
                                best_y = y

            if best_x > 0:
                mask = (mask ^ (1 << best_a)) | (1 << best_x) | (1 << best_y)
                chosen.remove(best_a)
                chosen_set.discard(best_a)
                chosen.extend([best_x, best_y])
                chosen_set.update([best_x, best_y])
                unchosen.remove(best_x)
                unchosen.remove(best_y)
                unchosen_sorted.remove(best_x)
                unchosen_sorted.remove(best_y)
                unchosen.append(best_a)
                unchosen_sorted.append(best_a)
                improved = True
                continue

        return mask

    # 2-for-3: remove a,b, add x,y,z
    def try_2_for_3(mask, chosen, unchosen_sorted, deadline):
        nc = len(chosen)
        best_gain = 0.0
        best_a = -1
        best_b = -1
        best_x = -1
        best_y = -1
        best_z = -1
        for i in range(nc):
            a = chosen[i]
            for j in range(i + 1, nc):
                b = chosen[j]
                base = mask ^ (1 << a) ^ (1 << b)
                cands = [x for x in unchosen_sorted if (conflict[x] & base) == 0]
                if len(cands) < 3:
                    continue
                cands = cands[:30]
                for k in range(len(cands)):
                    x = cands[k]
                    for l in range(k + 1, len(cands)):
                        y = cands[l]
                        if (conflict[x] & (1 << y)) != 0:
                            continue
                        for m_idx in range(l + 1, len(cands)):
                            z = cands[m_idx]
                            if (conflict[x] & (1 << z)) == 0 and (conflict[y] & (1 << z)) == 0:
                                g = weights[x] + weights[y] + weights[z] - weights[a] - weights[b]
                                if g > best_gain:
                                    best_gain = g
                                    best_a = a
                                    best_b = b
                                    best_x = x
                                    best_y = y
                                    best_z = z
        if best_x > 0:
            new_mask = (mask ^ (1 << best_a)) ^ (1 << best_b) | (1 << best_x) | (1 << best_y) | (1 << best_z)
            return new_mask, best_gain
        return None, 0.0

    # Multiple seeds with diverse strategies
    desc_order = list(range(N - 1, 0, -1))
    best_mask = greedy(desc_order)
    best_score = score_mask(best_mask)

    deadline_inner = DEADLINE - 10.0  # reserve 10s for polish
    best_mask, best_score = local_search(best_mask, deadline_inner)

    # Try diverse seeds if time remains
    rng = random.Random(12345)
    n_seeds = 12
    for s in range(n_seeds):
        if time.time() > deadline_inner:
            break
        order = list(range(1, N))
        if s % 3 == 0:
            # Weighted random: bias toward high weight
            order.sort(key=lambda a: -weights[a] + rng.random() * 0.5)
        elif s % 3 == 1:
            # Random permutation
            rng.shuffle(order)
        else:
            # Alternate high/low with noise
            order.sort(key=lambda a: (-weights[a] if (a + s) % 2 == 0 else weights[a]) + rng.random() * 0.3)
        m = greedy(order)
        m, sc = local_search(m, deadline_inner)
        if sc > best_score:
            best_score = sc
            best_mask = m

    # Final polish pass with exact rational arithmetic
    if time.time() < DEADLINE - 2.0:
        polish_deadline = DEADLINE - 2.0
        polished_mask = polish(best_mask, polish_deadline)
        polished_score = score_mask(polished_mask)
        if polished_score >= best_score:
            best_mask = polished_mask
            best_score = polished_score

    # Final: ensure we return a valid list
    result = []
    m = best_mask
    while m:
        lsb = m & (-m)
        a = lsb.bit_length() - 1
        result.append(a)
        m ^= lsb

    return result
