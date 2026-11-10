def construct():
    import time
    from math import gcd
    from fractions import Fraction
    N = 512
    start = time.time()
    deadline = start + 145.0  # safety margin under the 150s kill

    # Precompute prime factors for each number 1..511
    prime_factors = [set() for _ in range(512)]
    for p in range(2, 512):
        for m in range(p, 512, p):
            prime_factors[m].add(p)

    # Precompute conflict masks using prime factor overlap
    # mask[a] has bit b set if gcd(a, b) != 1 (i.e., they share a prime factor)
    mask = [0] * 512
    for a in range(1, 512):
        if not prime_factors[a]:
            continue
        m = 0
        for pf in prime_factors[a]:
            # All multiples of pf in 1..511 conflict with a
            for b in range(pf, 512, pf):
                if b != a:
                    m |= (1 << b)
        mask[a] = m

    def score_float(S):
        return sum(1.0 / (N - a) for a in S)

    def score_frac(S):
        return sum(Fraction(1, N - a) for a in S)

    def greedy(order, chosen_mask=0):
        S = []
        cm = chosen_mask
        for a in order:
            if (cm & mask[a]) == 0:
                S.append(a)
                cm |= (1 << a)
        return S, cm

    # Initial greedy in descending order (prefers large a -> large 1/(512-a))
    best, best_mask = greedy(range(511, 0, -1))
    best_score = score_float(best)
    best_frac = score_frac(best)

    def is_valid_set(S):
        """Check pairwise coprime using prime factors"""
        used_primes = set()
        for a in S:
            for pf in prime_factors[a]:
                if pf in used_primes:
                    return False
                used_primes.add(pf)
        return True

    # Local search phase 1: drop one chosen element, then re-greedy descending
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in list(best):
            # Greedy from scratch excluding x
            order = [a for a in range(511, 0, -1) if a != x]
            S2, _ = greedy(order)
            s2 = score_float(S2)
            if s2 > best_score + 1e-12:
                best = S2
                best_mask = 0
                for a in best:
                    best_mask |= (1 << a)
                best_score = s2
                best_frac = score_frac(best)
                improved = True
                break

    # Local search phase 2: 1-for-1 swap using exact rational comparison
    if time.time() < deadline:
        improved2 = True
        while improved2 and time.time() < deadline:
            improved2 = False
            best_set = set(best)
            for x in list(best):
                if time.time() >= deadline:
                    break
                rest_mask = best_mask & ~(1 << x)
                best_without_x = [a for a in best if a != x]
                # Find best y to swap in
                best_y = None
                best_y_gain = None
                for y in range(511, 0, -1):
                    if y in best_set:
                        continue
                    if (rest_mask & mask[y]) != 0:
                        continue
                    # y is coprime to all in best_without_x
                    gain = Fraction(1, N - y) - Fraction(1, N - x)
                    if best_y_gain is None or gain > best_y_gain:
                        best_y_gain = gain
                        best_y = y
                if best_y is not None and best_y_gain > 0:
                    new_best = best_without_x + [best_y]
                    new_score = score_float(new_best)
                    if new_score > best_score + 1e-12:
                        best = new_best
                        best_mask = 0
                        for a in best:
                            best_mask |= (1 << a)
                        best_score = new_score
                        best_frac = score_frac(best)
                        improved2 = True
                        break

    # Local search phase 3: 2-for-1 swap (remove two, add one)
    if time.time() < deadline:
        best_set = set(best)
        n_best = len(best)
        for i in range(n_best):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n_best):
                if time.time() >= deadline:
                    break
                x, y = best[i], best[j]
                rest_mask = best_mask & ~(1 << x) & ~(1 << y)
                best_without_xy = [a for a in best if a != x and a != y]
                # Find best z to add
                best_z = None
                best_z_gain = None
                for z in range(511, 0, -1):
                    if z in best_set:
                        continue
                    if (rest_mask & mask[z]) != 0:
                        continue
                    gain = Fraction(1, N - z) - Fraction(1, N - x) - Fraction(1, N - y)
                    if best_z_gain is None or gain > best_z_gain:
                        best_z_gain = gain
                        best_z = z
                if best_z is not None and best_z_gain > 0:
                    new_best = best_without_xy + [best_z]
                    new_score = score_float(new_best)
                    if new_score > best_score + 1e-12:
                        best = new_best
                        best_mask = 0
                        for a in best:
                            best_mask |= (1 << a)
                        best_score = new_score
                        best_frac = score_frac(best)
                        best_set = set(best)
                        n_best = len(best)
                        break
            else:
                continue
            break

    # Final polish: try all single additions (if any slot is free)
    if time.time() < deadline:
        for z in range(511, 0, -1):
            if z in set(best):
                continue
            if (best_mask & mask[z]) != 0:
                continue
            new_best = best + [z]
            new_score = score_float(new_best)
            if new_score > best_score + 1e-12:
                best = new_best
                best_mask |= (1 << z)
                best_score = new_score
                best_frac = score_frac(best)

    return best
