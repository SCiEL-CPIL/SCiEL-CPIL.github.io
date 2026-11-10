def construct():
    import time
    from math import gcd

    N = 512
    start = time.time()
    deadline = start + 145.0  # safety margin under 150s kill

    # Precompute prime factors for each number 1..511
    def prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors

    pf = [set() for _ in range(512)]
    for i in range(1, 512):
        pf[i] = prime_factors(i)

    def score(S):
        return sum(1.0 / (N - a) for a in S)

    def is_coprime_with_all(a, S):
        """Check if a is coprime with all elements in S using prime factor sets."""
        fa = pf[a]
        for b in S:
            if fa & pf[b]:  # intersection of prime factor sets
                return False
        return True

    def greedy(order):
        S = []
        for a in order:
            if is_coprime_with_all(a, S):
                S.append(a)
        return S

    # Initial greedy in descending order
    best = greedy(range(511, 0, -1))
    best_score = score(best)
    best_set = set(best)

    # Local search: drop one, re-greedy
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for x in list(best):
            if time.time() > deadline - 10:
                break
            cand = [a for a in range(511, 0, -1) if a != x]
            S2 = greedy(cand)
            s2 = score(S2)
            if s2 > best_score + 1e-12:
                best = S2
                best_score = s2
                best_set = set(best)
                improved = True
                break

    # Pair swap: remove one, add another (must be coprime with all remaining)
    if time.time() < deadline - 5:
        improved2 = True
        while improved2 and time.time() < deadline - 5:
            improved2 = False
            for x in list(best):
                if time.time() > deadline - 5:
                    break
                remaining = [a for a in best if a != x]
                for y in range(511, 0, -1):
                    if y == x or y in best_set:
                        continue
                    if is_coprime_with_all(y, remaining):
                        S2 = remaining + [y]
                        s2 = score(S2)
                        if s2 > best_score + 1e-12:
                            best = S2
                            best_score = s2
                            best_set = set(best)
                            improved2 = True
                            break
                if improved2:
                    break

    # Final greedy pass to fill in any gaps
    if time.time() < deadline - 2:
        best_set = set(best)
        for a in range(511, 0, -1):
            if a not in best_set and is_coprime_with_all(a, best):
                best.append(a)
                best_set.add(a)
        best_score = score(best)

    return best
