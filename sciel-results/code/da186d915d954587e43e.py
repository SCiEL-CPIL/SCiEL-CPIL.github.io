import time

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin under 150s kill

    N = 512
    # 1. Precompute primes and masks
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False

    primes = [i for i in range(2, N + 1) if is_prime[i]]
    prime_to_idx = {p: i for i, p in enumerate(primes)}

    # For each a in 1..511, compute bitmask of prime factors and weight
    masks = [0] * (N + 1)
    weights = [0.0] * (N + 1)
    for a in range(1, N):
        weights[a] = 1.0 / (N - a)
        mask = 0
        x = a
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                mask |= (1 << prime_to_idx[p])
                while x % p == 0:
                    x //= p
        if x > 1:
            mask |= (1 << prime_to_idx[x])
        masks[a] = mask
    # masks[1] = 0 (1 is coprime to everything)

    # All candidates in descending order of weight (= descending a)
    order = list(range(511, 0, -1))

    def greedy(cand_order):
        S = []
        m_acc = 0
        for a in cand_order:
            if masks[a] & m_acc == 0:
                S.append(a)
                m_acc |= masks[a]
        return S

    def score(S):
        return sum(weights[a] for a in S)

    # 2. Initial greedy descending
    best = greedy(order)
    best_mask = 0
    for a in best:
        best_mask |= masks[a]
    best_score = score(best)

    def time_ok():
        return time.time() < deadline

    # 3. Local search
    improved = True
    while improved and time_ok():
        improved = False
        n = len(best)
        best_set = set(best)

        # Move 1: drop one element, then re-greedy descending on the rest.
        # This can free multiple conflicts at once and admit several high-value
        # composites that a single 1-for-1 swap would miss.
        for i in range(n):
            if not time_ok():
                break
            x = best[i]
            cand = [a for a in order if a != x]
            S2 = greedy(cand)
            s2 = score(S2)
            if s2 > best_score + 1e-12:
                best = S2
                best_mask = 0
                for a in best:
                    best_mask |= masks[a]
                best_score = s2
                improved = True
                break

        if improved or not time_ok():
            continue

        # Move 2: add a number not in best; if it conflicts, drop the conflicts
        # and re-greedy to fill the freed space.
        for a in order:
            if not time_ok():
                break
            if a in best_set:
                continue
            conflicts = [b for b in best if masks[a] & masks[b] != 0]
            if not conflicts:
                # Direct add
                S2 = best + [a]
                s2 = best_score + weights[a]
                if s2 > best_score + 1e-12:
                    best = S2
                    best_mask |= masks[a]
                    best_score = s2
                    improved = True
                    break
                continue
            # Drop conflicts, keep rest, then re-greedy
            drop_set = set(conflicts)
            remaining = [b for b in best if b not in drop_set]
            rem_set = set(remaining)
            cand = [a] + [c for c in order if c not in rem_set and c not in drop_set]
            S2 = greedy(cand)
            s2 = score(S2)
            if s2 > best_score + 1e-12:
                best = S2
                best_mask = 0
                for a2 in best:
                    best_mask |= masks[a2]
                best_score = s2
                improved = True
                break

        if improved or not time_ok():
            continue

        # Move 3: swap one element x in best with one not in best (must be
        # coprime to all others in best).
        for i in range(n):
            if not time_ok():
                break
            x = best[i]
            for a in order:
                if not time_ok():
                    break
                if a in best_set:
                    continue
                # a must be coprime to every element in best except x
                ok = True
                for b in best:
                    if b != x and masks[a] & masks[b] != 0:
                        ok = False
                        break
                if ok:
                    s2 = best_score - weights[x] + weights[a]
                    if s2 > best_score + 1e-12:
                        best = [b for b in best if b != x] + [a]
                        best_mask = 0
                        for a2 in best:
                            best_mask |= masks[a2]
                        best_score = s2
                        improved = True
                        break
            if improved:
                break

    return best
