def construct():
    import time
    from math import gcd
    import random
    N = 512
    start = time.time()
    deadline = start + 140.0

    # Precompute prime factors as bitmask for each number 1..511
    primes = []
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, N + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, N + 1, i):
                is_prime[j] = False

    prime_index = {p: i for i, p in enumerate(primes)}
    nprimes = len(primes)

    # For each a in 1..511, compute bitmask of prime factors
    mask = [0] * (N + 1)
    for a in range(2, N):
        m = a
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                mask[a] |= (1 << prime_index[p])
                while m % p == 0:
                    m //= p
        if m > 1:
            mask[a] |= (1 << prime_index[m])
    # mask[1] = 0

    weights = [0.0] * (N + 1)
    for a in range(1, N):
        weights[a] = 1.0 / (N - a)

    def score(S):
        return sum(weights[a] for a in S)

    def greedy(order):
        S = []
        mask_acc = 0
        for a in order:
            if mask[a] & mask_acc == 0:
                S.append(a)
                mask_acc |= mask[a]
        return S

    # Initial greedy descending
    best = greedy(range(511, 0, -1))
    best_mask = 0
    for a in best:
        best_mask |= mask[a]
    best_score = score(best)

    def time_ok():
        return time.time() < deadline

    # Precompute weight-descending order for greedy restarts
    w_desc = sorted(range(511, 0, -1), key=lambda a: -weights[a])

    # Precompute conflict matrix: coprime[a][b] = True if mask[a] & mask[b] == 0
    # Use a list of bytearrays for memory efficiency and fast access
    coprime = [bytearray(N) for _ in range(N)]
    for a in range(1, N):
        for b in range(1, N):
            if mask[a] & mask[b] == 0:
                coprime[a][b] = 1

    improved = True
    while improved and time_ok():
        improved = False
        n = len(best)
        best_set = set(best)

        # Move 1: drop one element, then re-greedy descending on remaining candidates
        for i in range(n):
            if not time_ok():
                break
            x = best[i]
            cand_order = [a for a in w_desc if a != x]
            S2 = greedy(cand_order)
            s2 = score(S2)
            if s2 > best_score + 1e-12:
                best = S2
                best_mask = 0
                for a in best:
                    best_mask |= mask[a]
                best_score = s2
                improved = True
                break

        if improved or not time_ok():
            continue

        # Move 2: try adding a number not in best, and dropping all conflicts
        for a in w_desc:
            if not time_ok():
                break
            if a in best_set:
                continue
            # Find conflicts in best using precomputed coprime matrix
            conflicts = [b for b in best if coprime[a][b] == 0]
            if not conflicts:
                # Can add directly
                S2 = best + [a]
                s2 = best_score + weights[a]
                if s2 > best_score + 1e-12:
                    best = S2
                    best_mask |= mask[a]
                    best_score = s2
                    improved = True
                    break
                continue
            # Try dropping conflicts and adding a, then re-greedy
            drop_set = set(conflicts)
            remaining = [b for b in best if b not in drop_set]
            remaining_set = set(remaining)
            cand_order = [a] + [c for c in w_desc if c != a and c not in remaining_set and c not in drop_set]
            S2 = greedy(cand_order)
            s2 = score(S2)
            if s2 > best_score + 1e-12:
                best = S2
                best_mask = 0
                for a2 in best:
                    best_mask |= mask[a2]
                best_score = s2
                improved = True
                break

        if improved or not time_ok():
            continue

        # Move 3: swap one element x in best with one not in best
        for i in range(n):
            if not time_ok():
                break
            x = best[i]
            best_minus_x = [b for b in best if b != x]
            best_minus_x_set = set(best_minus_x)
            for a in w_desc:
                if not time_ok():
                    break
                if a in best_set:
                    continue
                # Check if a is coprime to all in best_minus_x
                ok = True
                for b in best_minus_x:
                    if coprime[a][b] == 0:
                        ok = False
                        break
                if ok:
                    S2 = best_minus_x + [a]
                    s2 = best_score - weights[x] + weights[a]
                    if s2 > best_score + 1e-12:
                        best = S2
                        best_mask = 0
                        for a2 in best:
                            best_mask |= mask[a2]
                        best_score = s2
                        improved = True
                        break
            if improved:
                break

        if improved or not time_ok():
            continue

        # Move 4: remove two elements, re-greedy
        for i in range(n):
            if not time_ok():
                break
            for j in range(i + 1, n):
                if not time_ok():
                    break
                x, y = best[i], best[j]
                removed = {x, y}
                cand_order = [a for a in w_desc if a not in removed]
                S2 = greedy(cand_order)
                s2 = score(S2)
                if s2 > best_score + 1e-12:
                    best = S2
                    best_mask = 0
                    for a in best:
                        best_mask |= mask[a]
                    best_score = s2
                    improved = True
                    break
            if improved:
                break

        if improved or not time_ok():
            continue

        # Move 5: 2-swap - drop two from best, add two not in best
        found = False
        for i in range(n):
            if not time_ok():
                break
            for j in range(i + 1, n):
                if not time_ok():
                    break
                x, y = best[i], best[j]
                best_minus_xy = [b for b in best if b != x and b != y]
                best_minus_xy_set = set(best_minus_xy)
                # Find top candidates not in best that are coprime to all best_minus_xy
                candidates = []
                for a in w_desc:
                    if a in best_set:
                        continue
                    ok = True
                    for b in best_minus_xy:
                        if coprime[a][b] == 0:
                            ok = False
                            break
                    if ok:
                        candidates.append(a)
                        if len(candidates) >= 2:
                            break
                if len(candidates) >= 2:
                    a, b = candidates[0], candidates[1]
                    if coprime[a][b] == 0:
                        S2 = best_minus_xy + [a, b]
                        s2 = best_score - weights[x] - weights[y] + weights[a] + weights[b]
                        if s2 > best_score + 1e-12:
                            best = S2
                            best_mask = 0
                            for a2 in best:
                                best_mask |= mask[a2]
                            best_score = s2
                            improved = True
                            found = True
                            break
            if found:
                break

        if improved or not time_ok():
            continue

        # Move 6: Perturbed re-greedy - shuffle a few elements in w_desc and re-greedy
        if time_ok():
            # Try a few random perturbations
            for _ in range(3):
                if not time_ok():
                    break
                perm = w_desc[:]
                # Swap a random pair
                idx1 = random.randint(0, len(perm) - 1)
                idx2 = random.randint(0, len(perm) - 1)
                perm[idx1], perm[idx2] = perm[idx2], perm[idx1]
                S2 = greedy(perm)
                s2 = score(S2)
                if s2 > best_score + 1e-12:
                    best = S2
                    best_mask = 0
                    for a in best:
                        best_mask |= mask[a]
                    best_score = s2
                    improved = True
                    break

    # Final polish: try adding any remaining coprime number (tighter check)
    best_set = set(best)
    for a in w_desc:
        if not time_ok():
            break
        if a in best_set:
            continue
        if mask[a] & best_mask == 0:
            best.append(a)
            best_mask |= mask[a]
            best_score += weights[a]
            best_set.add(a)

    # Final exact polish: one more pass of all moves with tighter tolerance
    if time_ok():
        best_set = set(best)
        n = len(best)
        # Try single additions
        for a in w_desc:
            if not time_ok():
                break
            if a in best_set:
                continue
            if mask[a] & best_mask == 0:
                best.append(a)
                best_mask |= mask[a]
                best_score += weights[a]
                best_set.add(a)
        # Try single swaps
        n = len(best)
        best_set = set(best)
        for i in range(n):
            if not time_ok():
                break
            x = best[i]
            best_minus_x = [b for b in best if b != x]
            for a in w_desc:
                if not time_ok():
                    break
                if a in best_set:
                    continue
                ok = True
                for b in best_minus_x:
                    if coprime[a][b] == 0:
                        ok = False
                        break
                if ok:
                    s2 = best_score - weights[x] + weights[a]
                    if s2 > best_score + 1e-12:
                        best = best_minus_x + [a]
                        best_mask = 0
                        for a2 in best:
                            best_mask |= mask[a2]
                        best_score = s2
                        best_set = set(best)
                        break

    return best
