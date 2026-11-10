import time
import random

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
    masks_local = masks  # local ref for speed

    def greedy(cand_order):
        S = []
        m_acc = 0
        for a in cand_order:
            if masks_local[a] & m_acc == 0:
                S.append(a)
                m_acc |= masks_local[a]
        return S

    def score(S):
        return sum(weights[a] for a in S)

    def time_ok():
        return time.time() < deadline

    def local_search(init_set):
        """Run local search starting from init_set, return (best_set, best_score)."""
        best = list(init_set)
        best_score = score(best)

        improved = True
        while improved and time_ok():
            improved = False
            n = len(best)
            best_set = set(best)

            # Move 1: drop one element, then re-greedy descending on the rest.
            for i in range(n):
                if not time_ok():
                    break
                x = best[i]
                cand = [a for a in order if a != x]
                S2 = greedy(cand)
                s2 = score(S2)
                if s2 > best_score + 1e-12:
                    best = S2
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
                conflicts = [b for b in best if masks_local[a] & masks_local[b] != 0]
                if not conflicts:
                    # Direct add
                    S2 = best + [a]
                    s2 = best_score + weights[a]
                    if s2 > best_score + 1e-12:
                        best = S2
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
                        if b != x and masks_local[a] & masks_local[b] != 0:
                            ok = False
                            break
                    if ok:
                        s2 = best_score - weights[x] + weights[a]
                        if s2 > best_score + 1e-12:
                            best = [b for b in best if b != x] + [a]
                            best_score = s2
                            improved = True
                            break
                if improved:
                    break

            if improved or not time_ok():
                continue

            # Move 4 (NEW): drop TWO elements, then re-greedy.
            # This can escape local optima where no single drop helps but
            # dropping two conflicting elements simultaneously admits a better set.
            for i in range(n):
                if not time_ok():
                    break
                for j in range(i + 1, n):
                    if not time_ok():
                        break
                    x = best[i]
                    y = best[j]
                    drop = {x, y}
                    cand = [a for a in order if a not in drop]
                    S2 = greedy(cand)
                    s2 = score(S2)
                    if s2 > best_score + 1e-12:
                        best = S2
                        best_score = s2
                        improved = True
                        break
                if improved:
                    break

        return best, best_score

    # 2. Initial greedy descending
    best, best_score = local_search(greedy(order))

    # 3. Randomized restarts: try a few perturbed greedy initializations
    rng = random.Random(42)
    num_restarts = 5
    for r in range(num_restarts):
        if not time_ok():
            break
        # Perturb: shuffle a random subset of the order to get a different greedy solution
        perm = list(order)
        # Randomly swap some adjacent pairs to create a slightly different ordering
        for _ in range(20):
            idx = rng.randint(0, len(perm) - 2)
            perm[idx], perm[idx + 1] = perm[idx + 1], perm[idx]
        init = greedy(perm)
        s_init = score(init)
        # Only proceed if the initial solution is not too far from best
        if s_init > best_score - 0.05:
            new_best, new_score = local_search(init)
            if new_score > best_score:
                best = new_best
                best_score = new_score

    return best
