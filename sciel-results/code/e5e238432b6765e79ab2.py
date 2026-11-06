"""
Pairwise coprime integers below 512 maximizing sum 1/(512-a).

Strategy:
1. Exact Maximum Weight Independent Set via MILP (PuLP/scipy) for global optimum.
2. Fallback: enhanced heuristic with bitmask + multi-start local search.
"""
import math
import random
import time
import sys

def construct():
    deadline = time.time() + 140.0
    N = 512

    # Precompute prime factor bitmasks
    is_prime = [True] * N
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N, i):
                is_prime[j] = False

    primes = [i for i in range(2, N) if is_prime[i]]
    prime_index = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)

    prime_factors_mask = [0] * N
    for a in range(2, N):
        x = a
        mask = 0
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                mask |= (1 << prime_index[p])
                while x % p == 0:
                    x //= p
        if x > 1:
            mask |= (1 << prime_index[x])
        prime_factors_mask[a] = mask
    prime_factors_mask[1] = 0

    def score(arr):
        return sum(1.0 / (N - a) for a in arr)

    # Weights: w[a] = 1/(N-a) for a in 1..511
    weights = [1.0 / (N - a) for a in range(N)]  # index 0 unused

    # Try MILP approach first
    try:
        from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpBinary, PULP_CBC_CMD
        import pulp

        # Build conflict graph: a and b conflict if gcd(a,b) > 1
        # We need: for each pair (a,b) with gcd(a,b)>1, x_a + x_b <= 1
        # Objective: maximize sum w_a * x_a

        # To keep the MILP tractable, we only need constraints for conflicting pairs.
        # Numbers 1..511, that's 511 variables.
        # Conflicting pairs: for each a, b where a<b and gcd(a,b)>1.

        # Precompute conflict pairs
        conflict_pairs = []
        for a in range(1, N):
            for b in range(a+1, N):
                if (prime_factors_mask[a] & prime_factors_mask[b]) != 0:
                    conflict_pairs.append((a, b))

        # Also, number 1 is coprime with everything, so no conflicts for 1.
        # (mask[1] = 0, so 1 & anything = 0, so no conflicts)

        if time.time() > deadline - 10:
            raise TimeoutError("Not enough time for MILP")

        prob = LpProblem("coprime_mwis", LpMaximize)

        # Variables
        x = {}
        for a in range(1, N):
            x[a] = LpVariable(f"x_{a}", cat=LpBinary)

        # Objective
        prob += lpSum(weights[a] * x[a] for a in range(1, N))

        # Constraints: for each conflicting pair
        for a, b in conflict_pairs:
            prob += x[a] + x[b] <= 1

        # Solve
        solver = PULP_CBC_CMD(msg=0, timeLimit=int(deadline - time.time() - 5))
        prob.solve(solver)

        if pulp.LpStatus[prob.status] == "Optimal":
            chosen = [a for a in range(1, N) if x[a].value() is not None and x[a].value() > 0.5]
            if chosen:
                s = score(chosen)
                # Verify pairwise coprime
                valid = True
                for i in range(len(chosen)):
                    for j in range(i+1, len(chosen)):
                        if math.gcd(chosen[i], chosen[j]) != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    return chosen
                else:
                    # Fallback
                    raise ValueError("MILP solution invalid")
        else:
            raise ValueError(f"MILP status: {pulp.LpStatus[prob.status]}")

    except Exception:
        pass

    # Fallback: heuristic approach
    def greedy_desc():
        chosen = []
        used_mask = 0
        for a in range(511, 0, -1):
            if (prime_factors_mask[a] & used_mask) == 0:
                chosen.append(a)
                used_mask |= prime_factors_mask[a]
        return chosen

    best = greedy_desc()
    best_score = score(best)

    all_nums = list(range(1, 512))
    rng = random.Random(42)

    def greedy_with_order(order):
        chosen = []
        used_mask = 0
        for a in order:
            if (prime_factors_mask[a] & used_mask) == 0:
                chosen.append(a)
                used_mask |= prime_factors_mask[a]
        return chosen

    import numpy as np
    np_rng = np.random.RandomState(42)
    all_weights = np.array([1.0 / (N - a) for a in all_nums])
    norm_w = all_weights / all_weights.sum()

    for trial in range(300):
        if time.time() > deadline - 20:
            break
        order_idx = np_rng.choice(len(all_nums), size=len(all_nums), replace=False, p=norm_w)
        order = [all_nums[i] for i in order_idx]
        cand = greedy_with_order(order)
        s = score(cand)
        if s > best_score:
            best_score = s
            best = cand

    def local_search(chosen, max_iters=100):
        chosen_set = set(chosen)
        current_score = sum(1.0 / (N - a) for a in chosen)
        used_mask = 0
        for a in chosen:
            used_mask |= prime_factors_mask[a]
        all_by_value = list(range(511, 0, -1))

        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time.time() > deadline - 10:
                break
            improved = False
            iters += 1

            # Try adding
            for a in all_by_value:
                if a in chosen_set:
                    continue
                if (prime_factors_mask[a] & used_mask) == 0:
                    chosen.append(a)
                    chosen_set.add(a)
                    used_mask |= prime_factors_mask[a]
                    current_score += 1.0 / (N - a)
                    improved = True
                    break

            if improved:
                continue

            # Try single swap
            sorted_chosen = sorted(chosen, key=lambda x: 1.0 / (N - x))
            for a in sorted_chosen:
                if time.time() > deadline - 10:
                    break
                remaining = [x for x in chosen if x != a]
                rem_mask = 0
                for x in remaining:
                    rem_mask |= prime_factors_mask[x]
                new_chosen = list(remaining)
                new_set = set(remaining)
                new_mask = rem_mask
                new_score = sum(1.0 / (N - x) for x in new_chosen)
                for c in all_by_value:
                    if c in new_set:
                        continue
                    if (prime_factors_mask[c] & new_mask) == 0:
                        new_chosen.append(c)
                        new_set.add(c)
                        new_mask |= prime_factors_mask[c]
                        new_score += 1.0 / (N - c)
                if new_score > current_score + 1e-15:
                    chosen = new_chosen
                    chosen_set = new_set
                    used_mask = new_mask
                    current_score = new_score
                    improved = True
                    break

            if improved:
                continue

            # Try two-element swap
            if len(chosen) >= 3:
                sorted_chosen2 = sorted(chosen, key=lambda x: 1.0 / (N - x))
                for i in range(len(sorted_chosen2) - 1):
                    if time.time() > deadline - 10:
                        break
                    a1, a2 = sorted_chosen2[i], sorted_chosen2[i+1]
                    remaining = [x for x in chosen if x != a1 and x != a2]
                    rem_mask = 0
                    for x in remaining:
                        rem_mask |= prime_factors_mask[x]
                    new_chosen = list(remaining)
                    new_set = set(remaining)
                    new_mask = rem_mask
                    new_score = sum(1.0 / (N - x) for x in new_chosen)
                    for c in all_by_value:
                        if c in new_set:
                            continue
                        if (prime_factors_mask[c] & new_mask) == 0:
                            new_chosen.append(c)
                            new_set.add(c)
                            new_mask |= prime_factors_mask[c]
                            new_score += 1.0 / (N - c)
                    if new_score > current_score + 1e-15:
                        chosen = new_chosen
                        chosen_set = new_set
                        used_mask = new_mask
                        current_score = new_score
                        improved = True
                        break

        return chosen, current_score

    best, best_score = local_search(best, max_iters=100)

    # Random restarts
    for restart in range(50):
        if time.time() > deadline - 15:
            break
        rng2 = random.Random(2000 + restart * 104729)
        strategy = restart % 3

        if strategy == 0:
            seed_candidates = list(range(511, 300, -1))
            rng2.shuffle(seed_candidates)
            chosen = []
            used_mask = 0
            for a in seed_candidates:
                if (prime_factors_mask[a] & used_mask) == 0:
                    chosen.append(a)
                    used_mask |= prime_factors_mask[a]
                if len(chosen) >= 3:
                    break
        elif strategy == 1:
            high_primes = [p for p in primes if p >= 300]
            rng2.shuffle(high_primes)
            chosen = []
            used_mask = 0
            for a in high_primes:
                if (prime_factors_mask[a] & used_mask) == 0:
                    chosen.append(a)
                    used_mask |= prime_factors_mask[a]
                if len(chosen) >= 3:
                    break
        else:
            chosen = [1]
            used_mask = 0
            seed_candidates = list(range(511, 400, -1))
            rng2.shuffle(seed_candidates)
            for a in seed_candidates:
                if a == 1:
                    continue
                if (prime_factors_mask[a] & used_mask) == 0:
                    chosen.append(a)
                    used_mask |= prime_factors_mask[a]
                if len(chosen) >= 3:
                    break

        if not chosen:
            continue

        for a in range(511, 0, -1):
            if a in chosen:
                continue
            if (prime_factors_mask[a] & used_mask) == 0:
                chosen.append(a)
                used_mask |= prime_factors_mask[a]

        chosen, s = local_search(chosen, max_iters=80)
        if s > best_score:
            best_score = s
            best = chosen

    # Final polish: add any remaining coprime numbers
    best_set = set(best)
    used_mask = 0
    for a in best:
        used_mask |= prime_factors_mask[a]
    for a in range(511, 0, -1):
        if a not in best_set:
            if (prime_factors_mask[a] & used_mask) == 0:
                best.append(a)
                best_set.add(a)
                used_mask |= prime_factors_mask[a]

    return best
