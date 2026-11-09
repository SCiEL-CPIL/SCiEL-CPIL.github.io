"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 115

    N = 512

    def get_primes(n):
        sieve = [True] * n
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]

    primes = get_primes(N)
    prime_to_idx = {p: i for i, p in enumerate(primes)}

    def get_factor_mask(n):
        mask = 0
        tmp = n
        for i, p in enumerate(primes):
            if p * p > tmp:
                break
            if tmp % p == 0:
                mask |= (1 << i)
                while tmp % p == 0:
                    tmp //= p
        if tmp > 1:
            mask |= (1 << prime_to_idx[tmp])
        return mask

    factor_masks = [0] * N
    for a in range(1, N):
        factor_masks[a] = get_factor_mask(a)

    scores = [0.0] * N
    for a in range(1, N):
        scores[a] = 1.0 / (N - a)

    # Sort candidates by score descending
    candidates = list(range(1, N))
    candidates.sort(key=lambda x: scores[x], reverse=True)

    # Precompute conflict sets for faster checking
    # conflicts[a] is a set of b such that gcd(a,b) > 1
    conflicts = [set() for _ in range(N)]
    for a in range(1, N):
        for b in range(a + 1, N):
            if factor_masks[a] & factor_masks[b]:
                conflicts[a].add(b)
                conflicts[b].add(a)

    best_score = 0.0
    best_solution = []

    def greedy_solve(ordering):
        chosen = []
        chosen_set = set()
        chosen_mask = 0
        for a in ordering:
            if factor_masks[a] & chosen_mask == 0:
                chosen.append(a)
                chosen_set.add(a)
                chosen_mask |= factor_masks[a]
        return chosen, chosen_set, chosen_mask

    # Initial greedy strategies
    greedy_orders = [
        candidates,
        sorted(range(1, N), key=lambda x: scores[x] / max(1, bin(factor_masks[x]).count('1')), reverse=True),
    ]

    for trial in range(10):
        if time.time() > deadline:
            break
        if trial < len(greedy_orders):
            order = greedy_orders[trial]
        else:
            order = list(range(1, N))
            random.shuffle(order)
            weighted = [(random.random() * 0.5 + scores[x] / max(1, bin(factor_masks[x]).count('1')), x) for x in order]
            weighted.sort(reverse=True)
            order = [x for _, x in weighted]

        chosen, chosen_set, chosen_mask = greedy_solve(order)
        sc = sum(scores[a] for a in chosen)
        if sc > best_score:
            best_score = sc
            best_solution = list(chosen)

    # B&B on top elements
    top_elements = candidates[:50]
    
    def bb_on_subset(elements):
        nonlocal best_score, best_solution
        n = len(elements)
        if n == 0:
            return [], 0.0

        elems = sorted(elements, key=lambda x: scores[x], reverse=True)
        best_sub = []
        best_sub_score = 0.0

        def rec(idx, current, current_score, current_mask):
            nonlocal best_sub, best_sub_score
            if time.time() > deadline:
                return
            if idx == n:
                if current_score > best_sub_score:
                    best_sub_score = current_score
                    best_sub = list(current)
                return

            a = elems[idx]
            rem_sum = sum(scores[elems[j]] for j in range(idx + 1, n))
            if current_score + rem_sum <= best_sub_score + 1e-12:
                return

            if factor_masks[a] & current_mask == 0:
                current.append(a)
                rec(idx + 1, current, current_score + scores[a], current_mask | factor_masks[a])
                current.pop()

            rec(idx + 1, current, current_score, current_mask)

        rec(0, [], 0.0, 0)
        return best_sub, best_sub_score

    if time.time() < deadline:
        sub_solution, sub_score = bb_on_subset(top_elements)
        sub_set = set(sub_solution)
        sub_mask = 0
        for a in sub_solution:
            sub_mask |= factor_masks[a]

        for a in candidates:
            if a in sub_set:
                continue
            if factor_masks[a] & sub_mask == 0:
                sub_solution.append(a)
                sub_set.add(a)
                sub_mask |= factor_masks[a]

        full_score = sum(scores[a] for a in sub_solution)
        if full_score > best_score:
            best_score = full_score
            best_solution = list(sub_solution)

    # Optimized Local Search
    def local_improve(solution):
        nonlocal best_score, best_solution
        sol = list(solution)
        sol_set = set(sol)
        sol_mask = 0
        for a in sol:
            sol_mask |= factor_masks[a]
        current_score = sum(scores[a] for a in sol)

        # Precompute top candidates for 2-swap to limit search space
        top_candidates = candidates[:100]

        for _ in range(50):
            if time.time() > deadline:
                break

            improved = False

            # 1-swap
            best_gain = 0.0
            best_move = None

            for a in sol:
                temp_mask = sol_mask & ~factor_masks[a]
                for b in top_candidates:
                    if b in sol_set:
                        continue
                    if factor_masks[b] & temp_mask:
                        continue
                    gain = scores[b] - scores[a]
                    if gain > best_gain + 1e-12:
                        best_gain = gain
                        best_move = ('swap', a, b)

            if best_move:
                mt, a, b = best_move
                sol.remove(a)
                sol.append(b)
                sol_set.remove(a)
                sol_set.add(b)
                sol_mask = (sol_mask & ~factor_masks[a]) | factor_masks[b]
                current_score += best_gain
                improved = True

            # 2-swap (limited to top candidates for b1, b2)
            n = len(sol)
            if n >= 2 and time.time() < deadline:
                for i in range(n):
                    if time.time() > deadline:
                        break
                    a1 = sol[i]
                    for j in range(i + 1, n):
                        a2 = sol[j]
                        temp_mask = sol_mask & ~factor_masks[a1] & ~factor_masks[a2]
                        # Find valid b1, b2 from top_candidates
                        valid_b = [b for b in top_candidates if b not in sol_set and (factor_masks[b] & temp_mask) == 0]
                        for idx1 in range(len(valid_b)):
                            b1 = valid_b[idx1]
                            for idx2 in range(idx1 + 1, len(valid_b)):
                                b2 = valid_b[idx2]
                                if factor_masks[b1] & factor_masks[b2]:
                                    continue
                                gain = scores[b1] + scores[b2] - scores[a1] - scores[a2]
                                if gain > best_gain + 1e-12:
                                    best_gain = gain
                                    best_move = ('2swap', a1, a2, b1, b2)

                if best_move and best_move[0] == '2swap':
                    _, a1, a2, b1, b2 = best_move
                    sol.remove(a1)
                    sol.remove(a2)
                    sol.append(b1)
                    sol.append(b2)
                    sol_set.remove(a1)
                    sol_set.remove(a2)
                    sol_set.add(b1)
                    sol_set.add(b2)
                    sol_mask = (sol_mask & ~factor_masks[a1] & ~factor_masks[a2]) | factor_masks[b1] | factor_masks[b2]
                    current_score += best_gain
                    improved = True

            if not improved:
                break

        if current_score > best_score:
            best_score = current_score
            best_solution = list(sol)

        return sol

    if time.time() < deadline:
        best_solution = local_improve(best_solution)

    # Final restarts
    if time.time() < deadline:
        for trial in range(10):
            if time.time() > deadline:
                break
            order = list(range(1, N))
            random.shuffle(order)
            weighted = [(random.random() * 0.3 + scores[x] / max(1, bin(factor_masks[x]).count('1')), x) for x in order]
            weighted.sort(reverse=True)
            order = [x for _, x in weighted]
            chosen, chosen_set, chosen_mask = greedy_solve(order)
            improved = local_improve(chosen)

    return best_solution
# EVOLVE-BLOCK-END
