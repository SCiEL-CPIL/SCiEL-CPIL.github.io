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
    num_primes = len(primes)

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

    # Build conflict graph: two numbers conflict if they share a prime factor
    # Represent as adjacency list using bitmasks for fast operations
    # Since we have ~97 primes, we use integer masks

    # Precompute: for each number, which other numbers conflict with it
    # Use a set of conflicting numbers for each candidate
    conflicts = [set() for _ in range(N)]
    for a in range(1, N):
        for b in range(a + 1, N):
            if factor_masks[a] & factor_masks[b]:
                conflicts[a].add(b)
                conflicts[b].add(a)

    # Sort candidates by score descending for branch-and-bound
    candidates = list(range(1, N))
    candidates.sort(key=lambda x: scores[x], reverse=True)

    # Precompute cumulative scores for upper bound estimation
    # For a given set of available candidates, upper bound = sum of top k scores
    # where k is max number we can still add (bounded by independence number estimate)

    # Branch and bound with greedy upper bound
    best_score = 0.0
    best_solution = []

    # Use a greedy solution as initial lower bound
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

    # Get initial solution with multiple greedy strategies
    greedy_orders = [
        candidates,  # pure score order
        sorted(range(1, N), key=lambda x: scores[x] / max(1, bin(factor_masks[x]).count('1')), reverse=True),
    ]

    for trial in range(20):
        if time.time() > deadline:
            break
        if trial < len(greedy_orders):
            order = greedy_orders[trial]
        else:
            order = list(range(1, N))
            random.shuffle(order)
            # Weighted random ordering
            weighted = [(random.random() * 0.5 + scores[x] / max(1, bin(factor_masks[x]).count('1')), x) for x in order]
            weighted.sort(reverse=True)
            order = [x for _, x in weighted]

        chosen, chosen_set, chosen_mask = greedy_solve(order)
        sc = sum(scores[a] for a in chosen)
        if sc > best_score:
            best_score = sc
            best_solution = list(chosen)

    # Now do branch and bound on the remaining decision
    # Key insight: we can model this as maximum weight independent set
    # Use recursive B&B with strong pruning

    def branch_and_bound(available, current_score, current_set, current_mask, depth=0):
        nonlocal best_score, best_solution
        if time.time() > deadline:
            return

        # Upper bound: current_score + sum of all remaining scores
        # Better bound: use greedy packing
        if not available:
            if current_score > best_score:
                best_score = current_score
                best_solution = list(current_set)
            return

        # Pruning: if current_score + sum(remaining scores) <= best_score, prune
        remaining_sum = sum(scores[a] for a in available)
        if current_score + remaining_sum <= best_score + 1e-12:
            return

        # Greedy upper bound: try to greedily pack from available
        # This gives a tighter bound than just sum
        greedy_upper = current_score
        temp_mask = current_mask
        temp_set = set(current_set)
        for a in available:
            if factor_masks[a] & temp_mask == 0:
                greedy_upper += scores[a]
                temp_mask |= factor_masks[a]
                temp_set.add(a)
        if greedy_upper <= best_score + 1e-12:
            return

        # Choose next candidate: pick the one with highest score that could improve
        # Try including the highest-score available candidate first
        # Then try excluding it (branch)

        # Sort available by score descending
        avail_sorted = sorted(available, key=lambda x: scores[x], reverse=True)

        # Branch 1: include avail_sorted[0]
        a = avail_sorted[0]
        if factor_masks[a] & current_mask == 0:
            new_available = [b for b in avail_sorted[1:] if b not in conflicts[a] and factor_masks[b] & (current_mask | factor_masks[a]) == 0]
            new_set = list(current_set) + [a]
            new_mask = current_mask | factor_masks[a]
            branch_and_bound(new_available, current_score + scores[a], new_set, new_mask, depth + 1)

        # Branch 2: exclude avail_sorted[0]
        new_available = avail_sorted[1:]
        if new_available:
            branch_and_bound(new_available, current_score, current_set, current_mask, depth + 1)

    # Start B&B from the best greedy solution's complement
    # Actually, let's do a full B&B but with good initial bound
    # The issue is that full B&B on 511 nodes is too large
    # Instead, use the greedy solution as a starting point and do local improvement via B&B on swaps

    # Better approach: use the greedy solution, then try to improve via targeted B&B
    # For each element in the solution, try replacing it with a better combination

    # Alternative: use a different structural approach
    # Since the score is 1/(512-a), the highest scores are for a near 511
    # Let's focus on the top-scoring elements and use B&B on a reduced set

    # Get top K elements by score
    K = 60
    top_elements = candidates[:K]

    # Do B&B on just these top elements to find the best subset
    # Then fill in remaining capacity with greedy

    def bb_on_subset(elements):
        nonlocal best_score, best_solution
        n = len(elements)
        if n == 0:
            return [], 0.0

        # Sort by score descending
        elems = sorted(elements, key=lambda x: scores[x], reverse=True)

        best_sub = []
        best_sub_score = 0.0

        # Recursive B&B
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

            # Upper bound: current_score + sum of remaining
            rem_sum = sum(scores[elems[j]] for j in range(idx + 1, n))
            if current_score + rem_sum <= best_sub_score + 1e-12:
                return

            # Branch 1: include a
            if factor_masks[a] & current_mask == 0:
                current.append(a)
                rec(idx + 1, current, current_score + scores[a], current_mask | factor_masks[a])
                current.pop()

            # Branch 2: exclude a
            rec(idx + 1, current, current_score, current_mask)

        rec(0, [], 0.0, 0)
        return best_sub, best_sub_score

    # Try B&B on top 50 elements
    if time.time() < deadline:
        sub_solution, sub_score = bb_on_subset(top_elements[:50])
        # Fill in with greedy on remaining
        sub_set = set(sub_solution)
        sub_mask = 0
        for a in sub_solution:
            sub_mask |= factor_masks[a]

        # Greedily add more elements that don't conflict
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

    # Local improvement: try swapping elements
    def local_improve(solution):
        nonlocal best_score, best_solution
        sol = list(solution)
        sol_set = set(sol)
        sol_mask = 0
        for a in sol:
            sol_mask |= factor_masks[a]
        current_score = sum(scores[a] for a in sol)

        for _ in range(50):
            if time.time() > deadline:
                break

            improved = False

            # Try 1-swap: remove one, add one
            best_gain = 0.0
            best_move = None

            for a in sol:
                temp_mask = sol_mask & ~factor_masks[a]
                for b in candidates:
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

            # Try 2-swap: remove two, add two
            n = len(sol)
            if n >= 2:
                for i in range(n):
                    if time.time() > deadline:
                        break
                    a1 = sol[i]
                    for j in range(i + 1, n):
                        a2 = sol[j]
                        temp_mask = sol_mask & ~factor_masks[a1] & ~factor_masks[a2]
                        valid = [b for b in candidates if b not in sol_set and (factor_masks[b] & temp_mask) == 0]
                        for idx1 in range(len(valid)):
                            b1 = valid[idx1]
                            for idx2 in range(idx1 + 1, len(valid)):
                                b2 = valid[idx2]
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

    # Final check
    if time.time() < deadline:
        # One more round of local improvement from a fresh greedy
        for trial in range(5):
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
