import time
import math
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    N = 512
    # Precompute weights: weight[a] = 1/(512 - a) for a in 1..511
    weights = [0.0] + [1.0 / (N - a) for a in range(1, N)]

    # Precompute conflict bitsets: conflict[a] is a bitset of numbers that conflict with a
    # We'll use Python integers as bitsets (bit i set means number i+1 conflicts)
    # Actually, let's use a set-based approach for clarity, then optimize if needed

    # Precompute which numbers conflict with each other
    # conflict_mask[a] = set of b != a such that gcd(a,b) != 1
    conflict_mask = [set() for _ in range(N)]
    for a in range(1, N):
        for b in range(a + 1, N):
            if gcd(a, b) != 1:
                conflict_mask[a].add(b)
                conflict_mask[b].add(a)

    def is_coprime_with_all(a, chosen_set):
        return a not in conflict_mask[a] | chosen_set if False else all(b not in conflict_mask[a] for b in chosen_set)

    def can_add(a, chosen_set):
        """Check if a can be added to chosen_set (maintaining pairwise coprime)."""
        for b in chosen_set:
            if b in conflict_mask[a]:
                return False
        return True

    def score_of(chosen):
        return sum(weights[a] for a in chosen)

    # Greedy initialization
    def greedy_init():
        chosen = []
        chosen_set = set()
        for a in range(N - 1, 0, -1):
            if can_add(a, chosen_set):
                chosen.append(a)
                chosen_set.add(a)
        return chosen, chosen_set

    best_chosen, best_chosen_set = greedy_init()
    best_score = score_of(best_chosen)

    # Branch and bound to find better solutions
    # We'll try to improve by exploring replacements

    def branch_and_bound_improve(current_chosen, current_score, t0, deadline):
        """Try to improve current solution using branch and bound on swaps."""
        nonlocal best_chosen, best_chosen_set, best_score

        if time.time() > deadline:
            return

        # For each element in current solution, try to replace it
        # with a better one, using B&B to explore combinations

        n = len(current_chosen)
        chosen_set = set(current_chosen)

        # Try single swaps first (fast)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                a_i = current_chosen[i]
                others = [current_chosen[j] for j in range(n) if j != i]
                others_set = set(others)

                # Find best replacement for a_i
                best_repl = None
                best_repl_val = weights[a_i]

                for a in range(N - 1, 0, -1):
                    if a == a_i or a in chosen_set:
                        continue
                    if can_add(a, others_set):
                        val = weights[a]
                        if val > best_repl_val + 1e-15:
                            best_repl_val = val
                            best_repl = a
                    # Early break: if remaining weights can't beat current
                    if weights[a] <= best_repl_val and a < N - 1:
                        # Since weights decrease as a decreases, we can break
                        # Actually weights[a] = 1/(512-a), so larger a means larger weight
                        # We're iterating from high a to low a, so weights decrease
                        break

                if best_repl is not None:
                    # Swap
                    current_chosen[i] = best_repl
                    chosen_set.remove(a_i)
                    chosen_set.add(best_repl)
                    current_score += weights[best_repl] - weights[a_i]
                    improved = True

                    if current_score > best_score:
                        best_score = current_score
                        best_chosen = list(current_chosen)
                        best_chosen_set = set(current_chosen)

        return current_chosen, current_score

    # Apply B&B improvement
    best_chosen, best_score = branch_and_bound_improve(
        list(best_chosen), best_score, start_time, deadline
    )

    # Try multiple random restarts with local search if time permits
    if time.time() < deadline - 10:
        import random
        random.seed(42)

        # Generate prime list for better initialization
        primes = []
        for a in range(2, N):
            is_prime = True
            for p in range(2, int(a ** 0.5) + 1):
                if a % p == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(a)
        primes.sort(reverse=True)

        for restart in range(10):
            if time.time() > deadline - 5:
                break

            # Randomized greedy initialization
            chosen = []
            chosen_set = set()

            # Mix of primes and composites in random order
            all_nums = list(range(1, N))
            random.shuffle(all_nums)
            # Sort by weight descending with some randomness
            all_nums.sort(key=lambda x: (weights[x] + random.random() * 0.01, x), reverse=True)

            for a in all_nums:
                if can_add(a, chosen_set):
                    chosen.append(a)
                    chosen_set.add(a)

            cur_score = score_of(chosen)

            # Local search improvement
            n = len(chosen)
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                for i in range(n):
                    if time.time() > deadline - 2:
                        break
                    a_i = chosen[i]
                    others = [chosen[j] for j in range(n) if j != i]
                    others_set = set(others)

                    best_repl = None
                    best_repl_val = weights[a_i]

                    for a in range(N - 1, 0, -1):
                        if a == a_i or a in chosen_set:
                            continue
                        if can_add(a, others_set):
                            val = weights[a]
                            if val > best_repl_val + 1e-15:
                                best_repl_val = val
                                best_repl = a
                        if weights[a] <= best_repl_val:
                            break

                    if best_repl is not None:
                        chosen[i] = best_repl
                        chosen_set.remove(a_i)
                        chosen_set.add(best_repl)
                        cur_score += weights[best_repl] - weights[a_i]
                        improved = True

                        if cur_score > best_score:
                            best_score = cur_score
                            best_chosen = list(chosen)
                            best_chosen_set = set(chosen)

    # Final check: try to add any remaining numbers that can be added
    if time.time() < deadline - 1:
        for a in range(N - 1, 0, -1):
            if a not in best_chosen_set and can_add(a, best_chosen_set):
                best_chosen.append(a)
                best_chosen_set.add(a)
                best_score += weights[a]

    return best_chosen
