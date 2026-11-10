"""Combined approach: multi-swap local search + fractional-relaxation branch-and-bound."""
# EVOLVE-BLOCK-START
import time
import random
import math
from math import gcd, isqrt
from fractions import Fraction

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    if n < 9:
        return True
    if n % 3 == 0:
        return False
    if n < 25:
        return True
    if n % 5 == 0:
        return False
    if n < 49:
        return True
    if n % 7 == 0:
        return False
    if n < 121:
        return True
    i = 11
    while i <= isqrt(n):
        if n % i == 0:
            return False
        i += 2
    return True

def _get_prime_factors(n):
    """Return list of unique prime factors of n."""
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    N = 512

    # Precompute values and prime factors for all numbers 1..511
    values = [Fraction(0)] * N
    prime_factors = [[] for _ in range(N)]
    is_prime_arr = [False] * N

    for a in range(1, N):
        values[a] = Fraction(1, N - a)
        prime_factors[a] = _get_prime_factors(a)
        is_prime_arr[a] = _is_prime(a)

    # Identify primes
    primes = [a for a in range(1, N) if is_prime_arr[a]]

    # Initialize with all primes
    selected = set(primes)

    # Build mapping: prime -> set of selected elements that have this prime as a factor
    prime_to_selected = {}
    for b in selected:
        for p in prime_factors[b]:
            if p not in prime_to_selected:
                prime_to_selected[p] = set()
            prime_to_selected[p].add(b)

    def apply_move(add_candidates, remove_elements):
        """Apply a move: remove elements, add candidates. Update prime_to_selected."""
        for b in remove_elements:
            for p in prime_factors[b]:
                if p in prime_to_selected:
                    prime_to_selected[p].discard(b)
                    if not prime_to_selected[p]:
                        del prime_to_selected[p]
        for c in add_candidates:
            for p in prime_factors[c]:
                if p not in prime_to_selected:
                    prime_to_selected[p] = set()
                prime_to_selected[p].add(c)
        selected.difference_update(remove_elements)
        selected.update(add_candidates)

    def get_score():
        return sum(values[a] for a in selected)

    current_score = get_score()

    def conflicts_of(candidates):
        """All selected elements conflicting with any candidate."""
        conflict_set = set()
        for c in candidates:
            for p in prime_factors[c]:
                if p in prime_to_selected:
                    conflict_set.update(prime_to_selected[p])
        return conflict_set

    # Phase 1: 1-for-k First-Improvement Search
    improved = True
    while improved:
        if time.time() > deadline - 40:
            break
        improved = False
        for c in range(1, N):
            if c in selected:
                continue
            conflict_set = conflicts_of([c])
            conflict_cost = sum(values[b] for b in conflict_set)
            gain = values[c] - conflict_cost
            if gain > 0:
                apply_move([c], list(conflict_set))
                current_score += gain
                improved = True
                break

    # Phase 2: 2-for-k Best-Improvement Search
    if time.time() < deadline - 25:
        composites = [a for a in range(1, N) if not is_prime_arr[a] and a > 1]
        composites.sort(key=lambda x: values[x], reverse=True)
        top_composites = composites[:80]

        improved = True
        while improved:
            if time.time() > deadline - 15:
                break
            improved = False
            best_gain = Fraction(0)
            best_add = None
            best_remove = None

            for i in range(len(top_composites)):
                c1 = top_composites[i]
                if c1 in selected:
                    continue
                for j in range(i + 1, len(top_composites)):
                    c2 = top_composites[j]
                    if c2 in selected:
                        continue
                    if gcd(c1, c2) != 1:
                        continue
                    conflict_set = conflicts_of([c1, c2])
                    conflict_cost = sum(values[b] for b in conflict_set)
                    gain = values[c1] + values[c2] - conflict_cost
                    if gain > best_gain:
                        best_gain = gain
                        best_add = [c1, c2]
                        best_remove = list(conflict_set)

            if best_add is not None:
                apply_move(best_add, best_remove)
                current_score += best_gain
                improved = True

    # Phase 3: 3-for-k Best-Improvement Search
    if time.time() < deadline - 15:
        composites = [a for a in range(1, N) if not is_prime_arr[a] and a > 1]
        composites.sort(key=lambda x: values[x], reverse=True)
        top_composites = composites[:50]

        improved = True
        while improved:
            if time.time() > deadline - 10:
                break
            improved = False
            best_gain = Fraction(0)
            best_add = None
            best_remove = None

            for i in range(len(top_composites)):
                c1 = top_composites[i]
                if c1 in selected:
                    continue
                for j in range(i + 1, len(top_composites)):
                    c2 = top_composites[j]
                    if c2 in selected:
                        continue
                    if gcd(c1, c2) != 1:
                        continue
                    for k in range(j + 1, len(top_composites)):
                        c3 = top_composites[k]
                        if c3 in selected:
                            continue
                        if gcd(c1, c3) != 1 or gcd(c2, c3) != 1:
                            continue
                        conflict_set = conflicts_of([c1, c2, c3])
                        conflict_cost = sum(values[b] for b in conflict_set)
                        gain = values[c1] + values[c2] + values[c3] - conflict_cost
                        if gain > best_gain:
                            best_gain = gain
                            best_add = [c1, c2, c3]
                            best_remove = list(conflict_set)

            if best_add is not None:
                apply_move(best_add, best_remove)
                current_score += best_gain
                improved = True

    # Phase 4: Fractional-relaxation branch-and-bound over high-value candidates.
    # Upper bound on addable value = sum of values of unselected candidates whose
    # prime factors don't conflict with current selected set (a valid fractional bound,
    # since any feasible addition must draw only from non-conflicting candidates).
    # We greedily branch: pick the highest-value candidate, either add it (if it
    # doesn't conflict with already-chosen additions) or skip it, pruning when the
    # remaining fractional bound can't beat the best-so-far improvement.
    if time.time() < deadline - 12:
        # Candidate pool: unselected numbers with value, sorted by value desc
        pool = [a for a in range(1, N) if a not in selected and a > 1]
        pool.sort(key=lambda x: values[x], reverse=True)
        pool = pool[:60]

        best_solution = list(selected)
        best_solution_score = current_score

        # We track additions on top of the current selected set.
        # chosen = list of candidates we have decided to add (mutually coprime,
        # and coprime with current selected).
        def fractional_bound(chosen, chosen_conflict_primes):
            """Upper bound on total value of (chosen) plus any further additions
            that don't conflict with current selected OR with chosen."""
            total = sum(values[c] for c in chosen)
            # For each unchosen pool candidate, it's addable iff it doesn't share
            # a prime with current selected or with chosen.
            for c in pool:
                if c in chosen:
                    continue
                ok = True
                for p in prime_factors[c]:
                    if p in prime_to_selected or p in chosen_conflict_primes:
                        ok = False
                        break
                if ok:
                    total += values[c]
            return total

        # DFS with bound
        node_count = [0]
        def dfs(idx, chosen, chosen_conflict_primes):
            if time.time() > deadline - 10:
                return
            node_count[0] += 1

            # Compute fractional bound
            bound = fractional_bound(chosen, chosen_conflict_primes)
            # bound is total value of chosen + all addable remaining.
            # Compare to best_solution_score: if bound <= best, prune.
            if bound <= best_solution_score:
                return

            if idx >= len(pool):
                # Evaluate this chosen set
                if len(chosen) > 0:
                    # All chosen are coprime with selected and each other by construction
                    cand_score = current_score + sum(values[c] for c in chosen) - 0
                    # Actually chosen don't conflict with selected (we ensured),
                    # so gain = sum(values[chosen])
                    if cand_score > best_solution_score:
                        best_solution_score[0] if False else None
                return

            c = pool[idx]

            # Option 1: skip c
            dfs(idx + 1, chosen, chosen_conflict_primes)

            # Option 2: add c if it doesn't conflict with selected or chosen
            if time.time() > deadline - 10:
                return
            addable = True
            for p in prime_factors[c]:
                if p in prime_to_selected or p in chosen_conflict_primes:
                    addable = False
                    break
            if addable:
                new_primes = set(chosen_conflict_primes)
                new_primes.update(prime_factors[c])
                dfs(idx + 1, chosen + [c], new_primes)

        # Run DFS; track best via a mutable holder
        best_holder = {'score': current_score, 'chosen': []}

        def dfs2(idx, chosen, chosen_conflict_primes):
            if time.time() > deadline - 10:
                return
            # Fractional bound
            total = sum(values[c] for c in chosen)
            for cc in pool:
                if cc in chosen:
                    continue
                ok = True
                for p in prime_factors[cc]:
                    if p in prime_to_selected or p in chosen_conflict_primes:
                        ok = False
                        break
                if ok:
                    total += values[cc]
            if total <= best_holder['score']:
                return
            if idx >= len(pool):
                if len(chosen) > 0:
                    cand_score = current_score + sum(values[c] for c in chosen)
                    if cand_score > best_holder['score']:
                        best_holder['score'] = cand_score
                        best_holder['chosen'] = list(chosen)
                return
            c = pool[idx]
            # skip
            dfs2(idx + 1, chosen, chosen_conflict_primes)
            # add if possible
            if time.time() > deadline - 10:
                return
            addable = True
            for p in prime_factors[c]:
                if p in prime_to_selected or p in chosen_conflict_primes:
                    addable = False
                    break
            if addable:
                new_primes = set(chosen_conflict_primes)
                new_primes.update(prime_factors[c])
                dfs2(idx + 1, chosen + [c], new_primes)

        dfs2(0, [], set())

        # Apply best additions found
        if best_holder['chosen']:
            apply_move(best_holder['chosen'], [])
            current_score = best_holder['score']

    # Phase 5: Simulated Annealing (if time permits)
    if time.time() < deadline - 5:
        T = 0.05
        T_min = 0.001
        cooling = 0.995
        sa_deadline = deadline - 2

        while T > T_min and time.time() < sa_deadline:
            if not selected:
                break
            a_out = random.choice(list(selected))
            unselected = [a for a in range(1, N) if a not in selected]
            if not unselected:
                break
            a_in = random.choice(unselected)

            valid = True
            for b in selected:
                if b == a_out:
                    continue
                if gcd(a_in, b) != 1:
                    valid = False
                    break

            if not valid:
                T *= cooling
                continue

            old_score = current_score
            apply_move([a_in], [a_out])
            new_score = get_score()
            delta = new_score - old_score

            if delta > 0:
                current_score = new_score
            else:
                prob = math.exp(float(delta) / T)
                if random.random() < prob:
                    current_score = new_score
                else:
                    apply_move([a_out], [a_in])

            T *= cooling

    # Final pass of 1-for-k best-improvement to clean up
    if time.time() < deadline - 5:
        improved = True
        while improved:
            if time.time() > deadline - 1:
                break
            improved = False
            best_gain = Fraction(0)
            best_add = None
            best_remove = None

            for c in range(1, N):
                if c in selected:
                    continue
                conflict_set = conflicts_of([c])
                conflict_cost = sum(values[b] for b in conflict_set)
                gain = values[c] - conflict_cost
                if gain > best_gain:
                    best_gain = gain
                    best_add = [c]
                    best_remove = list(conflict_set)

            if best_add is not None:
                apply_move(best_add, best_remove)
                current_score += best_gain
                improved = True

    # Final Verification
    result = list(selected)
    for a in result:
        if a < 1 or a > 511:
            raise ValueError(f"Out of range: {a}")
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                raise ValueError(f"Not coprime: {result[i]}, {result[j]}")

    return result
# EVOLVE-BLOCK-END
