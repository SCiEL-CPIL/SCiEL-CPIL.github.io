import math
import time
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 85  # safety margin under 90s

    N = 256

    # Sieve for primes
    def sieve(limit):
        is_p = [True] * (limit + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                for j in range(i * i, limit + 1, i):
                    is_p[j] = False
        return [i for i in range(2, limit + 1) if is_p[i]]

    primes = sieve(N)
    prime_set = set(primes)

    # Prime powers: largest p^k <= N for each prime p
    prime_powers = {}
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        prime_powers[p] = power

    # Base solution: 1 + all prime powers
    base_set = [1] + [prime_powers[p] for p in primes]
    base_sum = sum(base_set)

    # Precompute prime factorization for all numbers 1..N
    F = [None] * (N + 1)
    F[1] = frozenset()
    for i in range(2, N + 1):
        x = i
        factors = set()
        d = 2
        while d * d <= x:
            if x % d == 0:
                factors.add(d)
                while x % d == 0:
                    x //= d
            else:
                d += 1
        if x > 1:
            factors.add(x)
        F[i] = frozenset(factors)

    # For each subset of primes (size 2-4), find the best composite number
    # that is a product of powers of exactly those primes and <= N
    # Gain = composite - sum(prime_powers[p] for p in subset)
    best_for_subset = {}  # frozenset(primes) -> (value, gain)

    # Build all valid composites from prime subsets
    # A composite c uses exactly the primes in its factorization
    composite_by_primes = {}  # frozenset(primes) -> list of values
    for c in range(2, N + 1):
        if c in prime_set:
            continue  # skip primes, they're handled by prime_powers
        ps = F[c]
        if len(ps) < 2:
            continue  # prime powers already handled
        if len(ps) > 4:
            continue  # too many primes, unlikely to be optimal
        if ps not in composite_by_primes:
            composite_by_primes[ps] = []
        composite_by_primes[ps].append(c)

    # For each prime subset, pick the best composite
    for ps, vals in composite_by_primes.items():
        base_cost = sum(prime_powers[p] for p in ps)
        best_val = max(vals)
        gain = best_val - base_cost
        if gain > 0:
            best_for_subset[ps] = (best_val, gain)

    # Now find an optimal assignment: select a set of disjoint prime subsets
    # to maximize total gain. This is a weighted set packing problem.
    # Since primes are small (54 primes up to 256) and subsets are small,
    # use greedy + local search.

    # Sort subsets by gain/size ratio for greedy
    subsets = list(best_for_subset.keys())
    subsets.sort(key=lambda ps: -best_for_subset[ps][1] / len(ps))

    # Greedy: pick subsets in order if disjoint
    selected_subsets = []
    used_primes = set()
    for ps in subsets:
        if ps.isdisjoint(used_primes):
            selected_subsets.append(ps)
            used_primes |= ps

    # Build solution from greedy
    def build_solution(sel_subsets, all_primes):
        result = [1]
        used = set()
        for ps in sel_subsets:
            val, _ = best_for_subset[ps]
            result.append(val)
            used |= ps
        for p in all_primes:
            if p not in used:
                result.append(prime_powers[p])
        return result

    greedy_sol = build_solution(selected_subsets, primes)
    best_sum = sum(greedy_sol)
    best_sol = greedy_sol[:]

    # Local search: try swapping one selected subset for another
    # Or try adding/removing individual composites
    improved = True
    max_iter = 50
    iteration = 0

    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        if time.time() > deadline - 5:
            break

        # Try 1-exchange: remove one selected subset, try alternatives
        for idx in range(len(selected_subsets)):
            if time.time() > deadline - 5:
                break
            removed = selected_subsets[idx]
            remaining = [s for i, s in enumerate(selected_subsets) if i != idx]
            rem_used = set()
            for s in remaining:
                rem_used |= s

            # Try all subsets disjoint from rem_used that include at least one prime from removed
            for ps in subsets:
                if not ps.isdisjoint(rem_used):
                    continue
                if not ps & removed:
                    continue
                # Check if ps is "better" than removed in context
                new_sel = remaining + [ps]
                new_sol = build_solution(new_sel, primes)
                new_sum = sum(new_sol)
                if new_sum > best_sum:
                    best_sum = new_sum
                    best_sol = new_sol[:]
                    selected_subsets = new_sel[:]
                    improved = True

        # Try 2-exchange: swap two selected subsets
        if len(selected_subsets) >= 2:
            for i in range(len(selected_subsets)):
                for j in range(i + 1, len(selected_subsets)):
                    if time.time() > deadline - 5:
                        break
                    si = selected_subsets[i]
                    sj = selected_subsets[j]
                    rest = [s for k, s in enumerate(selected_subsets) if k != i and k != j]
                    rest_used = set()
                    for s in rest:
                        rest_used |= s

                    # Try replacing si with another subset
                    for ps in subsets:
                        if not ps.isdisjoint(rest_used | sj):
                            continue
                        if ps == si:
                            continue
                        new_sel = rest + [sj, ps]
                        new_sol = build_solution(new_sel, primes)
                        new_sum = sum(new_sol)
                        if new_sum > best_sum:
                            best_sum = new_sum
                            best_sol = new_sol[:]
                            selected_subsets = new_sel[:]
                            improved = True
                            break

    # Additional local search: try adding individual composites not in current solution
    current_factors = set()
    for x in best_sol:
        current_factors |= F[x]

    for c in range(2, N + 1):
        if time.time() > deadline - 3:
            break
        if c in best_sol:
            continue
        if F[c] & current_factors:
            continue
        # Can add c freely
        new_sol = best_sol + [c]
        new_sum = sum(new_sol)
        if new_sum > best_sum:
            best_sum = new_sum
            best_sol = new_sol[:]
            current_factors |= F[c]

    # Final polish: try removing one element and re-optimizing
    for i in range(len(best_sol)):
        if time.time() > deadline - 2:
            break
        if best_sol[i] == 1:
            continue
        rest = [x for j, x in enumerate(best_sol) if j != i]
        rest_factors = set()
        for x in rest:
            rest_factors |= F[x]

        # Find best replacement
        best_repl = None
        best_repl_val = 0
        for c in range(2, N + 1):
            if c in rest:
                continue
            if F[c] & rest_factors:
                continue
            if c > best_repl_val:
                best_repl_val = c
                best_repl = c

        if best_repl is not None and best_repl > best_sol[i]:
            new_sol = rest + [best_repl]
            new_sum = sum(new_sol)
            if new_sum > best_sum:
                best_sum = new_sum
                best_sol = new_sol[:]

    # Verify pairwise coprimality
    def verify(lst):
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if math.gcd(lst[i], lst[j]) != 1:
                    return False
        return True

    if not verify(best_sol):
        # Fallback to verified greedy
        if not verify(greedy_sol):
            # Fallback to base
            best_sol = base_set[:]
        else:
            best_sol = greedy_sol[:]

    # Ensure distinct and in range
    best_sol = list(set(x for x in best_sol if 1 <= x <= 256))

    # Final type check: all must be Python int
    best_sol = [int(x) for x in best_sol]

    return best_sol
