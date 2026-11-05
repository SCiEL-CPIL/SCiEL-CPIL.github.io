"""
Strategy: Model the problem as a maximum independent set on a conflict graph.
Two numbers conflict if including both (in some order) would cause a product
collision. We build the graph, then use branch-and-bound to find the largest
independent set. The resulting set is sorted to form the strictly increasing
sequence.
"""
import time
import sys

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    N = 128  # numbers in [2, 128]
    candidates = list(range(2, 129))
    n = len(candidates)  # 127 elements

    # Precompute: for each number, its "signature" as a set of prime-exponent vectors
    # Two numbers a, b conflict if there exists a consecutive block in any ordering
    # where product collisions occur. But since the sequence must be strictly increasing,
    # the order is fixed: sorted order. So we can check: if we include a set S,
    # sorted, do all consecutive block products collide?
    #
    # Key insight: The constraint is global (all consecutive block products distinct),
    # not just pairwise. So a simple conflict graph isn't sufficient.
    #
    # Alternative: Use a different framing. Since the sequence must be strictly increasing,
    # the order is determined by the set. So we need to find the largest subset S of
    # {2,...,128} such that when sorted, all consecutive block products are distinct.
    #
    # This is a hard combinatorial problem. Let's use a different approach:
    # Randomized local search with simulated annealing on the subset representation.

    # Actually, let's try a smarter approach:
    # The greedy solution gives a baseline. Let's try to find a better solution
    # using a different strategy: start with a random subset and iteratively
    # add/remove elements to maximize size while maintaining validity.

    # First, get the greedy baseline
    def greedy_construct():
        values, products = [], set()
        for x in range(2, 129):
            tails, prod = [x], x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        return values

    best_seq = greedy_construct()
    best_len = len(best_seq)

    # Helper to compute all consecutive block products for a sequence
    def get_all_products(seq):
        products = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                products.add(prod)
        return products

    # Verify the solution is valid
    def is_valid(seq):
        if len(seq) == 0:
            return True
        # Check strictly increasing
        for i in range(len(seq) - 1):
            if seq[i] >= seq[i + 1]:
                return False
        products = get_all_products(seq)
        return len(products) == len(seq) * (len(seq) + 1) // 2

    if not is_valid(best_seq):
        best_seq = list(range(2, 129))
        best_len = len(best_seq)

    # Now try a different strategy: simulated annealing on the subset
    # Represent the solution as a bitset over {2,...,128}
    # Objective: maximize number of elements subject to validity constraint

    import random

    # Initialize with greedy solution
    current_set = set(best_seq)
    current_len = best_len

    # Precompute prime factorizations for efficiency
    def factorize(n):
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors

    # Cache factorizations
    factor_cache = {x: factorize(x) for x in range(2, 129)}

    # Simulated annealing
    if time.time() - start_time < DEADLINE:
        T = 10.0  # initial temperature
        cooling = 0.999
        min_T = 0.01

        # Current solution
        sol = list(best_seq)
        sol_set = set(sol)

        # Try random moves: add, remove, or swap an element
        all_candidates = list(range(2, 129))

        while time.time() - start_time < DEADLINE and T > min_T:
            # Random move
            move_type = random.choice(['add', 'remove', 'swap'])

            if move_type == 'add' and len(sol) < 127:
                # Add a random element not in sol
                available = [x for x in all_candidates if x not in sol_set]
                if available:
                    new_elem = random.choice(available)
                    # Find a position to insert
                    # Binary search for insertion point
                    lo, hi = 0, len(sol)
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if sol[mid] < new_elem:
                            lo = mid + 1
                        else:
                            hi = mid
                    new_sol = sol[:lo] + [new_elem] + sol[lo:]
                    if is_valid(new_sol):
                        sol = new_sol
                        sol_set.add(new_elem)
                        if len(sol) > best_len:
                            best_len = len(sol)
                            best_seq = sol[:]
                    # Even if invalid, we might accept with probability (annealing)
                    # but for simplicity, only accept valid moves
            elif move_type == 'remove' and len(sol) > 1:
                # Remove a random element
                idx = random.randint(0, len(sol) - 1)
                removed = sol[idx]
                new_sol = sol[:idx] + sol[idx + 1:]
                # Always accept removal (reduces length but may allow better future adds)
                # But we want to maximize length, so only accept if it leads to improvement
                # For now, skip removals unless stuck
                if len(sol) > best_len + 5:  # only remove if we're well above best
                    sol = new_sol
                    sol_set.discard(removed)
            elif move_type == 'swap' and len(sol) >= 2:
                # Swap an element in sol with one not in sol
                available = [x for x in all_candidates if x not in sol_set]
                if available:
                    idx = random.randint(0, len(sol) - 1)
                    old_elem = sol[idx]
                    new_elem = random.choice(available)
                    new_sol = sol[:idx] + [new_elem] + sol[idx + 1:]
                    # Sort to maintain increasing order
                    new_sol.sort()
                    if is_valid(new_sol):
                        sol = new_sol
                        sol_set.discard(old_elem)
                        sol_set.add(new_elem)
                        if len(sol) > best_len:
                            best_len = len(sol)
                            best_seq = sol[:]

            T *= cooling

    # Final validation
    if not is_valid(best_seq):
        best_seq = greedy_construct()

    return best_seq
