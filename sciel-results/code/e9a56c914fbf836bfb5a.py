"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations
import random

def _lcm(a, b):
    return a * b // gcd(a, b)

def _num_omega(n):
    """Count distinct prime factors."""
    count = 0
    d = 2
    while d * d <= n:
        if n % d == 0:
            count += 1
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        count += 1
    return count

def construct():
    start_time = time.time()
    DEADLINE = 82.0  # safety margin before 90s kill

    n = 128
    universe = list(range(1, n + 1))
    universe_set = set(universe)

    # Precompute LCM table for all pairs
    lcm_table = {}
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[(i, j)] = _lcm(i, j)

    def get_lcm(a, b):
        if a <= b:
            return lcm_table[(a, b)]
        else:
            return lcm_table[(b, a)]

    # Precompute bad_pairs[x]: set of frozenset({a,b}) such that (a,b,x) is a bad triple
    # A triple (a,b,c) is bad if lcm(a,b)==lcm(a,c)==lcm(b,c)
    bad_pairs = [set() for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = get_lcm(a, b)
            for c in range(b + 1, n + 1):
                if get_lcm(a, c) == lab and get_lcm(b, c) == lab:
                    bad_pairs[c].add(frozenset((a, b)))
                    bad_pairs[b].add(frozenset((a, c)))
                    bad_pairs[a].add(frozenset((b, c)))

    def can_add(x, selected_set):
        """Check if x can be added to selected_set without creating a bad triple."""
        for pair in bad_pairs[x]:
            a, b = tuple(pair)
            if a in selected_set and b in selected_set:
                return False
        return True

    # Greedy with a given order
    def greedy(order):
        selected = []
        selected_set = set()
        for x in order:
            if can_add(x, selected_set):
                selected.append(x)
                selected_set.add(x)
        return selected

    # Local search: try to improve by removing a set of elements and adding others
    def local_search(initial_set, max_iterations=3000):
        selected = list(initial_set)
        selected_set = set(selected)
        best_size = len(selected)

        for iteration in range(max_iterations):
            if time.time() - start_time > DEADLINE:
                break

            improved = False

            # Try single swaps: remove one element, add one or more
            for remove_idx in range(len(selected)):
                if time.time() - start_time > DEADLINE:
                    break
                rem = selected[remove_idx]
                remaining = selected[:remove_idx] + selected[remove_idx + 1:]
                remaining_set = selected_set - {rem}

                # Try to add as many elements as possible from the remaining universe
                # Use a greedy fill: try to add elements that don't conflict
                added = []
                for x in universe:
                    if x in remaining_set:
                        continue
                    if can_add(x, remaining_set):
                        added.append(x)
                        remaining_set.add(x)

                new_selected = remaining + added
                if len(new_selected) > best_size:
                    selected = new_selected
                    selected_set = set(selected)
                    best_size = len(selected)
                    improved = True
                    break

            if not improved:
                # Try 2-swap: remove two elements, add more
                improved = False
                for i in range(len(selected)):
                    if time.time() - start_time > DEADLINE:
                        break
                    for j in range(i + 1, len(selected)):
                        rem1 = selected[i]
                        rem2 = selected[j]
                        remaining = [s for k, s in enumerate(selected) if k != i and k != j]
                        remaining_set = set(remaining)

                        added = []
                        for x in universe:
                            if x in remaining_set:
                                continue
                            if can_add(x, remaining_set):
                                added.append(x)
                                remaining_set.add(x)

                        new_selected = remaining + added
                        if len(new_selected) > best_size:
                            selected = new_selected
                            selected_set = set(selected)
                            best_size = len(selected)
                            improved = True
                            break
                    if improved:
                        break

            if not improved:
                break

        return selected

    # Generate multiple orderings
    def make_order(seed, strategy):
        if strategy == 0:
            # Natural order
            return list(universe)
        elif strategy == 1:
            # By number of prime factors
            return sorted(universe, key=lambda x: (_num_omega(x), x))
        elif strategy == 2:
            # Random
            random.seed(seed)
            order = list(universe)
            random.shuffle(order)
            return order
        elif strategy == 3:
            # Reverse natural
            return list(reversed(universe))
        elif strategy == 4:
            # By value descending, then omega
            return sorted(universe, key=lambda x: (-_num_omega(x), x))
        elif strategy == 5:
            # Random with bias toward low omega
            random.seed(seed)
            weights = [1.0 / (1 + _num_omega(x)) for x in universe]
            order = []
            remaining = list(universe)
            remaining_weights = weights[:]
            while remaining:
                total = sum(remaining_weights)
                r = random.random() * total
                cum = 0
                for idx, w in enumerate(remaining_weights):
                    cum += w
                    if cum >= r:
                        order.append(remaining[idx])
                        remaining.pop(idx)
                        remaining_weights.pop(idx)
                        break
            return order
        return list(universe)

    # Run multiple greedy passes with different strategies
    best_result = []
    best_size = 0

    strategies = [0, 1, 3, 4]
    for s in strategies:
        if time.time() - start_time > DEADLINE:
            break
        order = make_order(0, s)
        result = greedy(order)
        if len(result) > best_size:
            best_size = len(result)
            best_result = result

    # Randomized greedy passes
    for i in range(30):
        if time.time() - start_time > DEADLINE - 5:
            break
        order = make_order(i * 12345 + 67890, 2)
        result = greedy(order)
        if len(result) > best_size:
            best_size = len(result)
            best_result = result

    # Biased random greedy passes
    for i in range(15):
        if time.time() - start_time > DEADLINE - 5:
            break
        order = make_order(i * 99999 + 1, 5)
        result = greedy(order)
        if len(result) > best_size:
            best_size = len(result)
            best_result = result

    # Local search on the best result
    if time.time() - start_time < DEADLINE - 3:
        improved = local_search(best_result, max_iterations=5000)
        if len(improved) > best_size:
            best_size = len(improved)
            best_result = improved

    # Additional local search on other good results
    if time.time() - start_time < DEADLINE - 5:
        for attempt in range(10):
            if time.time() - start_time > DEADLINE - 3:
                break
            order = make_order(100000 + attempt * 7777, 2)
            result = greedy(order)
            if len(result) >= best_size - 2:  # only if close to best
                improved = local_search(result, max_iterations=3000)
                if len(improved) > best_size:
                    best_size = len(improved)
                    best_result = improved

    # Final local search polish
    if time.time() - start_time < DEADLINE - 1:
        improved = local_search(best_result, max_iterations=2000)
        if len(improved) > best_size:
            best_size = len(improved)
            best_result = improved

    # Ensure nonempty
    if not best_result:
        best_result = [1]

    return best_result
# EVOLVE-BLOCK-END
