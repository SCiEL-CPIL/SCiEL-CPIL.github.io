"""Rewritten program for large integer sets with no two-other-product divisibility."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    # Candidate set: all integers from 2 to 256
    all_candidates = list(range(2, 257))
    n_total = len(all_candidates)

    # Precompute prime factorizations for all numbers 2..256
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

    factor_cache = {}
    for x in range(2, 257):
        factor_cache[x] = factorize(x)

    def divides_product(a, b, c):
        """Check if a divides b*c using prime factorizations."""
        fa = factor_cache[a]
        fb = factor_cache.get(b, {})
        fc = factor_cache.get(c, {})
        # For each prime in a, check if exponent in b*c >= exponent in a
        for p, ea in fa.items():
            eb = fb.get(p, 0)
            ec = fc.get(p, 0)
            if eb + ec < ea:
                return False
        return True

    def can_add_fast(candidate, current_list):
        """Fast check: can we add candidate to current_list?
        Condition: for all distinct a,b,c in {candidate} ∪ current_list, a ∤ b*c.
        This means:
        1. candidate ∤ b*c for all distinct b,c in current_list
        2. for each a in current_list: a ∤ candidate * x for all x in current_list, x != a
        """
        n = len(current_list)
        # Check 1: candidate does not divide b*c for any distinct b,c in current_list
        for i in range(n):
            for j in range(i + 1, n):
                if divides_product(candidate, current_list[i], current_list[j]):
                    return False
        # Check 2: for each a in current_list, a does not divide candidate * x for any x != a
        for i in range(n):
            a = current_list[i]
            for j in range(n):
                if j == i:
                    continue
                if divides_product(a, candidate, current_list[j]):
                    return False
        return True

    def is_valid_set(s):
        """Check if set s satisfies the condition."""
        s_list = list(s)
        n = len(s_list)
        for i in range(n):
            a = s_list[i]
            for j in range(n):
                if j == i:
                    continue
                for k in range(j + 1, n):
                    if k == i:
                        continue
                    b = s_list[j]
                    c = s_list[k]
                    if (b * c) % a == 0:
                        return False
        return True

    # Greedy approach with multiple orderings
    best_set = set()
    best_size = 0

    # Try different orderings
    orderings = [
        sorted(all_candidates, reverse=True),  # descending
        sorted(all_candidates),  # ascending
        list(all_candidates),  # natural order
    ]

    # Add more random orderings for diversity
    random.seed(42)
    for _ in range(80):
        shuffled = all_candidates[:]
        random.shuffle(shuffled)
        orderings.append(shuffled)

    for ordering in orderings:
        if time.time() > deadline - 10:
            break
        current = set()
        current_list = []
        for c in ordering:
            if can_add_fast(c, current_list):
                current.add(c)
                current_list.append(c)
        if len(current) > best_size:
            best_set = current
            best_size = len(current)

    # Local search: try to add elements (greedy fill)
    not_in_best = [x for x in all_candidates if x not in best_set]
    best_list = list(best_set)
    improved = True
    while improved:
        improved = False
        still_not_in = []
        for c in not_in_best:
            if time.time() > deadline - 10:
                still_not_in.append(c)
                continue
            if can_add_fast(c, best_list):
                best_set.add(c)
                best_list.append(c)
                improved = True
                best_size = len(best_set)
            else:
                still_not_in.append(c)
        not_in_best = still_not_in
        if time.time() > deadline - 10:
            break

    # Local search: 1-for-1 swap (remove one, try to add a different one)
    if time.time() < deadline - 15:
        best_list = list(best_set)
        for i in range(len(best_list)):
            if time.time() > deadline - 10:
                break
            elem_to_remove = best_list[i]
            test_set = set(best_set)
            test_set.discard(elem_to_remove)
            test_list = [x for x in best_list if x != elem_to_remove]
            # Try each candidate not in test_set
            for c in all_candidates:
                if c in test_set:
                    continue
                if can_add_fast(c, test_list):
                    # Found a valid swap
                    best_set = test_set | {c}
                    best_list = test_list + [c]
                    improved = True
                    break
            if time.time() > deadline - 10:
                break

    # Additional 1-for-1 swap passes
    for pass_num in range(3):
        if time.time() > deadline - 10:
            break
        best_list = list(best_set)
        any_swap = False
        for i in range(len(best_list)):
            if time.time() > deadline - 10:
                break
            elem_to_remove = best_list[i]
            test_set = set(best_set)
            test_set.discard(elem_to_remove)
            test_list = [x for x in best_list if x != elem_to_remove]
            # Try each candidate not in test_set
            for c in all_candidates:
                if c in test_set:
                    continue
                if can_add_fast(c, test_list):
                    # Found a valid swap
                    best_set = test_set | {c}
                    best_list = test_list + [c]
                    any_swap = True
                    break
            if time.time() > deadline - 10:
                break
        if not any_swap:
            break

    # Local search: 1-for-2 swap (remove one, try to add two others)
    if time.time() < deadline - 10:
        best_list = list(best_set)
        for i in range(len(best_list)):
            if time.time() > deadline - 10:
                break
            elem_to_remove = best_list[i]
            test_set = set(best_set)
            test_set.discard(elem_to_remove)
            test_list = [x for x in best_list if x != elem_to_remove]
            candidates_to_add = [x for x in all_candidates if x not in test_set]
            added = []
            for c in candidates_to_add:
                if time.time() > deadline - 10:
                    break
                if len(added) >= 2:
                    break
                if can_add_fast(c, test_list + added):
                    added.append(c)
            if len(added) >= 2:
                new_set = test_set | set(added)
                if len(new_set) > len(best_set):
                    best_set = new_set
                    best_size = len(best_set)
                    best_list = list(best_set)

    # Second round of greedy fill after swaps
    if time.time() < deadline - 10:
        best_list = list(best_set)
        not_in_best = [x for x in all_candidates if x not in best_set]
        improved = True
        while improved:
            improved = False
            still_not_in = []
            for c in not_in_best:
                if time.time() > deadline - 5:
                    still_not_in.append(c)
                    continue
                if can_add_fast(c, best_list):
                    best_set.add(c)
                    best_list.append(c)
                    improved = True
                    best_size = len(best_set)
                else:
                    still_not_in.append(c)
            not_in_best = still_not_in
            if time.time() > deadline - 5:
                break

    # Final verification
    result = sorted(best_set)
    if not is_valid_set(result):
        # Fallback: try to build a valid set from scratch
        current = set()
        current_list = []
        for c in all_candidates:
            if can_add_fast(c, current_list):
                current.add(c)
                current_list.append(c)
        result = sorted(current)

    # Ensure we return a list of Python ints
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
