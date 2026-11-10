import math
import random
import time
from itertools import combinations
from fractions import Fraction

def construct():
    """
    Return a nonempty subset of [1..22] such that ALL reciprocal subset sums,
    including the empty sum, are distinct. Maximize cardinality.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    # LCM of 1..22 for converting reciprocal sums to integers
    lcm_all = 232792560  # LCM(1,2,...,22)
    
    # Precompute reciprocal values as integers (1/i * LCM)
    recip_int = {i: lcm_all // i for i in range(1, 23)}

    def check_dissociated(elems):
        """
        Check if the set of elements (1-based values 1..22)
        has all distinct reciprocal subset sums.
        Uses integer arithmetic for speed.
        Returns True if dissociated, False otherwise.
        """
        k = len(elems)
        if k == 0:
            return True

        seen = {0}
        current_sums = [0]

        for e in elems:
            v = recip_int[e]
            new_sums = []
            for s in current_sums:
                ns = s + v
                if ns in seen:
                    return False
                seen.add(ns)
                new_sums.append(ns)
            current_sums.extend(new_sums)

        return True

    def greedy_build(ordering):
        """
        Greedily build a dissociated set following the given ordering.
        Returns the set found.
        """
        S = []
        cur_sums = {0}
        for e in ordering:
            if time.time() > deadline:
                break
            v = recip_int[e]
            # Check collision: for any existing sum s, is s + v already in cur_sums?
            collision = False
            for s in cur_sums:
                if (s + v) in cur_sums:
                    collision = True
                    break
            if not collision:
                S.append(e)
                cur_sums = cur_sums | {s + v for s in cur_sums}
        return S

    def greedy_remove(heuristic_func):
        """
        Start with all 22 elements, repeatedly find a collision and remove
        one element involved, using the given heuristic.
        """
        if time.time() > deadline:
            return None

        S = list(range(1, 23))

        while True:
            if time.time() > deadline:
                return S if S else None

            k = len(S)
            if k == 0:
                return None

            # Check if S is dissociated
            if check_dissociated(S):
                return S

            # Find a collision: two different subsets with the same sum
            vals = [recip_int[e] for e in S]

            # Build subset sums with tracking of masks
            sum_to_mask = {0: 0}
            current = [(0, 0)]

            collision_found = False
            coll_elements = None

            for i in range(k):
                v = vals[i]
                new_entries = []
                for s, m in current:
                    ns = s + v
                    nm = m | (1 << i)
                    if ns in sum_to_mask:
                        # Collision!
                        m1 = nm
                        m2 = sum_to_mask[ns]
                        # Elements involved are those in the symmetric difference
                        diff_mask = m1 ^ m2
                        coll_local = []
                        for j in range(k):
                            if (diff_mask >> j) & 1:
                                coll_local.append(S[j])
                        coll_elements = coll_local
                        collision_found = True
                        break
                    else:
                        sum_to_mask[ns] = nm
                        new_entries.append((ns, nm))
                current.extend(new_entries)
                if collision_found:
                    break

            if not collision_found or not coll_elements:
                # Shouldn't happen, but return current set
                return S

            # Choose which element to remove
            x = heuristic_func(coll_elements)
            if x in S:
                S.remove(x)

        return S if S else None

    # Heuristic functions
    def remove_smallest(coll):
        """Remove element with smallest value (largest reciprocal)"""
        return min(coll)

    def remove_largest(coll):
        """Remove element with largest value (smallest reciprocal)"""
        return max(coll)

    def remove_random(coll):
        """Remove a random element from collision"""
        return random.choice(coll)

    best_set = [1]  # At minimum, a single element is always valid

    # Strategy 1: Greedy build with multiple orderings
    orderings = [
        list(range(1, 23)),           # Ascending
        list(range(22, 0, -1)),       # Descending
        list(range(2, 23)),           # Skip 1, ascending
        list(range(22, 1, -1)),       # Skip 1, descending
        list(range(11, 0, -1)) + list(range(12, 23)),  # Odd-ish pattern
        list(range(2, 23, 2)) + list(range(1, 23, 2)), # Evens then odds
    ]

    for ordering in orderings:
        if time.time() > deadline:
            break
        result = greedy_build(ordering)
        if result and len(result) > len(best_set):
            best_set = result

    # Random orderings
    for _ in range(15):
        if time.time() > deadline - 5:
            break
        order = list(range(1, 23))
        random.shuffle(order)
        result = greedy_build(order)
        if result and len(result) > len(best_set):
            best_set = result

    # Strategy 2: Greedy remove with multiple heuristics
    strategies = [
        ("smallest", remove_smallest),
        ("largest", remove_largest),
    ]

    for name, func in strategies:
        if time.time() > deadline:
            break
        result = greedy_remove(func)
        if result and len(result) > len(best_set):
            best_set = result

    # Random restarts with greedy remove
    for _ in range(10):
        if time.time() > deadline:
            break
        result = greedy_remove(remove_random)
        if result and len(result) > len(best_set):
            best_set = result

    # Strategy 3: Expand best_set by trying to add remaining elements
    if time.time() < deadline - 3:
        remaining = [e for e in range(1, 23) if e not in best_set]
        remaining.sort(reverse=True)
        
        cur_sums = {0}
        for e in best_set:
            v = recip_int[e]
            cur_sums = cur_sums | {s + v for s in cur_sums}
        
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            for e in remaining[:]:
                if time.time() > deadline - 2:
                    break
                v = recip_int[e]
                collision = False
                for s in cur_sums:
                    if (s + v) in cur_sums:
                        collision = True
                        break
                if not collision:
                    best_set.append(e)
                    cur_sums = cur_sums | {s + v for s in cur_sums}
                    remaining.remove(e)
                    changed = True

    # Strategy 4: Swap-based local search
    if time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            remaining = [e for e in range(1, 23) if e not in best_set]
            for i, e_remove in enumerate(best_set[:]):
                if time.time() > deadline - 1:
                    break
                for e_add in remaining:
                    if e_add in best_set:
                        continue
                    candidate = best_set[:i] + best_set[i+1:] + [e_add]
                    if check_dissociated(candidate):
                        # Try to expand this candidate
                        rem2 = [e for e in range(1, 23) if e not in candidate]
                        rem2.sort(reverse=True)
                        cur_sums2 = {0}
                        for e in candidate:
                            v = recip_int[e]
                            cur_sums2 = cur_sums2 | {s + v for s in cur_sums2}
                        for f in rem2:
                            if time.time() > deadline - 1:
                                break
                            v = recip_int[f]
                            collision = False
                            for s in cur_sums2:
                                if (s + v) in cur_sums2:
                                    collision = True
                                    break
                            if not collision:
                                candidate.append(f)
                                cur_sums2 = cur_sums2 | {s + v for s in cur_sums2}
                        if len(candidate) > len(best_set):
                            best_set = candidate
                            improved = True
                            break
                if improved:
                    break

    # Final safety: ensure best_set is nonempty
    if not best_set:
        best_set = [1]

    # Verify final result
    if not check_dissociated(best_set):
        # Fallback to a known valid set
        best_set = [2, 3, 5, 7, 11, 13, 17, 19, 22]
        if not check_dissociated(best_set):
            best_set = [1]

    return sorted(best_set)
