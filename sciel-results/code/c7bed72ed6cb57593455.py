"""Reciprocal-subsum-free set construction for [1,24]."""

def construct():
    import time
    from itertools import combinations
    from fractions import Fraction
    from math import gcd
    
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Precompute reciprocals as Fractions for exact arithmetic
    recip = {i: Fraction(1, i) for i in range(1, 25)}
    
    def is_valid(subset):
        """Check if subset is reciprocal-subsum-free."""
        subset = list(subset)
        n = len(subset)
        if n <= 2:
            return True
        # For each element, check if its reciprocal equals sum of reciprocals of a subset of others
        for i in range(n):
            target = recip[subset[i]]
            others = [subset[j] for j in range(n) if j != i]
            # Check all subsets of others
            for r in range(1, len(others) + 1):
                for sub in combinations(others, r):
                    s = sum(recip[x] for x in sub)
                    if s == target:
                        return False
        return True
    
    # Greedy construction: try to add elements in an order that maximizes set size
    # Strategy: start with all elements, then remove conflicts
    # Better: greedy add with conflict checking
    
    def can_add(current_set, candidate):
        """Check if candidate can be added to current_set."""
        if candidate in current_set:
            return True
        # Check two conditions:
        # 1. candidate's reciprocal is not a sum of subset of current_set
        # 2. For each element in current_set, its reciprocal is not a sum of subset of (current_set + candidate) minus itself
        # Condition 1:
        target = recip[candidate]
        for r in range(1, len(current_set) + 1):
            for sub in combinations(current_set, r):
                if sum(recip[x] for x in sub) == target:
                    return False
        # Condition 2: for each element x in current_set, check if recip[x] can be formed by subset of (current_set - {x} + {candidate})
        for x in current_set:
            target_x = recip[x]
            others = [y for y in current_set if y != x] + [candidate]
            for r in range(1, len(others) + 1):
                for sub in combinations(others, r):
                    if sum(recip[x2] for x2 in sub) == target_x:
                        return False
        return True
    
    # Greedy: try multiple orderings
    best_set = set()
    
    # Try adding elements from 1 to 24 in various orders
    orders = [
        list(range(1, 25)),
        list(range(24, 0, -1)),
        # Interleave large and small
        [i for i in range(1, 25) if i % 2 == 1] + [i for i in range(1, 25) if i % 2 == 0],
        [i for i in range(1, 25) if i % 2 == 0] + [i for i in range(1, 25) if i % 2 == 1],
        # Sort by reciprocal value (which is just descending i)
        list(range(24, 0, -1)),
        # Try random-ish orders
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    ]
    
    for order in orders:
        if time.time() > DEADLINE - 5:
            break
        current = set()
        for elem in order:
            if time.time() > DEADLINE - 3:
                break
            if can_add(current, elem):
                current.add(elem)
        if len(current) > len(best_set):
            best_set = set(current)
    
    # Local improvement: try to add elements not in best_set
    if time.time() < DEADLINE - 5:
        improved = True
        while improved and time.time() < DEADLINE - 5:
            improved = False
            for elem in range(1, 25):
                if elem not in best_set and time.time() < DEADLINE - 5:
                    if can_add(best_set, elem):
                        best_set.add(elem)
                        improved = True
    
    # Try swapping: remove one, add another
    if time.time() < DEADLINE - 3:
        for remove in list(best_set):
            if time.time() > DEADLINE - 3:
                break
            temp = set(best_set) - {remove}
            for add in range(1, 25):
                if add not in temp and add != remove:
                    if time.time() > DEADLINE - 2:
                        break
                    if can_add(temp, add):
                        new_set = temp | {add}
                        if len(new_set) > len(best_set):
                            best_set = new_set
                            break
    
    if not best_set:
        return [1]
    
    return sorted(best_set)
