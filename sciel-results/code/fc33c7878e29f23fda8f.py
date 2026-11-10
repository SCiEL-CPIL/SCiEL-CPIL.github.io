"""Rewritten program: direct construction of a large dissociated reciprocal set using exact arithmetic."""
import time
from fractions import Fraction

def _subset_sums_set(elems):
    """Return the set of all subset sums of reciprocals of elems (as Fractions)."""
    sums = {Fraction(0, 1)}
    for e in elems:
        r = Fraction(1, e)
        new_sums = set()
        for s in sums:
            new_sums.add(s + r)
        sums = sums | new_sums
    return sums

def _is_dissociated(elems):
    """Check if all subset sums of reciprocals are distinct using exact arithmetic."""
    if not elems:
        return True
    n = len(elems)
    expected = (1 << n)
    sums = _subset_sums_set(elems)
    return len(sums) == expected

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin within 90s limit
    
    universe = list(range(1, 23))
    
    best_set = []
    best_size = 0
    
    # Try multiple greedy orderings
    # Larger elements first (smaller reciprocals) is often better for dissociation
    orderings = [
        sorted(universe, reverse=True),
        sorted(universe),
        list(range(22, 1, -1)), # Skip 1
        list(range(2, 23)),     # Skip 1
    ]
    
    for ordering in orderings:
        if time.time() > deadline - 10:
            break
        cur = []
        cur_sums = {Fraction(0, 1)}
        for e in ordering:
            r = Fraction(1, e)
            # Check collision: for any existing sum s, is s + r already in cur_sums?
            collision = False
            # Optimization: check if any s + r is in cur_sums
            # This is O(2^k) per element, which is fine for k up to ~20
            for s in cur_sums:
                if (s + r) in cur_sums:
                    collision = True
                    break
            if not collision:
                cur.append(e)
                cur_sums = cur_sums | {s + r for s in cur_sums}
        
        if len(cur) > best_size:
            best_size = len(cur)
            best_set = cur[:]
    
    # Now try to expand best_set by adding remaining elements
    remaining = [e for e in universe if e not in best_set]
    # Sort remaining by descending value (smaller reciprocals)
    remaining.sort(reverse=True)
    
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for e in remaining[:]:
            if time.time() > deadline - 5:
                break
            test = best_set + [e]
            if _is_dissociated(test):
                best_set = test
                remaining.remove(e)
                changed = True
    
    # Try swapping: replace one element with a better one
    if time.time() < deadline - 3:
        for i in range(len(best_set)):
            if time.time() > deadline - 3:
                break
            for e in remaining[:]:
                candidate = best_set[:i] + best_set[i+1:] + [e]
                if _is_dissociated(candidate):
                    # Check if this swap allows further expansion
                    new_remaining = [x for x in universe if x not in candidate]
                    new_remaining.sort(reverse=True)
                    cur = candidate[:]
                    cur_sums = _subset_sums_set(cur)
                    for f in new_remaining:
                        if time.time() > deadline - 3:
                            break
                        r = Fraction(1, f)
                        collision = False
                        for s in cur_sums:
                            if (s + r) in cur_sums:
                                collision = True
                                break
                        if not collision:
                            cur.append(f)
                            cur_sums = cur_sums | {s + r for s in cur_sums}
                    if len(cur) > len(best_set):
                        best_set = cur
                        remaining = [x for x in universe if x not in best_set]
                        break
    
    # Verify
    if best_set and _is_dissociated(best_set):
        return sorted(best_set)
    
    # Fallback: a known good set (primes are often dissociated)
    return [2, 3, 5, 7, 11, 13, 17, 19, 22]
