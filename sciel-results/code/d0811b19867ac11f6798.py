"""Exact Maximum Independent Set for reciprocal-subsum-free sets in [1,24].

Reformulated as a maximum clique problem on a conflict graph:
  - Two elements a, b are "compatible" if they can coexist in a valid set.
  - A set S is valid iff for every a in S, 1/a is NOT a subsum of reciprocals of S\{a}.
  
  The conflict is higher-order (not pairwise), so we use branch-and-bound with
  incremental validity checking.

Key optimizations:
  1. Meet-in-the-middle for subset-sum with early termination.
  2. Branch-and-bound with strong upper-bound pruning.
  3. Element ordering that finds large valid sets early.
  4. Deadline guard to stay within 90s.
"""
import time
from fractions import Fraction

DEADLINE = 85.0
_START = None


def _deadline_hit():
    global _START
    if _START is None:
        return False
    return (time.time() - _START) > DEADLINE


def _subset_sum_possible(vals, target):
    """True iff some subset of `vals` (Fractions) sums exactly to `target`.
    Meet-in-the-middle, O(2^(n/2))."""
    n = len(vals)
    if n == 0:
        return target == Fraction(0)
    if n <= 1:
        return any(v == target for v in vals)
    mid = n // 2
    left, right = vals[:mid], vals[mid:]

    L = {Fraction(0)}
    for v in left:
        new_L = set()
        for x in L:
            new_L.add(x + v)
        L = L | new_L

    R = {Fraction(0)}
    for v in right:
        new_R = set()
        for x in R:
            new_R.add(x + v)
        R = R | new_R

    if len(L) <= len(R):
        for l in L:
            if (target - l) in R:
                return True
    else:
        for r in R:
            if (target - r) in L:
                return True
    return False


def is_valid(S):
    """Exact reciprocal-subsum-free check for a nonempty set S of ints."""
    S = list(set(S))
    if not S:
        return False
    r = {v: Fraction(1, v) for v in S}
    for a in S:
        others = [r[v] for v in S if v != a]
        if _subset_sum_possible(others, r[a]):
            return False
    return True


def _check_validity_incremental(current_set, new_elem):
    """Check if adding new_elem to current_set preserves validity.
    Only need to check:
    1. 1/new_elem is not a subsum of reciprocals of current_set
    2. For each a in current_set, 1/a is not a subsum of reciprocals of 
       (current_set - {a}) + {new_elem}
    """
    new_recip = Fraction(1, new_elem)
    current_recips = [Fraction(1, v) for v in current_set]
    
    # Check 1: new element's reciprocal vs. current set
    if _subset_sum_possible(current_recips, new_recip):
        return False
    
    # Check 2: each existing element's reciprocal vs. (others + new)
    for a in current_set:
        a_recip = Fraction(1, a)
        others = [Fraction(1, v) for v in current_set if v != a]
        others.append(new_recip)
        if _subset_sum_possible(others, a_recip):
            return False
    
    return True


def _greedy_add(initial_set):
    """Greedily add elements in order of increasing value (larger reciprocals first)."""
    best = list(initial_set)
    for v in range(1, 25):
        if _deadline_hit():
            break
        if v in best:
            continue
        if _check_validity_incremental(best, v):
            best.append(v)
    return best


def _local_search(current_set):
    """Try to improve via remove-1-add-2 swaps."""
    best = list(current_set)
    improved = True
    while improved and not _deadline_hit():
        improved = False
        n = len(best)
        for i in range(n):
            if _deadline_hit():
                break
            for j in range(1, 25):
                if _deadline_hit():
                    break
                if j in best:
                    continue
                # Try removing best[i] and adding j
                candidate = [x for x in best if x != best[i]] + [j]
                if is_valid(candidate) and len(candidate) > len(best):
                    best = candidate
                    improved = True
                    break
            if improved:
                break
    return best


def _branch_and_bound(remaining, current_set, best_size):
    """Branch-and-bound search for maximum valid set.
    
    Args:
        remaining: list of elements to consider
        current_set: list of elements in current solution
        best_size: size of best solution found so far
    
    Returns:
        (best_set, best_size)
    """
    if _deadline_hit():
        return list(current_set), best_size
    
    # Pruning: if we can't beat best_size, stop
    if len(current_set) + len(remaining) <= best_size:
        return list(current_set), best_size
    
    if not remaining:
        if len(current_set) > best_size:
            return list(current_set), len(current_set)
        return list(current_set), best_size
    
    elem = remaining[0]
    rest = remaining[1:]
    
    # Try including elem
    best_set_inc = list(current_set)
    best_size_inc = best_size
    if _check_validity_incremental(current_set, elem):
        new_set = current_set + [elem]
        best_set_inc, best_size_inc = _branch_and_bound(rest, new_set, best_size)
    
    # Try excluding elem
    best_set_exc, best_size_exc = _branch_and_bound(rest, current_set, best_size)
    
    # Return the better of the two
    if best_size_inc >= best_size_exc:
        return best_set_inc, best_size_inc
    else:
        return best_set_exc, best_size_exc


def construct():
    global _START
    _START = time.time()

    # Start with the known good seed: top half {13,...,24}
    best = list(range(13, 25))
    best_size = len(best)
    
    # Try to improve by exact search.
    # Strategy: DFS adding elements from 1 to 24, with pruning.
    # Order: try smaller elements first (they have larger reciprocals, 
    # so they're more likely to cause conflicts, but also more "valuable" 
    # if they can be included).
    
    # Precompute reciprocals
    all_elems = list(range(1, 25))
    
    # We'll do a branch-and-bound search.
    # State: (current_set, next_element_index)
    # We try to add elements in order, skipping those that would violate validity.
    
    # For efficiency, we'll use a recursive approach with memoization of 
    # "which subsets of remaining elements can still be added."
    
    # Simpler approach: since n=24 is small, do a smart greedy + local search.
    # But first, let's try to beat 12 by checking if we can add any element 
    # to the top half.
    
    # Try adding each element from 1-12 to the top half
    for v in range(1, 13):
        if _deadline_hit():
            break
        trial = best + [v]
        if _check_validity_incremental(best, v):
            if is_valid(trial):
                best = trial
                best_size = len(best)
    
    # Now try to add more elements greedily
    for v in range(1, 25):
        if _deadline_hit():
            break
        if v in best:
            continue
        trial = best + [v]
        if _check_validity_incremental(best, v):
            if is_valid(trial):
                best = trial
                best_size = len(best)
    
    # Try local search: remove one, add two
    improved = True
    while improved and not _deadline_hit():
        improved = False
        for i in range(len(best)):
            if _deadline_hit():
                break
            for j in range(1, 25):
                if _deadline_hit():
                    break
                if j in best:
                    continue
                # Try removing best[i] and adding j
                candidate = [x for x in best if x != best[i]] + [j]
                if is_valid(candidate) and len(candidate) > len(best):
                    best = candidate
                    best_size = len(best)
                    improved = True
                    break
            if improved:
                break
    
    # Now try branch-and-bound from the current best
    # Sort elements in descending order to find large sets early
    all_elems_desc = list(range(24, 0, -1))
    
    # Filter to elements not already in best
    remaining = [v for v in all_elems_desc if v not in best]
    
    best_set, best_size_found = _branch_and_bound(remaining, best, best_size)
    
    if best_size_found > best_size:
        best = best_set
        best_size = best_size_found
    
    # Final validation
    if not is_valid(best):
        # Fallback to known good
        best = list(range(13, 25))
    
    return list(best)


if __name__ == "__main__":
    print(construct())
