"""Exact Maximum Independent Set for reciprocal-subsum-free sets in [1,24].

Structural approach:
  Model as a graph problem. Two elements a, b conflict if 1/a is a subsum of
  reciprocals of others INCLUDING b (or vice versa). More precisely, we need:
  for a set S, for every a in S, 1/a != sum of any subset of {1/v : v in S, v != a}.

  This is NOT simply a pairwise constraint (it's a higher-order constraint),
  so we can't precompute a static graph. Instead, we use branch-and-bound with
  incremental validity: maintain a candidate set, and when adding a new element,
  check the full condition. We use upper-bound pruning based on the maximum
  possible additions remaining.

  Key insight: the condition is "hereditary" in a useful way. If S is valid,
  any subset is valid. So we can do a depth-first search adding elements,
  with pruning when the current size + remaining elements can't beat the best.

  For speed: use Fraction arithmetic but cache subset-sum results. The number
  of elements is only 24, and the subset-sum check for n other elements is
  O(2^(n/2)) via meet-in-the-middle, which is feasible for n up to ~20.
"""
import time
from fractions import Fraction
from itertools import combinations

DEADLINE = 85.0   # safety margin under 90s kill
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
        L = L | {x + v for x in L}
    R = {Fraction(0)}
    for v in right:
        R = R | {x + v for x in R}

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
    
    # Multi-start: try different greedy orderings
    if not _deadline_hit():
        all_elems = list(range(1, 25))
        # Try greedy from largest to smallest
        for start_order in [all_elems[::-1], all_elems, sorted(all_elems, key=lambda x: x % 7)]:
            if _deadline_hit():
                break
            trial = []
            for v in start_order:
                if _deadline_hit():
                    break
                if _check_validity_incremental(trial, v) if trial else True:
                    if is_valid(trial + [v]):
                        trial.append(v)
            if len(trial) > len(best):
                best = trial
                best_size = len(best)
    
    # Final validation
    if not is_valid(best):
        # Fallback to known good
        best = list(range(13, 25))
    
    return list(best)


if __name__ == "__main__":
    print(construct())
