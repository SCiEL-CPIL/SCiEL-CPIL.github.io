"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
from fractions import Fraction
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # We want to find the largest subset S of [1,24] such that
    # for every a in S, 1/a != sum(1/b for b in T) for any subset T of S\{a}
    
    # Greedy: try adding elements from largest to smallest
    # Larger elements have smaller reciprocals, harder to form as sums
    
    def is_valid_subset(S):
        """Check if subset S is valid (no reciprocal subsums)."""
        S_set = set(S)
        for a in S_set:
            target = Fraction(1, a)
            others = [b for b in S_set if b != a]
            if _subset_sum_exists(others, target):
                return False
        return True
    
    def _subset_sum_exists(elements, target):
        """Check if any subset of elements' reciprocals sums to target."""
        # elements is list of ints, target is Fraction
        # Use DP with a set of achievable sums
        # But this could be exponential... use memoized subset sum
        
        # For small sets, brute force with pruning
        n = len(elements)
        if n == 0:
            return target == 0
        
        # Use a set of achievable sums
        achievable = {Fraction(0)}
        for elem in elements:
            recip = Fraction(1, elem)
            new_achievable = set(achievable)
            for s in achievable:
                ns = s + recip
                if ns <= target:  # prune: all reciprocals positive
                    new_achievable.add(ns)
            achievable = new_achievable
            if target in achievable:
                return True
        return target in achievable
    
    # Greedy construction: add elements in descending order
    # This tends to work well because larger elements are "safer"
    
    def greedy_construct(order):
        """Try to build a valid set by adding elements in given order."""
        selected = []
        for elem in order:
            candidate = selected + [elem]
            # Only need to check if elem's reciprocal can be formed by others,
            # and if any existing element's reciprocal can now be formed
            # Actually, adding elem could create new subsums for existing elements
            # So we need full check, but we can be smart:
            # - Check if 1/elem = sum of subset of reciprocals of selected
            # - Check if for each a in selected, 1/a = sum of subset of reciprocals of (selected \ {a} + [elem])
            #   The only new possibility is using elem in the subset
            
            # Check 1/elem
            if _subset_sum_exists(selected, Fraction(1, elem)):
                continue
            
            # Check if adding elem creates a subsum for any existing element
            valid = True
            for a in selected:
                # 1/a could now be formed using elem
                # Previously it wasn't formed by subset of (selected \ {a})
                # Now check subset of (selected \ {a}) + [elem]
                # This means: 1/a = sum(subset of selected\{a}) OR 1/a = sum(subset of selected\{a}) + 1/elem
                # The second case: 1/a - 1/elem = sum(subset of selected\{a})
                target_without_elem = Fraction(1, a) - Fraction(1, elem)
                if target_without_elem > 0:
                    others_without_a = [b for b in selected if b != a]
                    if _subset_sum_exists(others_without_a, target_without_elem):
                        valid = False
                        break
            if valid:
                selected.append(elem)
        return selected
    
    # Try different orderings
    # Strategy: descending order first (largest elements)
    order_desc = list(range(24, 0, -1))
    best = greedy_construct(order_desc)
    
    # Also try ascending
    order_asc = list(range(1, 25))
    best_asc = greedy_construct(order_asc)
    if len(best_asc) > len(best):
        best = best_asc
    
    # Try mixed orders
    # Start with large, then fill in small ones that fit
    order_mixed = list(range(24, 12, -1)) + list(range(12, 0, -1))
    best_mixed = greedy_construct(order_mixed)
    if len(best_mixed) > len(best):
        best = best_mixed
    
    # Now try to improve by local search / backtracking
    # Try to add elements that weren't included
    remaining = [x for x in range(1, 25) if x not in best]
    
    # Try adding each remaining element and see if we can swap
    improved = list(best)
    for elem in remaining:
        if time.time() > deadline - 5:
            break
        # Try adding elem to current set
        candidate = improved + [elem]
        if is_valid_subset(candidate):
            improved = candidate
    
    if len(improved) > len(best):
        best = improved
    
    # Backtracking search for better solutions
    # Use the greedy result as a starting point and try to find larger sets
    
    def backtrack_search(remaining_elements, current_set, best_size):
        """DFS to find larger valid sets."""
        nonlocal best
        if time.time() > deadline - 3:
            return
        if len(current_set) >= len(best):
            # Try to extend
            pass
        if not remaining_elements:
            return
        if len(current_set) + len(remaining_elements) <= len(best):
            return
        
        for i, elem in enumerate(remaining_elements):
            if time.time() > deadline - 3:
                return
            candidate = current_set + [elem]
            if is_valid_subset(candidate):
                if len(candidate) > len(best):
                    best = candidate
                # Recurse
                backtrack_search(remaining_elements[i+1:], candidate, len(best))
            # Also try skipping elem
            # But we should prioritize adding
        
        # Prune: if we can't beat best
        if len(current_set) + len(remaining_elements) <= len(best):
            return
    
    # Try backtracking from a smaller base
    if len(best) < 20:
        # Try to find a 20+ element set
        # Start with large elements and try to add small ones
        base = [x for x in range(13, 25)]  # 12 elements
        small_elements = list(range(1, 13))
        
        # Try to add small elements one by one
        for elem in small_elements:
            if time.time() > deadline - 5:
                break
            candidate = base + [elem]
            if is_valid_subset(candidate):
                base = candidate
        
        if len(base) > len(best):
            best = base
    
    # Final attempt: systematic search with time limit
    if time.time() < deadline - 10 and len(best) < 22:
        # Try all subsets of a certain size using a smarter approach
        # Start from the best found and try to swap elements
        
        # Try removing one element and adding two others
        best_set = set(best)
        for a in list(best_set):
            if time.time() > deadline - 5:
                break
            reduced = [x for x in best if x != a]
            for b in range(1, 25):
                if time.time() > deadline - 5:
                    break
                if b in best_set:
                    continue
                for c in range(b+1, 25):
                    if time.time() > deadline - 5:
                        break
                    if c in best_set:
                        continue
                    candidate = reduced + [b, c]
                    if len(candidate) > len(best):
                        if is_valid_subset(candidate):
                            best = candidate
    
    # Ensure nonempty
    if not best:
        best = [24]
    
    return list(best)

# EVOLVE-BLOCK-END
