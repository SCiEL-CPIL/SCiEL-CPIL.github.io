"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from math import lcm
from functools import lru_cache
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Return a nonempty subset of [1,22] such that ALL reciprocal subset sums,
    including the empty sum, are distinct. Maximize cardinality.
    """
    deadline = time.time() + 85  # Safety margin before 90s limit
    
    # We want to find the largest subset S of {1,...,22} such that
    # all subset sums of {1/s : s in S} are distinct.
    # This is equivalent to: for any two distinct subsets A, B of S,
    # sum(1/a for a in A) != sum(1/b for b in B).
    
    # Scale by LCM of [1..22] to work with integers.
    # LCM(1..22) is large but manageable.
    # Actually, let's use a different approach: work with fractions or use
    # a common denominator that's the LCM of the elements in the candidate set.
    
    # Strategy: Try to find the maximum k such that there exists a subset
    # of size k from [1..22] with all reciprocal subset sums distinct.
    # We'll use a greedy/backtracking approach with pruning.
    
    # First, let's establish that a set is "dissociated" by checking
    # all 2^k subset sums. For k up to ~12, 2^12 = 4096, which is fine.
    # For larger k, we need smarter checks.
    
    def is_dissociated(elements):
        """Check if all reciprocal subset sums of elements are distinct."""
        k = len(elements)
        if k == 0:
            return True
        
        # Use integer arithmetic: multiply each 1/e by LCM of elements
        # to avoid floating point issues.
        # Actually, let's use Python's fractions for exactness.
        from fractions import Fraction
        
        subset_sums = set()
        # Generate all 2^k subset sums
        for mask in range(1 << k):
            s = Fraction(0, 1)
            for i in range(k):
                if mask & (1 << i):
                    s += Fraction(1, elements[i])
            if s in subset_sums:
                return False
            subset_sums.add(s)
        return True
    
    # Try sizes from 22 down to 1
    # For each size, try to find a valid subset using backtracking with pruning
    
    # Precompute: for efficiency, we can try combinations in a smart order.
    # Larger elements have smaller reciprocals, making collisions less likely
    # with small elements... actually it's complex. Let's just try.
    
    # We'll use a recursive backtracking approach that builds the set incrementally
    # and checks at each step whether the partial set is still potentially extendable.
    
    # For a set of size k, we have 2^k subset sums. If any two collide, it's invalid.
    # Key insight: if we build the set one element at a time, we can check
    # whether adding a new element creates a collision.
    
    # When we add element x to set S, the new subset sums are:
    # old sums: A = {sums of subsets of S}
    # new sums: B = {s + 1/x for s in A}
    # We need A ∩ B = ∅ and A ∪ B has no duplicates within itself.
    # Since A was already distinct, we just need:
    # 1. All elements of B are distinct (they are, since s -> s+1/x is injective)
    # 2. A ∩ B = ∅, i.e., for no a1, a2 in A do we have a1 = a2 + 1/x
    
    # So the check when adding x to S is:
    # For all pairs (a1, a2) in A × A: a1 - a2 != 1/x
    # Equivalently: 1/x is not in {a1 - a2 : a1, a2 in A, a1 != a2}
    # But also we need a1 != a2 + 1/x, which is the same as a1 - a2 != 1/x.
    # Note: a1 = a2 gives 0, and 1/x != 0, so we only need a1 != a2.
    
    # This is O(4^k) per check if we do it naively, but we can maintain
    # the set of differences.
    
    def can_add(current_sums, x):
        """Check if element x can be added given current subset sums."""
        from fractions import Fraction
        inv_x = Fraction(1, x)
        # Check if inv_x is in the set of differences of current_sums
        # differences = {a - b : a, b in current_sums, a != b}
        # But computing all differences is O(n^2) where n = 2^k
        # For k up to ~11, n up to 2048, n^2 = 4M, might be slow in Python.
        
        # Alternative: for each a in current_sums, check if a + inv_x is in current_sums
        # That's O(n) lookups.
        sum_set = set(current_sums)
        for a in current_sums:
            if a + inv_x in sum_set:
                return False
        return True
    
    # Backtracking search
    all_elements = list(range(1, 23))
    best_set = None
    best_size = 0
    
    # We'll try to build sets of decreasing size.
    # Start with the largest possible size and work down.
    
    # Actually, let's just do a full backtracking that tries to maximize size.
    # We'll sort elements and use a priority on which to include.
    
    # For efficiency, let's try sizes from 22 down to 1.
    # For each size, use backtracking to find any valid subset of that size.
    
    for target_size in range(22, 0, -1):
        if time.time() > deadline:
            break
        if target_size <= best_size:
            break
        
        # Use backtracking to find a subset of size target_size
        found = False
        current = []
        current_sums = [Fraction(0, 1)]  # Start with empty sum
        
        def backtrack(start_idx, remaining):
            nonlocal found, current, current_sums, best_set, best_size
            if time.time() > deadline:
                return True  # Signal to stop
            if found:
                return True
            
            if remaining == 0:
                # We have a valid set of size target_size
                best_set = list(current)
                best_size = target_size
                found = True
                return True
            
            # We need to pick 'remaining' elements from all_elements[start_idx:]
            # Pruning: if not enough elements left, return
            if len(all_elements) - start_idx < remaining:
                return False
            
            # Try each possible next element
            for i in range(start_idx, len(all_elements) - remaining + 1):
                if time.time() > deadline:
                    return True
                x = all_elements[i]
                if can_add(current_sums, x):
                    # Add x
                    new_sums = list(current_sums) + [s + Fraction(1, x) for s in current_sums]
                    current.append(x)
                    old_sums = current_sums
                    current_sums = new_sums
                    
                    if backtrack(i + 1, remaining - 1):
                        return True
                    
                    # Undo
                    current.pop()
                    current_sums = old_sums
            
            return False
        
        backtrack(0, target_size)
        if found:
            break
    
    if best_set is None:
        # Fallback: return a known valid set
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    return best_set
# EVOLVE-BLOCK-END
