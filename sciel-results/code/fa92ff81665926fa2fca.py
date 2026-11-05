import time
from math import gcd
from fractions import Fraction
import random

def _is_dissociated(S):
    """Check if the set of reciprocals of S has all distinct subset sums."""
    if not S:
        return True
    n = len(S)
    # Compute all 2^n subset sums using Fraction for exactness
    # For n up to ~14, 2^14 = 16384 is fine
    sums = [Fraction(0)]
    for a in S:
        # Add 1/a to each existing sum
        new_sums = [s + Fraction(1, a) for s in sums]
        sums.extend(new_sums)
    
    # Check for duplicates
    if len(set(sums)) != len(sums):
        return False
    return True

def _is_dissociated_fast(S):
    """Faster check using integer arithmetic with a common denominator approach.
    Instead of tracking all fractions, we can use the fact that for small n,
    we can compute subset sums as (numerator, denominator) pairs reduced."""
    if not S:
        return True
    n = len(S)
    if n > 16:
        # Too large for brute force, use a heuristic or just return False
        return False
    
    # Use (num, den) representation
    sums = [(0, 1)]
    for a in S:
        new_sums = []
        for (num, den) in sums:
            new_num = num * a + den
            new_den = den * a
            g = gcd(new_num, new_den)
            new_sums.append((new_num // g, new_den // g))
        sums.extend(new_sums)
    
    if len(set(sums)) != len(sums):
        return False
    return True

def construct():
    deadline = time.time() + 85  # 85s safety margin (90s limit)
    
    universe = list(range(1, 23))
    best = []
    best_size = 0
    
    # First, try to find a large set using greedy with multiple orderings
    # Then use local search to improve
    
    def greedy_add(order, current=None):
        """Try to add elements in the given order, skipping conflicts."""
        if current is None:
            current = []
        for a in order:
            if time.time() > deadline:
                break
            test = current + [a]
            if _is_dissociated_fast(test):
                current = test
        return current
    
    # Try several orderings
    random.seed(42)
    
    orderings = []
    # Natural order
    orderings.append(universe)
    # Reverse
    orderings.append(universe[::-1])
    # Sorted by some heuristic
    orderings.append(sorted(universe, key=lambda x: x))
    # Random shuffles
    for _ in range(30):
        shuffled = universe[:]
        random.shuffle(shuffled)
        orderings.append(shuffled)
    
    for order in orderings:
        if time.time() > deadline:
            break
        result = greedy_add(order)
        if len(result) > best_size:
            best = result
            best_size = len(result)
    
    # Now do local search: try to add one element at a time
    # Also try swapping elements
    
    def can_add(current, a):
        """Check if a can be added to current."""
        if a in current:
            return False
        test = current + [a]
        return _is_dissociated_fast(test)
    
    # Iterative improvement
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding any element not in best
        for a in universe:
            if time.time() > deadline:
                break
            if a not in best and can_add(best, a):
                best = best + [a]
                best_size += 1
                improved = True
        
        # Try swapping: remove one, add another
        if not improved:
            for i in range(len(best)):
                if time.time() > deadline:
                    break
                reduced = best[:i] + best[i+1:]
                for a in universe:
                    if time.time() > deadline:
                        break
                    if a in reduced:
                        continue
                    test = reduced + [a]
                    if _is_dissociated_fast(test):
                        # Check if this swap improves (it doesn't change size, but might allow future additions)
                        # For now, just accept it if it leads to a different set
                        if test != best:
                            best = test
                            improved = True
                            break
                if improved:
                    break
    
    # Try a more systematic branch and bound
    def bb_search(available, current, depth=0):
        nonlocal best, best_size
        if time.time() > deadline:
            return
        if len(current) + len(available) <= best_size:
            return
        if not available:
            return
        
        a = available[0]
        rest = available[1:]
        
        # Branch 1: include a
        if can_add(current, a):
            bb_search(rest, current + [a], depth + 1)
        
        # Branch 2: exclude a
        bb_search(rest, current, depth + 1)
    
    # Run B&B from a few starting points
    # Start from empty
    if time.time() < deadline:
        bb_search(universe[:], [])
    
    # Also try starting from the current best and see if we can extend
    if time.time() < deadline and best_size < len(universe):
        # Try to find elements that can be added to best
        remaining = [x for x in universe if x not in best]
        if remaining:
            # Do a smaller B&B on the remaining elements
            def bb_extend(available, current):
                nonlocal best, best_size
                if time.time() > deadline:
                    return
                if len(current) + len(available) <= best_size:
                    return
                if not available:
                    return
                
                a = available[0]
                rest = available[1:]
                
                # Branch 1: include a
                if can_add(current, a):
                    bb_extend(rest, current + [a])
                
                # Branch 2: exclude a
                bb_extend(rest, current)
            
            bb_extend(remaining, best[:])
    
    # Final check: make sure best is valid
    if best and not _is_dissociated_fast(best):
        # Fallback to a known small valid set
        best = [1]
    
    return best
