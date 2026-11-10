"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations
from math import gcd
from functools import reduce

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    start_time = time.time()
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    def check_dissociated_integer(elements):
        """Check using integer arithmetic via LCM. Returns True if dissociated."""
        k = len(elements)
        if k == 0:
            return True
        try:
            L = reduce(lcm, elements)
            if L > 10**18:  # Too large for efficient integer arithmetic
                return None  # Signal to use Fraction instead
            int_vals = [L // x for x in elements]
            sums = {0}
            for v in int_vals:
                new_sums = set()
                for s in sums:
                    ns = s + v
                    if ns in sums:
                        return False
                    new_sums.add(ns)
                sums.update(new_sums)
            return len(sums) == (1 << k)
        except Exception:
            return None
    
    def check_dissociated_fraction(elements):
        """Check using Fraction arithmetic."""
        k = len(elements)
        if k == 0:
            return True
        reciprocals = [Fraction(1, x) for x in elements]
        sums = {Fraction(0, 1)}
        for r in reciprocals:
            new_sums = set()
            for s in sums:
                ns = s + r
                if ns in sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
        return len(sums) == (1 << k)
    
    def check_dissociated(elements):
        """Unified check: try integer first, fall back to Fraction."""
        result = check_dissociated_integer(elements)
        if result is not None:
            return result
        return check_dissociated_fraction(elements)
    
    def incremental_check_add(current_sums, current_seen, element):
        """Check if adding 1/element maintains dissociation. Returns (new_sums, new_seen) or None."""
        r = Fraction(1, element)
        new_sums = []
        for s in current_sums:
            ns = s + r
            if ns in current_seen:
                return None
            new_sums.append(ns)
        new_seen = current_seen | set(new_sums)
        return (new_sums, new_seen)
    
    all_elements = list(range(1, 23))
    best_set = []
    best_size = 0
    
    # Diverse ordering strategies
    orderings = [
        sorted(all_elements, reverse=True),  # Largest denominators first
        sorted(all_elements),                 # Smallest denominators first
        # Interleaved large-small
        [22, 1, 21, 2, 20, 3, 19, 4, 18, 5, 17, 6, 16, 7, 15, 8, 14, 9, 13, 10, 12, 11],
        # Primes first
        [2, 3, 5, 7, 11, 13, 17, 19, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22],
        # Composites first
        [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 2, 3, 5, 7, 11, 13, 17, 19],
        # Alternating parity starting even
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        # Reverse alternating
        [22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        # Pairs summing to 23
        [1, 22, 2, 21, 3, 20, 4, 19, 5, 18, 6, 17, 7, 16, 8, 15, 9, 14, 10, 13, 11, 12],
    ]
    
    for order in orderings:
        if time.time() > deadline:
            break
        
        current = []
        current_sums = [Fraction(0, 1)]
        current_seen = {Fraction(0, 1)}
        
        for elem in order:
            if time.time() > deadline:
                break
            if elem in current:
                continue
            result = incremental_check_add(current_sums, current_seen, elem)
            if result is not None:
                current_sums, current_seen = result
                current.append(elem)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
    
    # Local improvement: try swaps and additions
    if best_set and time.time() < deadline - 10:
        improved = True
        max_rounds = 5
        for _ in range(max_rounds):
            if time.time() > deadline - 10:
                break
            progress = False
            
            # Try adding any missing element
            for elem in sorted(all_elements, reverse=True):
                if time.time() > deadline - 10:
                    break
                if elem in best_set:
                    continue
                test_set = best_set + [elem]
                if check_dissociated(test_set):
                    best_set = test_set
                    best_size = len(best_set)
                    progress = True
                    break
            
            # Try swapping elements
            if not progress:
                for i, elem in enumerate(best_set):
                    if time.time() > deadline - 10:
                        break
                    for new_elem in all_elements:
                        if new_elem in best_set:
                            continue
                        candidate_set = best_set[:i] + [new_elem] + best_set[i+1:]
                        candidate_set.sort()
                        if check_dissociated(candidate_set):
                            # After swap, try to add more
                            best_set = candidate_set
                            # Try adding remaining elements
                            for add_elem in all_elements:
                                if time.time() > deadline - 10:
                                    break
                                if add_elem in best_set:
                                    continue
                                test_add = best_set + [add_elem]
                                if check_dissociated(test_add):
                                    best_set = test_add
                            best_size = len(best_set)
                            progress = True
                            break
                    if progress:
                        break
            
            if not progress:
                break
    
    if not best_set:
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    return best_set
# EVOLVE-BLOCK-END
