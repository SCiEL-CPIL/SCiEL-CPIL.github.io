"""Find a large dissociated reciprocal subset of [1,22]."""
import time
from fractions import Fraction
from itertools import combinations

# EVOLVE-BLOCK-START
import sys

def _is_dissociated(elements):
    """Check if all subset sums of 1/e for e in elements are distinct."""
    n = len(elements)
    if n == 0:
        return True
    # Use integer arithmetic: multiply by LCM(1..22)
    # LCM(1..22) = 232792560
    LCM = 232792560
    recips = [LCM // e for e in elements]
    # Generate all subset sums
    sums = set()
    sums.add(0)
    for r in recips:
        new_sums = set()
        for s in sums:
            ns = s + r
            if ns in sums:
                return False
            new_sums.add(ns)
        sums.update(new_sums)
        if len(sums) != 2 ** (len(recips[:recips.index(r)+1])):
            return False
    return True

def _is_dissociated_fast(elements):
    """Faster check using incremental subset sum generation."""
    n = len(elements)
    if n == 0:
        return True
    LCM = 232792560
    recips = [LCM // e for e in elements]
    sums = {0}
    for i, r in enumerate(recips):
        new_sums = set()
        for s in sums:
            ns = s + r
            if ns in sums:
                return False
            new_sums.add(ns)
        sums = sums | new_sums
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s margin
    
    # Try to find the largest dissociated set
    # Greedy + backtracking approach
    # Start with empty set, try to add elements in various orders
    
    # Strategy: Try all subsets of size k from largest k down
    # But that's expensive. Instead, use greedy with backtracking.
    
    # First, let's try a simple greedy: add elements in order 1,2,3,...,22
    # and check if the set remains dissociated
    greedy_set = []
    for x in range(1, 23):
        if time.time() > deadline:
            break
        candidate = greedy_set + [x]
        if _is_dissociated_fast(candidate):
            greedy_set = candidate
    
    best = greedy_set
    
    # Now try to improve by swapping
    # Try different starting orders
    import random
    random.seed(42)
    
    # Try multiple random orderings
    for trial in range(200):
        if time.time() > deadline:
            break
        order = list(range(1, 23))
        random.shuffle(order)
        greedy_set = []
        for x in order:
            if time.time() > deadline:
                break
            candidate = greedy_set + [x]
            if _is_dissociated_fast(candidate):
                greedy_set = candidate
        if len(greedy_set) > len(best):
            best = greedy_set
    
    # Try local search: try to add an element not in best
    if time.time() < deadline - 5:
        current = list(best)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for x in range(1, 23):
                if x not in current:
                    candidate = current + [x]
                    if _is_dissociated_fast(candidate):
                        current = candidate
                        improved = True
                        break
            if len(current) > len(best):
                best = current
    
    # Try removing one element and adding two others
    if time.time() < deadline - 10 and len(best) >= 3:
        current = list(best)
        for i in range(len(current)):
            if time.time() > deadline - 10:
                break
            without_i = current[:i] + current[i+1:]
            for x in range(1, 23):
                if x not in without_i:
                    for y in range(x+1, 23):
                        if y not in without_i:
                            candidate = without_i + [x, y]
                            if _is_dissociated_fast(candidate):
                                if len(candidate) > len(best):
                                    best = candidate
                                break
            if time.time() > deadline - 10:
                break
    
    if not best:
        best = [1]
    
    return sorted(best)
# EVOLVE-BLOCK-END
