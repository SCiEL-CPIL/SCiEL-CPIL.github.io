"""
Strategy: Local Search (Hill Climbing) with Greedy Initialization.
1. Initialize with a valid set using greedy removal from [1, 24].
2. Iteratively attempt to add removed elements back.
3. If stuck, perform swaps (remove one, add one) to escape local optima.
4. Use exact Fraction arithmetic to ensure validity.
"""

import time
from fractions import Fraction
from itertools import combinations

_START_TIME = None
DEADLINE = 75.0  # Safety margin against 90s limit

def _deadline_hit():
    global _START_TIME
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > DEADLINE

def _is_valid_set(S):
    """
    Check if set S is reciprocal-subsum-free.
    For each a in S, 1/a must NOT equal sum of reciprocals of any subset of S \ {a}.
    """
    S_set = set(S)
    for a in S_set:
        target = Fraction(1, a)
        others = [x for x in S_set if x != a]
        
        # Check all subsets of others? 
        # |others| can be up to 23. 2^23 is ~8M. Too slow for many checks.
        # However, we can use a meet-in-the-middle or DFS with pruning.
        # Given the small range [1, 24], and that we are adding elements back,
        # we can optimize the check.
        
        # A faster check: 
        # The sum of reciprocals of a subset of integers is a rational number.
        # We can generate all possible sums of reciprocals of subsets of 'others' 
        # and check if target is among them.
        
        n = len(others)
        if n == 0:
            continue
            
        # Meet in the middle for subset sum of reciprocals
        # Split others into two halves
        half = n // 2
        left = others[:half]
        right = others[half:]
        
        # Generate all subset sums for left
        left_sums = set()
        for i in range(1 << len(left)):
            s = Fraction(0, 1)
            for j in range(len(left)):
                if (i >> j) & 1:
                    s += Fraction(1, left[j])
            left_sums.add(s)
            
        # Generate all subset sums for right and check for complement
        for i in range(1 << len(right)):
            s = Fraction(0, 1)
            for j in range(len(right)):
                if (i >> j) & 1:
                    s += Fraction(1, right[j])
            
            if (target - s) in left_sums:
                return False
                
    return True

def _greedy_init():
    """
    Greedy removal from full set [1, 24] to get a valid starting set.
    """
    S = list(range(1, 25))
    # Remove elements in order of increasing 'criticality' or just iterate?
    # A common heuristic: remove elements that cause the most conflicts.
    # Or simply: iterate 1..24, if adding it creates a violation, remove it.
    # Actually, the observed best program did "greedy removal from full set".
    # Let's replicate that: Start with full set, remove one element at a time 
    # that resolves the most violations, or simply remove the smallest element 
    # involved in a violation.
    
    # Simpler: Start with empty, add 1..24? No, that's building up.
    # The note says "Greedy removal from full set [1,24] ... finds larger valid sets".
    # Let's try: Start with [1..24]. While invalid, find a violation and remove one of the elements involved.
    # To maximize size, we should remove the element whose removal resolves the most conflicts.
    
    S_set = set(range(1, 25))
    
    while True:
        if _is_valid_set(S_set):
            break
            
        # Find a violation
        # We need to find a specific a and subset T such that 1/a = sum(1/t for t in T).
        # This is expensive to find from scratch each time.
        # Alternative: Just try removing elements 1 by 1?
        
        # Let's try a simpler heuristic: 
        # Iterate through elements 1..24. If removing element x makes the set valid?
        # No, that's O(N * ValidityCheck).
        
        # Let's stick to the "remove minimal conflicting elements" idea.
        # Identify all elements involved in violations.
        # For each a in S_set:
        #   Check if 1/a is a subset sum of reciprocals of S_set \ {a}.
        #   If yes, mark a as 'bad'.
        #   Also, mark elements in the subset T as 'bad' (they contribute to the sum).
        # Remove the 'bad' element with the smallest value? Or the one that appears in most violations?
        
        bad_elements = set()
        
        for a in S_set:
            target = Fraction(1, a)
            others = [x for x in S_set if x != a]
            n = len(others)
            if n == 0: continue
            
            half = n // 2
            left = others[:half]
            right = others[half:]
            
            left_sums = {} # map sum to subset (list of indices)
            for i in range(1 << len(left)):
                s = Fraction(0, 1)
                subset = []
                for j in range(len(left)):
                    if (i >> j) & 1:
                        s += Fraction(1, left[j])
                        subset.append(left[j])
                # Keep the first subset found for a sum, or just track existence?
                # We need to know WHICH elements form the sum to mark them bad.
                if s not in left_sums:
                    left_sums[s] = subset
            
            for i in range(1 << len(right)):
                s = Fraction(0, 1)
                subset = []
                for j in range(len(right)):
                    if (i >> j) & 1:
                        s += Fraction(1, right[j])
                        subset.append(right[j])
                
                complement = target - s
                if complement in left_sums:
                    # Violation found for a
                    bad_elements.add(a)
                    bad_elements.update(subset)
                    bad_elements.update(left_sums[complement])
                    break # One violation per a is enough to mark it bad
        
        if not bad_elements:
            # Should not happen if _is_valid_set returned False
            break
            
        # Remove the smallest bad element? Or the one that is in the most violations?
        # Removing smallest tends to keep larger numbers which have smaller reciprocals, 
        # potentially reducing the chance of sums matching 1/a for large a?
        # Actually, 1/large_a is small. Sums of reciprocals of large numbers are small.
        # 1/small_a is large. Sums of reciprocals of small numbers are large.
        # Conflicts often involve small numbers.
        
        # Heuristic: Remove the smallest element in bad_elements.
        min_bad = min(bad_elements)
        S_set.remove(min_bad)
        
        if _deadline_hit():
            break
            
    return list(S_set)

def _try_add(S_set, candidate):
    """Try to add candidate to S_set. Return True if valid."""
    new_set = S_set | {candidate}
    if _is_valid_set(new_set):
        return True
    return False

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # 1. Initialize
    S = _greedy_init()
    S_set = set(S)
    removed = set(range(1, 25)) - S_set
    
    best_set = set(S_set)
    best_size = len(best_set)
    
    # 2. Local Search: Try to add back removed elements
    # Sort removed elements. Maybe try adding larger ones first? 
    # Or smaller? 
    # Let's try all permutations of additions? No, just greedy addition.
    # Iterate through removed elements, try to add. If successful, add it.
    # This is a greedy construction on top of the greedy removal.
    
    # To escape local optima, we can do multiple passes or swaps.
    
    # Strategy A: Greedy Add
    for cand in sorted(removed):
        if _deadline_hit():
            break
        if _try_add(S_set, cand):
            S_set.add(cand)
            removed.remove(cand)
    
    if len(S_set) > best_size:
        best_set = set(S_set)
        best_size = len(best_set)
        
    # Strategy B: Swap Search
    # If we are stuck, try swapping an element in S_set with one in removed.
    # This allows us to "move" elements.
    
    # We can do this iteratively.
    improved = True
    while improved and not _deadline_hit():
        improved = False
        # Try all pairs (x in S_set, y in removed)
        # To save time, we can limit the number of swaps tried per iteration
        # or just iterate through all.
        # |S_set| <= 24, |removed| <= 24. Max 576 pairs.
        # Each swap check involves _is_valid_set which is O(2^(N/2)).
        # N=24 -> 2^12 = 4096 per half. 4096 * 4096 checks? No, MITM is fast.
        
        for x in list(S_set):
            if _deadline_hit(): break
            for y in list(removed):
                if _deadline_hit(): break
                # Try swap x -> y
                new_set = S_set.copy()
                new_set.remove(x)
                new_set.add(y)
                
                if _is_valid_set(new_set):
                    # Accept swap
                    S_set = new_set
                    removed.remove(y)
                    removed.add(x)
                    improved = True
                    if len(S_set) > best_size:
                        best_set = set(S_set)
                        best_size = len(best_set)
                    break # Restart inner loop? Or continue? 
                    # Continuing might find more swaps. But modifying S_set during iteration is tricky.
                    # Let's break and restart the while loop to re-evaluate.
                    
        # If no swap was made in this pass, we might be stuck in a local optimum 
        # where no single swap improves validity (but maybe a double swap would?).
        # For now, single swap hill climbing is the structural change.
        
    # Final check
    if not _is_valid_set(best_set):
        # Fallback to a known valid set if something went wrong
        # Though _is_valid_set should guarantee this
        pass
        
    return list(best_set)

if __name__ == "__main__":
    print(construct())
