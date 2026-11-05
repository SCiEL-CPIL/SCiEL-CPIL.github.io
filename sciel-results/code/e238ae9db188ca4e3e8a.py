"""Rewritten program using meet-in-the-middle with integer arithmetic for reciprocal-subsum-free set search."""
from math import gcd
from functools import reduce
from itertools import combinations
from math import lcm

def construct():
    """
    Return a nonempty subset of [1, 24] that is reciprocal-subsum-free:
    no reciprocal of a selected element equals the sum of reciprocals of a subset of the OTHER selected elements.
    Maximize cardinality.
    """
    
    def is_valid(S):
        """Check if set S is reciprocal-subsum-free using integer arithmetic with LCM scaling."""
        S_list = list(S)
        n = len(S_list)
        
        for i, x in enumerate(S_list):
            # Check if 1/x can be formed by sum of reciprocals of a subset of S\{x}
            others = [y for j, y in enumerate(S_list) if j != i]
            m = len(others)
            if m == 0:
                continue
            
            # Compute LCM of all elements in others
            # 1/x = sum(1/y for y in T)  <=>  LCM/others / x = sum(LCM/others / y for y in T)
            # Let L = LCM(others). Then L/x = sum(L/y for y in T)
            # L/x might not be integer, so multiply by x: L = x * sum(L/y for y in T)
            # Actually: 1/x = sum(1/y for y in T)
            # Multiply both sides by L*x: L = x * sum(L/y for y in T)
            # So we need: sum(L/y for y in T) = L/x
            # L/x must be integer for this to work with integers.
            
            # Better approach: use LCM of {x} union others
            L = x
            for y in others:
                L = lcm(L, y)
            
            # Now: 1/x = sum(1/y for y in T)
            # Multiply by L: L/x = sum(L/y for y in T)
            # L/x is an integer since x divides L.
            target = L // x
            values = [L // y for y in others]
            
            # Meet-in-the-middle subset sum
            half = m // 2
            left_vals = values[:half]
            right_vals = values[half:]
            
            # Compute all subset sums for left half
            left_sums = set()
            left_sums.add(0)
            for v in left_vals:
                new_sums = set()
                for s in left_sums:
                    new_sums.add(s + v)
                left_sums |= new_sums
            
            # For each subset sum of right half, check if target - that sum is in left_sums
            right_sums = [0]
            for v in right_vals:
                right_sums.extend(s + v for s in right_sums)
            
            for rs in right_sums:
                if (target - rs) in left_sums:
                    return False
        
        return True
    
    # Try to find a valid set of size 16, 15, 14, 13
    # Start with the largest elements as they're most likely to work
    
    # Try sizes from 16 down to 13
    for size in [16, 15, 14, 13]:
        # Generate candidate sets: prefer larger elements
        # Use combinations with a heuristic ordering
        # Try the set of the 'size' largest elements first
        top = list(range(25 - size, 25))
        if is_valid(top):
            return top
        
        # Try variations by replacing elements with smaller ones
        # Use a systematic approach: try all combinations of size 'size' from [1..24]
        # but prioritize those with larger elements
        # For efficiency, use a greedy/backtracking approach
        
        import random
        random.seed(42)
        
        # Try many random variations
        found = False
        for attempt in range(2000):
            # Start with top elements, then randomly replace some
            cand = list(range(25 - size, 25))
            num_swaps = random.randint(1, min(6, size // 2))
            for _ in range(num_swaps):
                idx = random.randint(0, len(cand) - 1)
                old_val = cand[idx]
                if old_val > 1:
                    new_val = random.randint(1, old_val - 1)
                    if new_val not in cand:
                        cand[idx] = new_val
            cand = sorted(set(cand))
            if len(cand) != size:
                continue
            if is_valid(cand):
                return cand
        
        # Also try systematic search: replace one element at a time
        for pos in range(size):
            for new_val in range(1, 25 - size + pos):
                if new_val in top:
                    continue
                cand = top.copy()
                cand[pos] = new_val
                cand = sorted(set(cand))
                if len(cand) == size and is_valid(cand):
                    return cand
    
    # Fallback: return the seed answer
    return list(range(13, 25))
