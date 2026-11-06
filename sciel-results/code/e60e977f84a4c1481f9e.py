"""Program for the suite task; finds a large subset of [1,256] where no a divides b+c."""
import sys
from functools import lru_cache

N = 256

def construct():
    # We want a subset S of [1, N] such that for all a < b <= c in S, a does not divide b+c.
    # Strategy: greedy with backtracking, processing from largest to smallest.
    # For each candidate x, check if adding x to current set S violates the constraint.
    # If x is added, then for all b, c in S with x < b <= c, we need x not to divide b+c.
    # Also, for existing a in S with a < x <= y (y in S), we need a not to divide x+y.
    # And for existing a, b in S with a < b <= x, we need a not to divide b+x.
    
    # Actually, let's think more carefully.
    # If we have a set S and we want to add x:
    # 1. For all pairs (b, c) in S with x < b <= c: x must not divide b+c.
    # 2. For all pairs (a, y) in S with a < x <= y: a must not divide x+y.
    # 3. For all pairs (a, b) in S with a < b <= x: a must not divide b+x.
    # 4. For all pairs (a, x) where a in S and a < x: a must not divide x+x = 2x. (This is case 2 with y=x, but x is not in S yet)
    #    Wait, the condition is a < b <= c. If we add x, then triples involving x are:
    #    - x < b <= c where b, c in S: x | (b+c) is forbidden.
    #    - a < x <= c where a, c in S: a | (x+c) is forbidden.
    #    - a < b <= x where a, b in S: a | (b+x) is forbidden.
    #    - a < x <= x where a in S: a | (x+x) is forbidden. (This is a special case of a < x <= c with c=x)
    
    # Let's implement a greedy approach from N down to 1.
    # At each step, try to add the current number if it doesn't conflict.
    # This is a heuristic, but it often works well for these types of problems.
    
    # Actually, let's try a more systematic approach.
    # The interval [171, 256] has size 86.
    # Can we do better? Let's try to add some numbers below 171.
    # If we add x < 171, we need to check conflicts with all pairs in the current set.
    
    # Let's implement a greedy algorithm that starts with the largest numbers and tries to add smaller ones.
    
    S = set()
    # Start with the largest numbers
    for x in range(N, 0, -1):
        # Check if x can be added to S
        can_add = True
        for b in S:
            for c in S:
                if b < c and x < b:
                    if (b + c) % x == 0:
                        can_add = False
                        break
                if b < c and x == b:
                    # This case is handled below
                    pass
            if not can_add:
                break
        if not can_add:
            continue
        
        # Check all pairs (a, c) in S where a < x <= c
        for a in S:
            for c in S:
                if a < x <= c:
                    if (x + c) % a == 0:
                        can_add = False
                        break
            if not can_add:
                break
        if not can_add:
            continue
        
        # Check all pairs (a, b) in S where a < b <= x
        for a in S:
            for b in S:
                if a < b <= x:
                    if (b + x) % a == 0:
                        can_add = False
                        break
            if not can_add:
                break
        if not can_add:
            continue
        
        # Check a < x <= x, i.e., a | 2x
        for a in S:
            if a < x and (2 * x) % a == 0:
                can_add = False
                break
        if not can_add:
            continue
        
        S.add(x)
    
    return list(S)
