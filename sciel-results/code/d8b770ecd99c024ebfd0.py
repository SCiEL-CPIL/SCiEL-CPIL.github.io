import sys
import time
import numpy as np
from itertools import combinations_with_replacement

def construct():
    n = 4096
    max_sum = 3 * n
    
    # Precompute all pairs (i, j) with i <= j and their sum
    # This is useful for generating new sums when adding a new element x
    # Actually, let's just generate them on the fly but efficiently.
    
    # We will use a bytearray for used_sums for O(1) lookup and update
    used_sums = bytearray(max_sum + 1)
    
    # The set of chosen elements
    chosen = []
    
    # Global best
    best_set = []
    best_size = 0
    
    # Time limit
    start_time = time.time()
    timeout = 55.0
    
    # We will implement a backtracking search.
    # To make it fast, we need to be smart about which candidates to try.
    # A key insight for B_h sequences: the greedy algorithm (picking the smallest possible next element)
    # is often optimal or near-optimal for maximizing size in [1, N].
    # However, sometimes a slightly larger choice allows more future elements.
    # Backtracking allows us to explore alternatives if the greedy path leads to a dead end earlier.
    
    # But full backtracking over 4096 candidates at each step is too slow.
    # We need a strong heuristic.
    
    # Heuristic: Try candidates in increasing order, but limit the branching factor.
    # Or, use a "lookahead" heuristic: pick the candidate that minimizes the number of future conflicts.
    # This is expensive.
    
    # Alternative: Use a randomized restart approach with a greedy-like backtracking.
    
    # Let's try a depth-first search with a limited width.
    # At each depth, we only try the first K valid candidates (e.g., K=5 or 10).
    # This keeps the search tree manageable.
    
    # Also, we can use a "jump" heuristic: if we are stuck, try a larger number.
    
    # Let's implement a recursive function.
    # State: current list of chosen elements, current used_sums bytearray.
    # We need to efficiently check if a new element x can be added.
    
    def can_add(x, chosen_list):
        """
        Check if x can be added to chosen_list without creating sum collisions.
        Returns a list of new sums if valid, else None.
        """
        new_sums = []
        
        # Triples involving x:
        # 1. (x, x, x)
        s = 3 * x
        if used_sums[s]:
            return None
        new_sums.append(s)
        
        # 2. (a, x, x) for a in chosen_list
        for a in chosen_list:
            s = a + 2 * x
            if used_sums[s]:
                return None
            new_sums.append(s)
        
        # 3. (a, b, x) for a <= b in chosen_list
        for i in range(len(chosen_list)):
            a = chosen_list[i]
            for j in range(i, len(chosen_list)):
                b = chosen_list[j]
                s = a + b + x
                if used_sums[s]:
                    return None
                new_sums.append(s)
        
        # Check for internal collisions among new_sums
        # Since we are adding x, the new sums are distinct if the triples are distinct.
        # But we must ensure no two new sums are equal.
        # Actually, the triples are distinct because (a,b,c) is unique.
        # But could two different triples involving x have the same sum?
        # e.g., (x,x,x) and (a,b,c)? No, because all elements in the triple are from chosen_list + {x}.
        # The only way two triples involving x have the same sum is if they are the same triple.
        # Wait, (a, x, x) and (b, c, x) could have the same sum?
        # a + 2x = b + c + x => a + x = b + c.
        # Since a, b, c are in chosen_list and x > all of them (if we enforce increasing order),
        # then a + x > b + c is not necessarily true.
        # But if we enforce x > max(chosen_list), then x is the largest element.
        # Then in (a, x, x), the sum is a + 2x.
        # In (b, c, x), the sum is b + c + x.
        # a + 2x = b + c + x => a + x = b + c.
        # Since x > b, c, then a + x > b + c is false? No.
        # Example: a=1, x=10, b=2, c=9. 1+10=11, 2+9=11. Sum=21.
        # So yes, collisions can happen between new sums.
        # So we must check if any two new sums are equal.
        
        # However, if we just check against used_sums, we might miss internal collisions.
        # So we should use a set for new_sums to check for duplicates.
        new_sums_set = set()
        for s in new_sums:
            if s in new_sums_set:
                return None
            new_sums_set.add(s)
        
        return new_sums
    
    def backtrack(start_val, depth):
        nonlocal best_set, best_size, start_time
        
        if time.time() - start_time > timeout:
            return
        
        # Update best
        if depth > best_size:
            best_size = depth
            best_set = chosen[:]
        
        # Pure Greedy Strategy: Try only the first valid candidate.
        # This ensures we finish quickly and get a good baseline.
        # We can try a few alternatives if we have time, but let's start with 1.
        
        tried = 0
        max_tries = 1
        
        for x in range(start_val, n + 1):
            if time.time() - start_time > timeout:
                return
            
            if tried >= max_tries:
                break
            
            new_sums = can_add(x, chosen)
            if new_sums is not None:
                # Add x
                chosen.append(x)
                for s in new_sums:
                    used_sums[s] = 1
                
                # Recurse
                backtrack(x + 1, depth + 1)
                
                # Undo
                chosen.pop()
                for s in new_sums:
                    used_sums[s] = 0
                
                tried += 1
        
    # Initialize
    backtrack(1, 0)
    
    return best_set
