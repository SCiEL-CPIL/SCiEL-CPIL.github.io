"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n = 512
    
    # Step 1: Base-3 digit-based construction (Cantor-like set)
    # Select numbers in [1, n] whose base-3 representation contains no digit '2'
    # This set is 3-AP-free, and therefore 4-AP-free.
    S = set()
    for x in range(1, n + 1):
        num = x
        valid = True
        while num > 0:
            if num % 3 == 2:
                valid = False
                break
            num //= 3
        if valid:
            S.add(x)
            
    # Helper function to check if adding x creates a 4-AP
    # Returns True if x can be added safely
    def can_add(x, S):
        # Case 1: x, x+d, x+2d, x+3d
        max_d1 = (n - x) // 3
        for d in range(1, max_d1 + 1):
            if (x + d in S) and (x + 2*d in S) and (x + 3*d in S):
                return False

        # Case 2: x-d, x, x+d, x+2d
        max_d2 = min(x - 1, (n - x) // 2)
        for d in range(1, max_d2 + 1):
            if (x - d in S) and (x + d in S) and (x + 2*d in S):
                return False

        # Case 3: x-2d, x-d, x, x+d
        max_d3 = min((x - 1) // 2, n - x)
        for d in range(1, max_d3 + 1):
            if (x - 2*d in S) and (x - d in S) and (x + d in S):
                return False

        # Case 4: x-3d, x-2d, x-d, x
        max_d4 = (x - 1) // 3
        for d in range(1, max_d4 + 1):
            if (x - 3*d in S) and (x - 2*d in S) and (x - d in S):
                return False

        return True

    # Local Search
    improved = True
    while improved:
        improved = False
        
        # Phase 1: Try to add any missing element
        # We iterate through all x, and if it can be added, we add it.
        # This is a greedy fill.
        for x in range(1, n + 1):
            if x not in S:
                if can_add(x, S):
                    S.add(x)
                    improved = True
        
        # Phase 2: Try swaps (remove one, add one)
        # We iterate through elements y in S.
        # Remove y, then try to add any x not in S.
        # If we can add x, we keep the change and restart the search.
        
        # To avoid modifying S while iterating, we make a list of candidates for removal
        candidates = list(S)
        
        for y in candidates:
            S.remove(y)
            
            # Try to add any missing element
            added = False
            for x in range(1, n + 1):
                if x not in S:
                    if can_add(x, S):
                        S.add(x)
                        added = True
                        improved = True
                        break # Break inner loop, restart outer loop via 'improved' flag
            
            if not added:
                # Revert
                S.add(y)
                
            # If we added something, we break out of the 'for y' loop 
            # because S has changed and 'candidates' is stale.
            # The 'while improved' loop will restart the whole process.
            if added:
                break

    result = sorted(list(S))
    return result
# EVOLVE-BLOCK-END
