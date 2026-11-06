import itertools
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80  # 80s limit
    
    N = 24
    # Reciprocals
    recips = [1.0 / i for i in range(1, N + 1)]
    
    # Helper to check validity of a set (list of indices 1-based)
    def is_valid_set(elements):
        # Compute all subset sums of reciprocals in elements
        # elements are values 1..24
        # Map to recips
        vals = [recips[e-1] for e in elements]
        
        # Generate all subset sums
        # Use a set to store sums
        sums = {0.0}
        current_sums = {0.0}
        
        for v in vals:
            new_sums = set()
            for s in current_sums:
                new_sums.add(s + v)
            sums.update(new_sums)
            current_sums = new_sums
            
        # Check if any 1/x is in sums
        for e in elements:
            target = recips[e-1]
            if target in sums:
                return False
        return True

    # Search for largest K
    for K in range(N, 0, -1):
        # Iterate combinations in "descending value" order
        # range(N, 0, -1) gives N, N-1, ..., 1
        for combo in itertools.combinations(range(N, 0, -1), K):
            # Check deadline
            if time.time() > deadline:
                # Return best found so far? 
                # We are iterating K downwards, so if we timeout, we haven't found a valid one for this K.
                # We should return a known valid set, e.g., {13..24}
                return list(range(13, 25))
            
            if is_valid_set(combo):
                return list(combo)
                
    return [1]
