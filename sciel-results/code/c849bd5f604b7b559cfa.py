"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Start with the valid set of all numbers > 128
    # This is a valid clique because lcm(a,b) > 256 for distinct a,b in (128, 256]
    S = set(range(129, 257))
    
    # We will try to add numbers from 2 to 256 that are not in S
    # If adding x requires removing some y in S, we do it if 1/x > sum(1/y)
    
    # Candidates to consider adding
    # We iterate through all possible x in [2, 256]
    # Using a list allows us to re-evaluate candidates in subsequent passes
    candidates = list(range(2, 257))
    
    improved = True
    # Limit passes to avoid infinite loops or excessive time if no clear improvement
    # But 'improved' flag breaks naturally.
    while improved and time.time() < deadline:
        improved = False
        # Sort candidates by reciprocal (descending) to prioritize high-value additions?
        # This is a heuristic. Smaller numbers have higher 1/x.
        # Sorting every pass is O(N log N). N=255. Cheap.
        # However, the order of application within a pass matters.
        # Let's just iterate 2..256.
        
        for x in candidates:
            if time.time() > deadline:
                return list(S)
            
            if x in S:
                continue
            
            # Find all y in S that conflict with x
            # Conflict: lcm(x, y) <= 256
            conflicts = []
            for y in S:
                if lcm(x, y) <= 256:
                    conflicts.append(y)
            
            if not conflicts:
                # x is compatible with all in S
                # Gain is 1/x, Loss is 0
                S.add(x)
                improved = True
            else:
                # Check if the gain of adding x outweighs the loss of removing conflicts
                gain = 1.0 / x
                loss = sum(1.0 / y for y in conflicts)
                
                # Strictly greater to ensure progress and avoid oscillation
                if gain > loss:
                    for y in conflicts:
                        S.remove(y)
                    S.add(x)
                    improved = True
                    
    return list(S)
# EVOLVE-BLOCK-END
