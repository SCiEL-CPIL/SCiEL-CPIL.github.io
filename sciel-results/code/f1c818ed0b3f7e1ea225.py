"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def is_valid(S):
    """
    Check if no element in S is the mean of a subset of at least 2 other elements.
    Uses bitset DP for subset sums of differences.
    """
    n = len(S)
    for i, a in enumerate(S):
        # Collect differences t - a for all t in S, t != a
        diffs = []
        for j, t in enumerate(S):
            if i != j:
                diffs.append(t - a)
        
        # B0: bitset for sums reachable by empty set (always just 0)
        # B1: bitset for sums reachable by non-empty subsets
        # We shift by 25000 to handle negative indices if we used array, 
        # but with Python ints, we can just shift and mask if needed.
        # However, Python ints are arbitrary precision, so shifting left/right is safe.
        # We just need to check if bit 0 is set in B1.
        
        B0 = 1  # Bit 0 set
        B1 = 0  # No non-empty subsets yet
        
        for v in diffs:
            if v > 0:
                # Shift left by v
                B1 = B1 | (B1 << v) | (B0 << v)
            elif v < 0:
                # Shift right by -v
                B1 = B1 | (B1 >> -v) | (B0 >> -v)
            else:
                # v == 0, shouldn't happen as t != a
                pass
        
        # Check if sum 0 is reachable by a non-empty subset
        if B1 & 1:
            return False
            
    return True

def greedy(order):
    S = []
    for x in order:
        candidate = S + [x]
        if is_valid(candidate):
            S = candidate
    return S

def construct():
    start_time = time.time()
    deadline = 55.0 # Safety margin
    
    best_set = []
    
    # 1. Ascending
    S_asc = greedy(range(1, 513))
    if len(S_asc) > len(best_set):
        best_set = S_asc
        
    # 2. Descending
    if time.time() - start_time < deadline:
        S_desc = greedy(range(512, 0, -1))
        if len(S_desc) > len(best_set):
            best_set = S_desc
            
    # 3. Random searches
    # We can run multiple random permutations
    # Estimate time for one greedy run:
    # ~512 candidates * 48 checks * 48 diffs * shift cost
    # Should be < 1s.
    
    while time.time() - start_time < deadline:
        random_order = list(range(1, 513))
        random.shuffle(random_order)
        S_rand = greedy(random_order)
        if len(S_rand) > len(best_set):
            best_set = S_rand
            # If we found 48, we can stop or continue to be sure
            if len(best_set) >= 48:
                break
                
    return best_set
# EVOLVE-BLOCK-END
