"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def _can_add(x, S):
    """
    Check if x can be added to S.
    S is a list of selected integers.
    Condition:
    1. x does not divide the sum of any 3 distinct elements in S.
    2. No element y in S divides the sum of x and any 2 distinct elements in S \ {y}.
    """
    n = len(S)
    
    # Check 1: x divides sum of any 3 distinct elements in S
    # Iterate over all combinations of 3 from S
    # If n < 3, this loop doesn't run, which is correct (no sum of 3 exists)
    if n >= 3:
        for a, b, c in combinations(S, 3):
            if (a + b + c) % x == 0:
                return False

    # Check 2: For each y in S, y does not divide (x + a + b) for any distinct a,b in S \ {y}
    # We iterate y. Then we need pairs from S \ {y}.
    # To avoid creating new lists, we can iterate indices or use combinations on the list with y removed.
    # Efficient way:
    for i, y in enumerate(S):
        # We need pairs (a, b) from S such that a != y and b != y and a != b.
        # Sum = x + a + b.
        # Check (x + a + b) % y != 0.
        
        # If n-1 < 2, i.e., n < 3, we can't form such a sum.
        # But if n >= 3, we can.
        if n < 3:
            continue
            
        # Create a list of elements excluding y?
        # Or just iterate indices.
        # S is small, so filtering is okay.
        others = [v for j, v in enumerate(S) if j != i]
        
        if len(others) < 2:
            continue
            
        for a, b in combinations(others, 2):
            if (x + a + b) % y == 0:
                return False

    return True

def _greedy_random(candidates, max_size=48):
    """
    Greedy construction with random order.
    """
    random.shuffle(candidates)
    selected = []
    for x in candidates:
        if len(selected) >= max_size:
            break
        if _can_add(x, selected):
            selected.append(x)
    return selected

def _greedy_descending(candidates, max_size=48):
    """
    Greedy construction with descending order (Seed behavior).
    """
    selected = []
    for x in sorted(candidates, reverse=True):
        if len(selected) >= max_size:
            break
        if _can_add(x, selected):
            selected.append(x)
    return selected

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0  # 50 seconds to be safe
    
    candidates = list(range(1, 193))
    best_set = []
    
    # 1. Try descending order (Seed strategy)
    current_set = _greedy_descending(candidates)
    if len(current_set) > len(best_set):
        best_set = current_set
    
    # 2. Try random restarts
    # Estimate time per run.
    # If one run takes ~0.1s, we can do ~500 runs.
    # If ~1s, we can do ~50 runs.
    # We will adaptively determine how many to run based on elapsed time.
    
    attempts = 0
    while time.time() < deadline:
        current_set = _greedy_random(candidates)
        if len(current_set) > len(best_set):
            best_set = current_set
            # If we hit the max possible size (48), we can stop early?
            # Or maybe we can try to find 49 if allowed? 
            # The prompt says "maximum=48". I will assume 48 is the hard cap.
            if len(best_set) == 48:
                # It's rare to find 48, so we might want to keep searching to ensure it's valid?
                # _can_add ensures validity.
                # We can break if we are confident 48 is the max.
                # But let's run a few more to see if we can get lucky with a different 48-set or if 49 is possible (if cap is soft).
                # Given "maximum=48", I'll stick to 48.
                # We can break to save time, or continue to verify robustness.
                # Let's continue for a bit to see if we can find a different valid 48-set, 
                # though size is the only metric.
                pass
        
        attempts += 1
        # If we haven't found 48 after many tries, maybe it's hard.
        # But we just return the best found.
        
    return best_set

# EVOLVE-BLOCK-END
