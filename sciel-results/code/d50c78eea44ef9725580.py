import time
import random
import sys

def _has_3ap(S):
    """
    Returns True if S contains a 3-term arithmetic progression.
    i.e., there exist distinct a, b, c in S such that a + c = 2b.
    This corresponds to b being the mean of a and c.
    """
    s = set(S)
    # To optimize, we can iterate pairs (a, c) and check if (a+c)/2 is in S.
    # Or iterate b and check if there are a, c in S such that a+c=2b.
    # The latter is harder to do efficiently without DP or sorting.
    # Simple O(N^2) check:
    n = len(S)
    for i in range(n):
        for j in range(i + 1, n):
            a = S[i]
            b = S[j]
            # We need c such that a + c = 2b => c = 2b - a
            c = 2 * b - a
            if c in s and c != a and c != b:
                return True
    return False

def _is_valid_set(S):
    """
    Checks if no element in S is the mean of a subset of at least 2 other elements.
    1. Quick check for k=2 (3-term AP).
    2. Full DP check for k >= 3.
    """
    n = len(S)
    if n < 3:
        return True
        
    # Quick check for k=2
    if _has_3ap(S):
        return False
        
    # Full DP check for k >= 3
    # We can reuse the DP logic but start k from 3?
    # The DP calculates all k. We just check k >= 3.
    
    for i, a in enumerate(S):
        others = [x for j, x in enumerate(S) if j != i]
        
        dp = [0] * (len(others) + 1)
        dp[0] = 1
        
        for x in others:
            for j in range(len(others) - 1, -1, -1):
                if dp[j]:
                    dp[j+1] |= (dp[j] << x)
                    
        for k in range(3, len(others) + 1): # Start from 3 since k=2 is handled
            target_sum = k * a
            if (dp[k] >> target_sum) & 1:
                return False
                
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s margin
    
    best_set = [1]  # Always start with a valid set of size 1
    
    rng = random.Random(42)
    
    # Strategy: Greedy Construction with Restarts
    for restart in range(500):
        if time.time() > deadline:
            break
            
        current_S = [1]
        remaining = list(range(2, 513))
        rng.shuffle(remaining)
        
        for x in remaining:
            if time.time() > deadline:
                break
                
            new_S = current_S + [x]
            
            # Fast pre-check: 3-AP
            if _has_3ap(new_S):
                continue
                
            # Full check
            if _is_valid_set(new_S):
                current_S = new_S
                
            if len(current_S) >= 48:
                break
                
        if len(current_S) > len(best_set):
            best_set = current_S
            if len(best_set) == 48:
                break
                
    return best_set

if __name__ == "__main__":
    s = construct()
    print(f"Found set of size {len(s)}")
    print(s)
    if len(s) > 0:
        print(f"Valid: {_is_valid_set(s)}")
