import time
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 45.0 # Leave 15s buffer

    # Precompute APs for fast checking
    # is_ap[a][b][c] is too big. 
    # Instead, check on the fly or precompute a set of forbidden triples.
    # For N=256, number of APs is ~ N^2/2 = 32k.
    # We can store APs as a set of tuples (a, b, c) with 0-indexing.
    
    # Better: For local search, we need to check if adding x creates an AP.
    # This is O(|S|) per check.
    # |S| is at most ~100. N=256.
    # Total checks in local search could be high.
    
    # Let's use a bitset or set for S.
    S = set()
    
    # Greedy Initialization
    # Sort candidates by weight (1/i) descending -> i ascending
    for i in range(1, N + 1):
        if time.time() > deadline:
            break
        valid = True
        # Check if adding i creates a 3-AP with any two existing elements
        # Or with one existing and one implicit? No, AP is 3 distinct terms.
        # If we add i, it must not form (a,b,c) with a,b,c in S ∪ {i}.
        # Since a,b,c must be distinct, and i is new:
        # 1. i is middle: a = i-d, c = i+d. a,c in S.
        # 2. i is end: 
        #    - i is a: b=i+d, c=i+2d. b,c in S.
        #    - i is c: a=i-2d, b=i-d. a,b in S.
        
        # Efficient check:
        # For each j in S:
        #   If i is middle: k = 2*i - j. If k in S, invalid.
        #   If i is end and j is middle: k = 2*j - i. If k in S, invalid.
        #   If i is end and j is end: k = (i+j)//2. If k in S, invalid.
        
        for j in S:
            # i is middle
            k1 = 2 * i - j
            if k1 in S:
                valid = False
                break
            # j is middle
            k2 = 2 * j - i
            if k2 in S:
                valid = False
                break
            # i and j are ends
            if (i + j) % 2 == 0:
                k3 = (i + j) // 2
                if k3 in S:
                    valid = False
                    break
        if valid:
            S.add(i)
            
    current_weight = sum(1.0 / i for i in S)
    
    # Local Search: Hill Climbing
    # We will try to improve the weight.
    # Moves: Add one, Remove one, Swap one.
    # To be efficient, we can prioritize high weight additions/removals.
    
    # Precompute weights
    weights = {i: 1.0 / i for i in range(1, N + 1)}
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try adding any element not in S
        # Sort candidates by weight descending to find good additions early
        candidates = [i for i in range(1, N + 1) if i not in S]
        candidates.sort(key=lambda x: weights[x], reverse=True)
        
        for i in candidates:
            if time.time() > deadline:
                break
            # Check validity of adding i
            valid = True
            for j in S:
                k1 = 2 * i - j
                if k1 in S:
                    valid = False
                    break
                k2 = 2 * j - i
                if k2 in S:
                    valid = False
                    break
                if (i + j) % 2 == 0:
                    k3 = (i + j) // 2
                    if k3 in S:
                        valid = False
                        break
            if valid:
                S.add(i)
                current_weight += weights[i]
                improved = True
                break # Restart loop as S changed
                
        # 2. Try swapping: remove j from S, add i not in S
        # This is O(|S| * (N-|S|)). 100 * 150 = 15k checks.
        # Each check is O(|S|) = 100. Total 1.5M ops. Fast.
        
        # We need to be careful not to do this infinite loop.
        # Only do swap if it increases weight.
        
        # To avoid checking all swaps if an addition was made, we break above.
        # If no addition was made, we try swaps.
        
        if not improved:
            # Try swaps
            # We want to maximize weight.
            # Remove j, add i. Delta = w[i] - w[j].
            # We only care if w[i] > w[j].
            
            # Sort S by weight ascending (low weight elements are good candidates for removal)
            # Sort candidates by weight descending (high weight elements are good candidates for addition)
            
            removals = sorted(list(S), key=lambda x: weights[x])
            additions = sorted([i for i in range(1, N + 1) if i not in S], key=lambda x: weights[x], reverse=True)
            
            found_swap = False
            for j in removals:
                if time.time() > deadline:
                    break
                if found_swap:
                    break
                for i in additions:
                    if weights[i] <= weights[j]:
                        break # No point checking lower weight additions for this j
                    # Check if swapping j for i is valid
                    # Temporarily remove j
                    S.remove(j)
                    
                    # Check if i can be added to S (which no longer has j)
                    valid = True
                    for k in S:
                        k1 = 2 * i - k
                        if k1 in S:
                            valid = False
                            break
                        k2 = 2 * k - i
                        if k2 in S:
                            valid = False
                            break
                        if (i + k) % 2 == 0:
                            k3 = (i + k) // 2
                            if k3 in S:
                                valid = False
                                break
                    
                    if valid:
                        S.add(i)
                        current_weight += weights[i] - weights[j]
                        improved = True
                        found_swap = True
                        break
                    else:
                        # Restore j
                        S.add(j)
                        
    return list(S)
