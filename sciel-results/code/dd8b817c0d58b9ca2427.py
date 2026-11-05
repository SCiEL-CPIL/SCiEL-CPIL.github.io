import time
import random

# Precompute 4-APs for fast checking
N = 512
aps_by_element = [[] for _ in range(N + 1)]
for a in range(1, N + 1):
    # d such that a+3d <= N
    max_d = (N - a) // 3
    for d in range(1, max_d + 1):
        b = a + d
        c = a + 2 * d
        e = a + 3 * d
        # The AP is (a, b, c, e)
        aps_by_element[a].append((b, c, e))
        aps_by_element[b].append((a, c, e))
        aps_by_element[c].append((a, b, e))
        aps_by_element[e].append((a, b, c))

def can_add(x, S):
    """
    Check if x can be added to set S without creating a 4-AP.
    S is a set of integers.
    """
    # Check if any 4-AP containing x has its other 3 elements in S
    for others in aps_by_element[x]:
        if others[0] in S and others[1] in S and others[2] in S:
            return False
    return True

def construct():
    start_time = time.time()
    # Keep a safety margin of 10 seconds
    deadline = start_time + 50.0
    
    best_set = set()
    best_size = 0
    
    # We will run multiple restarts with random orderings
    # Each restart:
    # 1. Shuffle [1, N]
    # 2. Greedily add elements if they don't create a 4-AP
    # 3. Do a second pass to add any remaining elements that can now be added
    
    while time.time() < deadline:
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        S = set()
        
        # First pass: Greedy
        for x in candidates:
            if can_add(x, S):
                S.add(x)
        
        # Second pass: Try to add any elements that were skipped
        # This is important because the order in the first pass might have blocked
        # elements that could have been added if added earlier or later.
        # Since S is static during this pass, we just check all missing elements.
        # To be efficient, we iterate through the original range.
        for x in range(1, N + 1):
            if x not in S:
                if can_add(x, S):
                    S.add(x)
        
        # Optional: Third pass? Sometimes adding one element allows another?
        # No, adding elements only restricts future additions.
        # So if we can't add x now, we can't add it later unless we remove something.
        # Since we are only adding, the second pass is sufficient to reach a local maximum
        # for the given "order of consideration" if we consider all elements.
        # Actually, the second pass checks ALL missing elements.
        # If x cannot be added, it's because of elements already in S.
        # Adding more elements to S won't make x addable.
        # So one full scan of missing elements is enough to saturate the set.
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
            # If we found a perfect set (unlikely), we could break, but let's run a bit more
            # or just let the loop handle it.
            
    return sorted(best_set)
