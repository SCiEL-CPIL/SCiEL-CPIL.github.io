import time
import random
from math import gcd

def construct():
    start_time = time.time()
    limit = 512
    # Safety margin: 50s to ensure we finish before 60s/90s limits
    deadline = start_time + 50 
    n = limit

    # Precompute compatibility as bitmasks
    compat_mask = [0] * (n + 1)
    for i in range(1, n + 1):
        mask = 0
        for j in range(1, n + 1):
            if i * j // gcd(i, j) <= n:
                mask |= (1 << j)
        compat_mask[i] = mask

    def can_add(S_mask, x):
        return (S_mask & ~compat_mask[x]) == 0

    def complete(S_mask):
        # Single pass: add all elements that can be added, in order 1..n
        # Optimization: Only iterate over bits not already set?
        # The current implementation iterates 1..n. 
        # We can speed this up by checking only candidates not in S_mask.
        changed = True
        while changed:
            changed = False
            # Iterate only over elements not in S_mask
            # This is still O(N) per pass, but potentially fewer passes?
            # Actually, a single pass in order 1..n is sufficient for "add all possible" 
            # if we define "possible" relative to the *current* state.
            # However, adding x might make y compatible? No, adding elements only restricts the set.
            # If x is compatible with S, and y is compatible with S, they might not be compatible with each other.
            # So adding x doesn't help y. Adding x only hurts.
            # Therefore, a single pass 1..n is correct and sufficient.
            for x in range(1, n + 1):
                if (S_mask & (1 << x)) == 0 and can_add(S_mask, x):
                    S_mask |= (1 << x)
        return S_mask

    def mask_to_set(S_mask):
        return [i for i in range(1, n + 1) if (S_mask & (1 << i))]

    def popcount(x):
        # Use bit_count if available (Python 3.10+), fallback to bin
        if hasattr(int, 'bit_count'):
            return x.bit_count()
        return bin(x).count('1')

    def local_search(S_mask):
        S_mask = complete(S_mask)
        cur_size = popcount(S_mask)
        improved = True
        # Limit iterations to prevent hanging
        max_iters = 50 
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline - 5:
            improved = False
            iters += 1
            S_list = mask_to_set(S_mask)
            
            # 1-for-1 swaps: remove e, add best x, then complete
            for e in S_list:
                if time.time() > deadline - 5:
                    break
                S_minus = S_mask & ~(1 << e)
                best_x = -1
                best_size = cur_size
                
                # Optimization: Pre-filter candidates
                # x must be compatible with S_minus
                # Upper bound on size after adding x:
                # It will include S_minus (size cur_size - 1) + x (1) + any other y in [1..n] not in S_minus
                # that is compatible with S_minus U {x}.
                # A cheap upper bound: (cur_size - 1) + 1 + (number of y not in S_minus U {x} compatible with x)
                # But we don't know which y will be added by complete.
                # Let's just iterate x, but skip if x is in S_mask (except e)
                for x in range(1, n + 1):
                    if x == e or (S_mask & (1 << x)):
                        continue
                    if can_add(S_minus, x):
                        # Quick check: Is x compatible with all in S_minus? Yes, by can_add.
                        # Potential size is at least cur_size. 
                        # To improve, we need complete(S_minus | (1<<x)) to have > cur_size.
                        # This requires that adding x allows adding at least 1 new element that wasn't in S.
                        # Or simply that the closure is larger.
                        
                        # Expensive step: complete
                        S2 = complete(S_minus | (1 << x))
                        s2 = popcount(S2)
                        if s2 > best_size:
                            best_size = s2
                            best_x = x
                            
                if best_x > 0 and best_size > cur_size:
                    S_mask = complete(S_minus | (1 << best_x))
                    cur_size = best_size
                    improved = True

        return S_mask

    best_mask = 0
    best_size = 0

    # Strategy 1: Greedy from small numbers (1..n order)
    S = 0
    for x in range(1, n + 1):
        if can_add(S, x):
            S |= (1 << x)
    S = local_search(S)
    if popcount(S) > best_size:
        best_size = popcount(S)
        best_mask = S

    # Strategy 2: Greedy by compatibility count (most compatible first)
    comp_counts = [popcount(compat_mask[i]) for i in range(n + 1)]
    order = sorted(range(1, n + 1), key=lambda x: -comp_counts[x])
    S = 0
    for x in order:
        if can_add(S, x):
            S |= (1 << x)
    S = local_search(S)
    if popcount(S) > best_size:
        best_size = popcount(S)
        best_mask = S

    # Strategy 3: Start with {1..k} for various k, then complete
    for k in [20, 21, 22, 23, 24, 25, 26]:
        if time.time() > deadline - 10:
            break
        S = 0
        for x in range(1, k + 1):
            S |= (1 << x)
        S = complete(S)
        S = local_search(S)
        if popcount(S) > best_size:
            best_size = popcount(S)
            best_mask = S

    # Strategy 4: Random Restarts with diverse seeds
    rng = random.Random(12345)
    attempts = 0
    while time.time() < deadline - 5:
        attempts += 1
        S = 0
        # Random subset of small numbers as seed
        seed_size = rng.randint(5, 15)
        candidates = list(range(1, min(n, 60)))
        rng.shuffle(candidates)
        for x in candidates[:seed_size]:
            if can_add(S, x):
                S |= (1 << x)
        S = complete(S)
        S = local_search(S)
        if popcount(S) > best_size:
            best_size = popcount(S)
            best_mask = S

    return mask_to_set(best_mask)
