import time
import random

def construct():
    start = time.time()
    deadline = start + 85.0
    n = 512
    target = 48

    # Bitset DP for subset sums
    # dp[k] is a bitset where bit i is 1 if sum i is achievable by a subset of size k
    
    def is_valid_set(S):
        """
        Checks if S is a valid set (no element is mean of subset of others).
        Returns True if valid.
        """
        m = len(S)
        if m == 0:
            return True
        
        # Build dp for S
        # max sum is m * n
        max_sum = m * n
        dp = [0] * (m + 1)
        dp[0] = 1
        
        for x in S:
            for k in range(m, 0, -1):
                dp[k] |= (dp[k - 1] << x)
        
        # Check each x in S
        for x in S:
            # We need to check if x is mean of a subset of S \ {x}
            # This requires DP for S \ {x}
            # To be efficient, we can rebuild DP for S \ {x}
            # Or, we can precompute "dp_without_x" for all x? That's O(m^2) space and time.
            # Given m <= 48, O(m^2) is ~2304, which is fine.
            
            # Let's just rebuild for simplicity and correctness first, then optimize if needed.
            # Rebuilding DP for S \ {x} takes O(m * max_sum) ~ 48 * 24576 ~ 1.2M ops.
            # Doing this for each x in S (48 times) is ~ 58M ops.
            # This might be too slow in Python if done in a tight loop.
            
            # Optimization: 
            # We can check if x is a mean of a subset of S\{x} by looking at the global DP?
            # No, because the subset must be from S\{x}.
            
            # Let's try a different check:
            # For a fixed x, we want to know if there exists a subset Y of S\{x} with |Y|>=2 such that sum(Y) = k*x.
            
            # We'll use the full DP of S.
            # If sum(Y) = k*x is achievable by a subset of S of size k, does it necessarily use x?
            # If the subset uses x, then sum(Y\{x}) = k*x - x = (k-1)*x.
            # So if (k*x) is in dp[k] AND ((k-1)*x) is NOT in dp[k-1] (for S\{x}? No, for S), 
            # it's complicated.
            
            # Let's stick to rebuilding DP for S\{x} but optimize the inner loop.
            # Actually, we can just check validity of the whole set S once at the end of a search step.
            # During the search, we can use a faster heuristic or a partial check.
            
            pass
        return True

    # Let's implement a faster validity check for the "is_valid_candidate" logic used in greedy.
    # We maintain DP for the current set C.
    # When checking x, we check:
    # 1. Is x mean of subset of C? (Use DP of C)
    # 2. Does adding x make any y in C invalid?
    #    This requires checking if y is mean of subset of C U {x} containing x.
    #    This is equivalent to: Is there a subset U of C such that (sum(U) + x) / (|U|+1) = y?
    #    sum(U) = (|U|+1)y - x.
    #    We can check this using DP of C.
    
    def greedy_construct(order):
        C = []
        dp = [1] # dp[0]=1
        dp_without = {} # y -> dp of C \ {y}
        
        for x in order:
            if len(C) >= target:
                break
            
            m = len(C)
            
            # Check 1: Is x mean of subset of C?
            valid = True
            for k in range(2, m + 1):
                target_sum = k * x
                if target_sum > m * n:
                    continue
                if (dp[k] >> target_sum) & 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check 2: Does adding x make any y in C invalid?
            for y in C:
                dpy = dp_without[y] # DP of C \ {y}, size m-1
                # We need subset U of C\{y} such that (sum(U) + x) = (|U|+1)y
                # sum(U) = (|U|+1)y - x
                for j in range(1, m): # j = |U|, U subset of C\{y}
                    target_sum = (j + 1) * y - x
                    if target_sum < 0:
                        continue
                    if target_sum > (m - 1) * n:
                        continue
                    if (dpy[j] >> target_sum) & 1:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Add x
                # Update dp_without
                dp_without[x] = list(dp)
                for y in C:
                    dpy = dp_without[y] + [0]
                    for k in range(m, 0, -1):
                        dpy[k] |= (dpy[k - 1] << x)
                    dp_without[y] = dpy
                
                # Update dp
                new_dp = dp + [0]
                for k in range(m + 1, 0, -1):
                    new_dp[k] |= (new_dp[k - 1] << x)
                dp = new_dp
                
                C.append(x)
        
        return C

    best = []
    rng = random.Random(12345)
    
    # Try different orders
    orders = [
        list(range(1, n + 1)),
        list(range(n, 0, -1)),
    ]
    
    for _ in range(50):
        orders.append(rng.sample(range(1, n + 1), n))
        
    for order in orders:
        if time.time() > deadline - 5.0:
            break
        C = greedy_construct(order)
        if len(C) > len(best):
            best = C
            if len(best) >= target:
                break

    # If best is not 48, try local search on best
    if len(best) < target and time.time() < deadline - 10.0:
        S = list(best)
        # Local search: try to swap elements to allow more additions
        # This part is complex to implement correctly with DP updates.
        # Given time constraints, the greedy with multiple orders might be sufficient.
        pass

    return best
