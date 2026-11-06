import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    max_n = 512
    max_size = 48
    max_sum = max_n * max_size  # 24576

    # Precompute static safety scores:
    # How many "conflict partners" does x have?
    # For k=2: count of pairs (u,v) with u+v=2x.
    # For k=3: count of triples (u,v,w) with u+v+w=3x.
    # We approximate by just counting k=2 conflicts, as they are most common.
    # score[x] = number of y in [1, 512] \ {x} such that 2x-y is in [1, 512] and != x, y
    safety_scores = [0] * (max_n + 1)
    for x in range(1, max_n + 1):
        count = 0
        # u < x < v, u+v = 2x
        # u ranges from max(1, 2x - max_n) to x - 1
        start_u = max(1, 2 * x - max_n)
        end_u = x - 1
        if start_u <= end_u:
            count = end_u - start_u + 1
        safety_scores[x] = count

    # Order by safety (lowest score first)
    order_safe = sorted(range(1, max_n + 1), key=lambda x: safety_scores[x])
    order_edge = sorted(range(1, max_n + 1), key=lambda x: -abs(x - 256.5))
    order_asc = range(1, max_n + 1)
    order_desc = range(max_n, 0, -1)

    def build_dp(S):
        """Build dp table for set S."""
        dp = [0] * (max_size + 1)
        dp[0] = 1
        for x in S:
            for k in range(len(S) - 1, -1, -1):
                dp[k + 1] |= (dp[k] << x)
        return dp

    def check_valid(x, S, dp):
        """
        Check if x can be added to S.
        """
        n_s = len(S)
        
        # Check 1: x is not the mean of any subset T of S with |T| >= 2
        # sum(T) = x * |T|
        for k in range(2, n_s + 1):
            target_sum = x * k
            if target_sum > max_sum:
                continue
            if (dp[k] >> target_sum) & 1:
                return False
        
        # Check 2: No y in S becomes the mean of a subset involving x
        # y = mean(T' = T U {x}), T subset S \ {y}, |T| = j >= 1
        # sum(T) + x = y * (j + 1) => sum(T) = y * (j + 1) - x
        # Conservative: Check if sum(T) is achievable by j elements from S.
        # If yes, we reject x. (This might reject valid x if the subset T includes y)
        for y in S:
            for j in range(1, n_s + 1):
                target_sum = y * (j + 1) - x
                if target_sum < 0 or target_sum > max_sum:
                    continue
                if (dp[j] >> target_sum) & 1:
                    return False
                    
        return True

    def greedy_build(order, S_init=None):
        """
        Builds a set using the given order of candidates.
        """
        S = list(S_init) if S_init else []
        if not S:
            dp = [0] * (max_size + 1)
            dp[0] = 1
        else:
            dp = build_dp(S)
        
        # Convert S to a set for O(1) lookup
        S_set = set(S)
        
        for x in order:
            if len(S) >= max_size:
                break
            if time.time() > deadline - 5.0:
                break
            if x in S_set:
                continue
            
            if check_valid(x, S, dp):
                S.append(x)
                S_set.add(x)
                # Update DP
                for k in range(len(S) - 1, -1, -1):
                    dp[k + 1] |= (dp[k] << x)
                    
        return S

    def local_search(S):
        """
        Attempts to improve the set S by swapping elements.
        Returns a potentially better set.
        """
        if time.time() > deadline - 10.0:
            return S
            
        S = list(S)
        dp = build_dp(S)
        S_set = set(S)
        
        # Try to add elements first
        # We scan all candidates to see if any can be added
        # This is O(N * check_cost)
        
        improved = True
        while improved and len(S) < max_size:
            improved = False
            for x in range(1, max_n + 1):
                if x in S_set:
                    continue
                if time.time() > deadline - 5.0:
                    return S
                if check_valid(x, S, dp):
                    S.append(x)
                    S_set.add(x)
                    for k in range(len(S) - 1, -1, -1):
                        dp[k + 1] |= (dp[k] << x)
                    improved = True
                    break # Restart scan as DP changed
        
        if len(S) == max_size:
            return S
            
        # If stuck, try swaps
        # Pick a random element to remove, then try to add any valid element
        # We do this for a limited number of steps
        max_swaps = 500
        for _ in range(max_swaps):
            if time.time() > deadline - 5.0:
                return S
                
            # Pick a random element from S to remove
            # We prefer removing elements that are "central" or have high conflict score
            # For simplicity, random
            idx = random.randint(0, len(S) - 1)
            removed = S[idx]
            
            # Remove it
            S.pop(idx)
            S_set.remove(removed)
            dp = build_dp(S) # Rebuild DP, cost is O(|S|^2 * bit_ops)
            
            # Try to add any valid element
            added = -1
            for x in range(1, max_n + 1):
                if x in S_set:
                    continue
                if check_valid(x, S, dp):
                    added = x
                    break
            
            if added != -1:
                # Accept swap
                S.append(added)
                S_set.add(added)
                # Update DP
                for k in range(len(S) - 1, -1, -1):
                    dp[k + 1] |= (dp[k] << added)
            else:
                # Revert
                S.append(removed)
                S_set.add(removed)
                dp = build_dp(S)
                
            # If we added something, we might be able to add more, 
            # but the loop structure above only adds one. 
            # The "improved" loop at the start will catch further additions in the next iteration if we restructure.
            # For now, we just do swaps.
            
        return S

    best_set = []
    best_size = 0

    # 1. Deterministic Greedies
    S1 = greedy_build(order_safe)
    best_set = S1
    best_size = len(best_set)
    
    S2 = greedy_build(order_edge)
    if len(S2) > best_size:
        best_set = S2
        best_size = len(best_set)
        
    S3 = greedy_build(order_desc)
    if len(S3) > best_size:
        best_set = S3
        best_size = len(best_set)

    # 2. Local Search on Best Set
    if time.time() < deadline - 15.0:
        S_ls = local_search(best_set)
        if len(S_ls) > best_size:
            best_set = S_ls
            best_size = len(best_set)

    # 3. Randomized Restarts
    random.seed(42)
    while time.time() < deadline - 10.0:
        # Create a noisy safe order
        order = list(order_safe)
        # Shuffle the bottom 50% to introduce variation
        half = max_n // 2
        random.shuffle(order[half:])
        
        S_rand = greedy_build(order)
        
        # Apply local search if it's promising (close to best)
        if len(S_rand) >= best_size - 2 and len(S_rand) < max_size:
            if time.time() < deadline - 5.0:
                S_rand = local_search(S_rand)
                
        if len(S_rand) > best_size:
            best_set = S_rand
            best_size = len(best_set)
            if best_size >= max_size:
                break

    return best_set
