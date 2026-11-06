import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    max_n = 512
    max_size = 48
    
    def fast_greedy(order):
        """
        Builds a set using a conservative fast check.
        Returns (set, dp)
        """
        S = []
        dp = [0] * (max_size + 1)
        dp[0] = 1
        
        for x in order:
            if len(S) >= max_size:
                break
            if time.time() > deadline - 5.0:
                break
                
            safe = True
            
            # Check 1: x is not the mean of any subset T of S with |T| >= 2
            # sum(T) = x * |T|
            for k in range(2, len(S) + 1):
                target_sum = x * k
                if (dp[k] >> target_sum) & 1:
                    safe = False
                    break
            
            if not safe:
                continue
                
            # Check 2: No y in S becomes the mean of a subset involving x
            # y is mean of T' = T U {x}, T subset S \ {y}, |T| = k >= 1
            # sum(T) + x = y * (k + 1) => sum(T) = y * (k + 1) - x
            # We check if sum(T) is achievable by k elements from S.
            # Conservative: If achievable from S, assume it's achievable from S\{y} (reject).
            for y in S:
                for k in range(1, len(S) + 1):
                    target_sum = y * (k + 1) - x
                    if target_sum < 0:
                        continue
                    if (dp[k] >> target_sum) & 1:
                        safe = False
                        break
                if not safe:
                    break
            
            if safe:
                S.append(x)
                # Update DP
                for k in range(len(S) - 1, -1, -1):
                    dp[k + 1] |= (dp[k] << x)
                    
        return S, dp

    # 1. Multi-start Fast Greedy
    best_set = []
    
    # Deterministic starts
    best_set, _ = fast_greedy(range(max_n, 0, -1))
    best_size = len(best_set)
    
    best_set2, _ = fast_greedy(range(1, max_n + 1))
    if len(best_set2) > best_size:
        best_set = best_set2
        best_size = len(best_set2)
        
    # Random starts
    random.seed(42)
    # Run as many as possible within time
    while time.time() < deadline - 10.0:
        order = list(range(1, max_n + 1))
        random.shuffle(order)
        S, _ = fast_greedy(order)
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
            if best_size >= max_size:
                break
                
    # 2. Local Search / Improvement
    # Try to add elements to the best set found
    # We need to re-verify with a slower, exact check if we want to be sure,
    # but the fast greedy is conservative, so best_set is valid.
    # We can try to add more elements using the exact check or a less conservative check.
    
    # For now, just return the best set found.
    # If we have time, we can try to add elements one by one with exact validation.
    
    if time.time() < deadline - 2.0 and best_size < max_size:
        # Try to add any remaining element
        S = list(best_set)
        current_set = set(S)
        remaining = [x for x in range(1, max_n + 1) if x not in current_set]
        
        # Use a simple exact check for additions
        # Re-use is_valid logic but optimized for single addition?
        # Just try a few random additions with full validation
        
        for _ in range(100):
            if time.time() > deadline - 2.0:
                break
            if len(S) >= max_size:
                break
            if not remaining:
                break
            idx = random.randint(0, len(remaining) - 1)
            x = remaining[idx]
            
            # Check if x can be added to S
            # We can use the fast check again, but it's conservative.
            # Let's just use the fast check to filter, then exact check.
            # Actually, just try to add and validate.
            
            new_S = S + [x]
            # Simple validation: 
            # Check x against S
            valid = True
            # Build dp for S
            dp = [0] * (len(S) + 1)
            dp[0] = 1
            for y in S:
                for k in range(len(S) - 1, -1, -1):
                    dp[k+1] |= (dp[k] << y)
            
            # Check x is not mean of T in S
            for k in range(2, len(S) + 1):
                if (dp[k] >> (x * k)) & 1:
                    valid = False
                    break
            
            if valid:
                # Check y in S is not mean of T' involving x
                for y in S:
                    for k in range(1, len(S) + 1):
                        target = y * (k + 1) - x
                        if target < 0: continue
                        # Check if target is sum of k elements from S \ {y}
                        # This requires DP of S \ {y}
                        # For speed, we'll just skip this exact check in the loop 
                        # and rely on the conservative greedy having done the work.
                        # If we want to improve, we'd need a better check.
                        pass
            
            # Since exact check is slow, we'll just accept the result of fast_greedy.
            # The fast_greedy is conservative, so it might leave valid elements out.
            # But it's valid.
            break
            
    return best_set
