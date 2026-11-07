import random

def construct():
    max_val = 256
    max_size = 60
    
    # dp[k] is a bitset where bit s is 1 if sum s is achievable with k elements
    # We start with 0 elements, sum 0
    dp = [0] * (max_size + 1)
    dp[0] = 1  # bit 0 is set
    
    S = []
    
    # Try to add numbers from 1 to 256
    # We might want to try different orders if we don't reach max_size
    # But let's start with increasing order
    for x in range(1, max_val + 1):
        if len(S) >= max_size:
            break
            
        # Check if adding x is valid
        # new_dp[k+1] |= (dp[k] << x)
        # We need to check disjointness for the new state
        
        # Copy dp to new_dp
        new_dp = dp[:]
        
        # Update new_dp in reverse order to avoid using x multiple times in one step
        # Actually, we are adding x to the EXISTING set S.
        # So we update based on the current dp.
        # new_dp[k] = dp[k] | (dp[k-1] << x) ?? No.
        # The new set of sums for cardinality k with x added is:
        # Sums_k(new) = Sums_k(old) U { s + x | s in Sums_{k-1}(old) }
        
        # So:
        # new_dp[k] = dp[k] | (dp[k-1] << x)
        
        # Let's compute new_dp
        # We only need to check up to len(S) + 1
        current_max_k = len(S)
        
        # Initialize new_dp for the relevant range
        # dp is already correct for 0..current_max_k
        # We need to update up to current_max_k + 1
        
        # Note: dp[k] for k > current_max_k is 0
        
        valid = True
        # We need to compute the full new_dp to check disjointness
        # But we can do it efficiently
        
        # Let's just compute the full new_dp for 0..max_size
        # It's fast enough.
        
        temp_dp = [0] * (max_size + 1)
        for k in range(max_size + 1):
            # Sums from not using x
            temp_dp[k] |= dp[k]
            # Sums from using x (if k > 0)
            if k > 0:
                temp_dp[k] |= (dp[k-1] << x)
        
        # Check disjointness
        # For all i != j, temp_dp[i] & temp_dp[j] must be 0
        # This is O(K^2) bit operations
        # K is at most 60. 60*60 = 3600 ops. Very fast.
        
        # Optimization: We can check incrementally or just run the full check
        for i in range(max_size + 1):
            if temp_dp[i] == 0:
                continue
            for j in range(i + 1, max_size + 1):
                if temp_dp[j] == 0:
                    continue
                if (temp_dp[i] & temp_dp[j]) != 0:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            dp = temp_dp
            S.append(x)
            
    # If we didn't reach max_size, we might try a different strategy
    # But the greedy is usually strong.
    # Let's ensure we return a list of ints
    
    return S

# To improve robustness, we can try a few random shuffles if the first attempt fails
# But the prompt asks for a single construct function.
# Let's add a simple retry mechanism with different orders if needed.

def construct():
    max_val = 256
    max_size = 60
    
    def try_order(order):
        dp = [0] * (max_size + 1)
        dp[0] = 1
        S = []
        
        for x in order:
            if len(S) >= max_size:
                break
            
            current_max_k = len(S)
            
            # Compute new_dp
            temp_dp = [0] * (max_size + 1)
            for k in range(max_size + 1):
                temp_dp[k] |= dp[k]
                if k > 0:
                    temp_dp[k] |= (dp[k-1] << x)
            
            # Check disjointness
            valid = True
            for i in range(max_size + 1):
                if temp_dp[i] == 0:
                    continue
                for j in range(i + 1, max_size + 1):
                    if temp_dp[j] == 0:
                        continue
                    if (temp_dp[i] & temp_dp[j]) != 0:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                dp = temp_dp
                S.append(x)
                
        return S

    # Try increasing order
    S = try_order(range(1, max_val + 1))
    if len(S) == max_size:
        return S
        
    # Try decreasing order
    S = try_order(range(max_val, 0, -1))
    if len(S) == max_size:
        return S
        
    # Try a few random orders
    import random
    for _ in range(10):
        order = list(range(1, max_val + 1))
        random.shuffle(order)
        S = try_order(order)
        if len(S) == max_size:
            return S
            
    # Return the best one found
    # In this case, we just return the last one, or the largest
    # Let's keep track of the best
    best_S = []
    
    def run_all():
        best = []
        orders = [
            range(1, max_val + 1),
            range(max_val, 0, -1)
        ]
        for _ in range(20):
            o = list(range(1, max_val + 1))
            random.shuffle(o)
            orders.append(o)
            
        for o in orders:
            S = try_order(o)
            if len(S) > len(best):
                best = S
                if len(best) == max_size:
                    break
        return best

    S = run_all()
    return S
