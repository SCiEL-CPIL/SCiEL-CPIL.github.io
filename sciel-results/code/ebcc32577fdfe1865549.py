import random
import math

def construct():
    max_val = 256
    max_size = 60
    
    def calculate_cost(S):
        # S is a list of 60 ints
        dp = [0] * (max_size + 1)
        dp[0] = 1
        for x in S:
            # Update in reverse to avoid using x multiple times
            for k in range(max_size - 1, -1, -1):
                if dp[k]:
                    dp[k+1] |= (dp[k] << x)
        
        # Check disjointness efficiently
        # seen holds all sums achieved by any cardinality so far
        seen = 0
        for k in range(max_size + 1):
            if dp[k]:
                if (seen & dp[k]) != 0:
                    # We found at least one conflict.
                    # For SA, we might want the count, but 0 vs non-zero is often enough
                    # to guide towards 0.
                    # Let's return a small number > 0 to indicate invalid.
                    # Or count the bits? Counting bits is expensive.
                    # Let's just return 1 if invalid, 0 if valid.
                    # This makes the landscape rugged (binary), but SA can still work.
                    # To improve, we can count how many bits are in the intersection.
                    # (seen & dp[k]).bit_count()
                    # This is fast in Python 3.10+.
                    return (seen & dp[k]).bit_count()
                seen |= dp[k]
        return 0

    best_S = []
    best_cost = float('inf')
    
    # Try a few restarts of Simulated Annealing
    for restart in range(100):
        # Initialize with a random set
        S = random.sample(range(1, max_val + 1), max_size)
        cost = calculate_cost(S)
        
        if cost == 0:
            return S
        
        # SA parameters
        T = 5.0
        # Limit steps to keep total time low
        for step in range(2000):
            # Pick a random index to swap
            i = random.randint(0, max_size - 1)
            # Pick a random value not in S
            # Use a set for O(1) lookup? 
            # S is a list. Converting to set every time is O(N).
            # We can maintain a set alongside S.
            # For simplicity, let's just retry.
            b = random.randint(1, max_val)
            # Optimization: if b in S, keep generating
            # Since 60/256 is small, this is fast.
            while b in S:
                b = random.randint(1, max_val)
                
            old_a = S[i]
            S[i] = b
            new_cost = calculate_cost(S)
            
            delta = new_cost - cost
            
            if delta < 0:
                cost = new_cost
                if cost == 0:
                    return S
            else:
                # Accept with probability exp(-delta / T)
                # If delta is large, prob is low.
                if random.random() < math.exp(-delta / T):
                    cost = new_cost
                else:
                    S[i] = old_a # Revert
            
            # Cool down
            T *= 0.995
            
        if cost < best_cost:
            best_cost = cost
            best_S = S[:]
            
    # Fallback to greedy if SA didn't find a perfect solution
    # This ensures we return a valid set (possibly smaller)
    def greedy_construct():
        dp = [0] * (max_size + 1)
        dp[0] = 1
        S = []
        for x in range(1, max_val + 1):
            if len(S) >= max_size:
                break
            
            temp_dp = dp[:]
            for k in range(len(S), -1, -1):
                if dp[k]:
                    temp_dp[k+1] |= (dp[k] << x)
            
            # Check validity
            seen = 0
            valid = True
            for k in range(len(S) + 2):
                if temp_dp[k]:
                    if (seen & temp_dp[k]) != 0:
                        valid = False
                        break
                    seen |= temp_dp[k]
            
            if valid:
                dp = temp_dp
                S.append(x)
        return S

    if best_cost != 0:
        return greedy_construct()
    
    return best_S
