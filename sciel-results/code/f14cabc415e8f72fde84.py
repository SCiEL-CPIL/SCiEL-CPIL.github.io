import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    max_val = 256
    max_size = 60
    max_sum = max_size * max_val  # 15360
    
    def check_disjoint(dp):
        """Check that all dp[k] for different k are pairwise disjoint."""
        for i in range(len(dp)):
            for j in range(i + 1, len(dp)):
                if dp[i] & dp[j]:
                    return False
        return True
    
    def try_order(order):
        """Try adding elements in the given order, return the valid set."""
        dp = [0] * (max_size + 1)
        dp[0] = 1  # bit 0 set: sum 0 with 0 elements
        S = []
        
        for x in order:
            if len(S) >= max_size:
                break
            if time.time() > deadline:
                break
            
            # Try adding x
            new_dp = dp[:]
            # Update from high k to low k to avoid using x multiple times
            for k in range(len(S), -1, -1):
                if new_dp[k]:
                    new_dp[k + 1] |= (new_dp[k] << x)
            
            # Check disjointness
            if check_disjoint(new_dp):
                dp = new_dp
                S.append(x)
        
        return S
    
    # Strategy 1: Ascending order
    best_set = try_order(range(1, max_val + 1))
    
    # Strategy 2: Descending order
    if time.time() < deadline:
        desc_set = try_order(range(max_val, 0, -1))
        if len(desc_set) > len(best_set):
            best_set = desc_set
    
    # Strategy 3: Try several random permutations
    if time.time() < deadline:
        for attempt in range(200):
            if time.time() > deadline - 2.0:
                break
            random.seed(attempt * 12345 + 67890)
            order = list(range(1, max_val + 1))
            random.shuffle(order)
            s = try_order(order)
            if len(s) > len(best_set):
                best_set = s
    
    # Ensure we return Python ints in a list
    return [int(x) for x in best_set]
