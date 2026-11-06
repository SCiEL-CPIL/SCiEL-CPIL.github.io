import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55 seconds to be safe within 60s limit
    n_max = 512
    max_size = 48
    
    best_set = []
    
    def update_dp(dp_list, x):
        new_dp = [0] * len(dp_list)
        new_dp[0] = 1
        for k in range(1, len(dp_list)):
            new_dp[k] = dp_list[k] | (dp_list[k-1] << x)
        return new_dp

    def try_order(order):
        nonlocal best_set
        if len(best_set) >= max_size:
            return
        
        S = []
        # dp[k] is a bitset of sums achievable by subsets of S of size k
        dp = [0] * (max_size + 1)
        dp[0] = 1
        
        # dp_excl[y] is a list of bitsets for S \ {y}
        dp_excl = {}
        
        for x in order:
            if time.time() > deadline:
                return
            
            if x in S:
                continue
            
            # Check 1: Is x the mean of any subset of S?
            # Need k >= 2
            valid = True
            for k in range(2, len(S) + 1):
                target = k * x
                if (dp[k] >> target) & 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check 2: Does adding x make any y in S invalid?
            for y in S:
                # j ranges from 2 to len(S)
                for j in range(2, len(S) + 1):
                    v_size = j - 1
                    target_sum = j * y - x
                    if target_sum < 0:
                        continue
                    if (dp_excl[y][v_size] >> target_sum) & 1:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Add x to S
            S.append(x)
            
            # Update dp
            old_dp = dp
            dp = update_dp(dp, x)
            
            # Update dp_excl
            for y in S:
                if y == x:
                    dp_excl[x] = old_dp
                else:
                    dp_excl[y] = update_dp(dp_excl[y], x)
            
            if len(S) > len(best_set):
                best_set = S[:]
                
            if len(S) == max_size:
                return

    # Orders to try
    orders = [
        list(range(1, n_max + 1)),
        list(range(n_max, 0, -1)),
    ]
    
    # Add some random orders if time permits
    for _ in range(20):
        if time.time() > deadline:
            break
        rand_order = list(range(1, n_max + 1))
        random.shuffle(rand_order)
        orders.append(rand_order)
        
    for order in orders:
        if time.time() > deadline:
            break
        try_order(order)
        
    return best_set
