import time
import random

def _can_add_with_dp(dp, current_set, cand, n):
    """
    Check if cand can be added to current_set given the DP table for current_set.
    Returns True if safe. n = len(current_set)
    """
    if n == 0:
        return True
    
    # 1. cand is not mean of any subset of current_set (size >= 2)
    for k in range(2, n + 1):
        target = k * cand
        if (dp[k] >> target) & 1:
            return False
            
    # 2. No a in current_set is mean of subset of (current_set U {cand}) including cand.
    # For each a in current_set, check if there exists T subset of current_set, |T|>=1
    # such that sum(T) + cand = (|T|+1) * a
    # i.e., sum(T) = a*(|T|+1) - cand
    for a in current_set:
        for k in range(1, n + 1):
            target = a * (k + 1) - cand
            if 0 <= target and (dp[k] >> target) & 1:
                return False
    return True

def _update_dp(dp, cand, k_prev):
    """
    Update DP table in-place to include cand.
    k_prev = size of set before adding cand.
    """
    for k in range(k_prev + 1, 0, -1):
        dp[k] |= (dp[k - 1] << cand)

def construct():
    start_time = time.time()
    deadline = start_time + 88.0  # Safety margin for 90s limit
    
    best_set = []
    best_len = 0
    
    nums = list(range(1, 513))
    
    # Generate a large number of random orders
    # We will generate them on the fly to save memory and time
    
    # Strategy 1: Greedy with deterministic orders
    orders = []
    
    # Edge first
    edge_order = []
    l, r = 1, 512
    while l <= r:
        edge_order.append(l)
        if l != r:
            edge_order.append(r)
        l += 1
        r -= 1
    orders.append(edge_order)
    
    # Ascending, descending
    orders.append(list(range(1, 513)))
    orders.append(list(range(512, 0, -1)))
    
    # Run deterministic orders
    for order in orders:
        if time.time() > deadline - 5.0:
            break
            
        current_set_list = []
        max_dp_size = 50
        dp = [0] * (max_dp_size + 1)
        dp[0] = 1
        
        for cand in order:
            if time.time() > deadline - 2.0:
                break
            
            n = len(current_set_list)
            if _can_add_with_dp(dp, current_set_list, cand, n):
                current_set_list.append(cand)
                _update_dp(dp, cand, n)
                
        if len(current_set_list) > best_len:
            best_set = current_set_list[:]
            best_len = len(best_set)
            
            if best_len >= 48:
                return best_set

    # Strategy 2: Many random restarts
    # Estimate time per restart.
    # If we have 80s left, and restart takes 0.05s, we can do 1600 restarts.
    # We will just loop until deadline.
    
    seed = 0
    while time.time() < deadline - 2.0:
        if best_len >= 48:
            break
            
        rnd = random.Random(seed)
        order = list(range(1, 513))
        rnd.shuffle(order)
        seed += 1
        
        current_set_list = []
        max_dp_size = 50
        dp = [0] * (max_dp_size + 1)
        dp[0] = 1
        
        for cand in order:
            if time.time() > deadline - 1.0:
                break
            
            n = len(current_set_list)
            if _can_add_with_dp(dp, current_set_list, cand, n):
                current_set_list.append(cand)
                _update_dp(dp, cand, n)
                
        if len(current_set_list) > best_len:
            best_set = current_set_list[:]
            best_len = len(best_set)
            
            if best_len >= 48:
                return best_set

    return best_set
