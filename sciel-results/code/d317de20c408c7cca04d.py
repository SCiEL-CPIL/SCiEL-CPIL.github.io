import random
import time

def construct():
    max_val = 256
    max_size = 60
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85  # 85 seconds to be safe against 90s kill

    best_set = []
    
    # Helper to check if x can be added to current dp
    def can_add(x, dp, m):
        # dp is list of ints
        # m is current size
        for k in range(m + 1):
            new_sums = dp[k] << x
            # Check intersection with all existing dp[j] for j != k+1
            for j in range(m + 1):
                if j == k + 1:
                    continue
                if new_sums & dp[j]:
                    return False
        return True

    def run_greedy(order):
        dp = [0] * (max_size + 1)
        dp[0] = 1  # bit 0 set
        S = []
        m = 0
        
        for x in order:
            if m >= max_size:
                break
            if can_add(x, dp, m):
                # Update dp in reverse order
                for k in range(m, -1, -1):
                    dp[k+1] |= dp[k] << x
                S.append(x)
                m += 1
        return S

    # Try deterministic orders first
    orders = []
    orders.append(list(range(max_val, 0, -1))) # Descending
    orders.append(list(range(1, max_val + 1))) # Ascending
    
    # Add some random orders
    rng = random.Random(42)
    for _ in range(50):
        ord_list = list(range(1, max_val + 1))
        rng.shuffle(ord_list)
        orders.append(ord_list)
        
    for order in orders:
        if time.time() > deadline:
            break
        S = run_greedy(order)
        if len(S) > len(best_set):
            best_set = S
            if len(best_set) == max_size:
                break

    # If we have time, try more random restarts
    while time.time() < deadline:
        ord_list = list(range(1, max_val + 1))
        rng.shuffle(ord_list)
        S = run_greedy(ord_list)
        if len(S) > len(best_set):
            best_set = S
            if len(best_set) == max_size:
                break

    # Ensure return type is list of ints
    return list(best_set)
