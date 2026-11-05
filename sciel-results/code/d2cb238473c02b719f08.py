import time
import random

def construct():
    deadline = time.time() + 85.0  # Safety margin
    
    # Candidate pool: All numbers 1 to 512
    # We will try multiple random orders
    all_candidates = list(range(1, 513))
    
    best_set = []
    best_size = 0
    
    random.seed(42)
    
    # Number of restarts to attempt
    # We estimate time per restart.
    # If one restart takes 0.1s, we can do 800 restarts.
    # If it takes 1s, we can do 80.
    
    def build_set(order):
        selected = []
        selected_set = set()
        
        for x in order:
            if time.time() > deadline - 1.0:
                break
            if x in selected_set:
                continue
            
            n = len(selected)
            valid = True
            
            if n >= 2:
                # Check 1: Is x the mean of any subset of selected?
                # If order is descending, x < min(selected), so mean(selected) > x.
                # So we can skip this check if we ensure descending order.
                # But we are using random order.
                # Let's do the check.
                
                # Build DP for selected
                max_sum = sum(selected)
                dp = [0] * (n + 1)
                dp[0] = 1
                for elem in selected:
                    for k in range(n, 0, -1):
                        dp[k] |= (dp[k-1] << elem)
                
                for k in range(2, n + 1):
                    target = k * x
                    if target <= max_sum and (dp[k] >> target) & 1:
                        valid = False
                        break
            
            if valid and n >= 1:
                # Check 2: Is any y in selected the mean of a subset involving x?
                # For each y in selected:
                # Check if exists U subset of selected\{y} such that sum(U) = (|U|+1)*y - x
                for y in selected:
                    others = [z for z in selected if z != y]
                    m = len(others)
                    if m == 0:
                        continue
                    
                    max_others = sum(others)
                    # Build DP for others
                    dp2 = [0] * (m + 1)
                    dp2[0] = 1
                    for elem in others:
                        for k in range(m, 0, -1):
                            dp2[k] |= (dp2[k-1] << elem)
                    
                    for k in range(1, m + 1):
                        target = (k + 1) * y - x
                        if 0 <= target <= max_others:
                            if (dp2[k] >> target) & 1:
                                valid = False
                                break
                    if not valid:
                        break
            
            if valid:
                selected.append(x)
                selected_set.add(x)
                if len(selected) >= 48:
                    break
        
        return selected

    # Strategy 1: Descending Order (often good for mean constraints)
    order_desc = sorted(all_candidates, reverse=True)
    res = build_set(order_desc)
    if len(res) > best_size:
        best_size = len(res)
        best_set = res
    
    # Strategy 2: Random Restarts
    # Use time to determine how many restarts
    start = time.time()
    while time.time() < deadline - 2.0:
        random.shuffle(all_candidates)
        res = build_set(all_candidates)
        if len(res) > best_size:
            best_size = len(res)
            best_set = res
            if best_size == 48:
                break

    return best_set
