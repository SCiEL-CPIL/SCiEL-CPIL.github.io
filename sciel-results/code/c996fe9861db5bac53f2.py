import time
import random
import math

def construct():
    # Denominators 2 to 48
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of 2..48
    # LCM(1..48)
    # Primes powers: 2^4, 3^3, 5^2, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47
    L = 1
    # 2^4
    L *= 16
    # 3^3
    L *= 27
    # 5^2
    L *= 25
    # 7
    L *= 7
    # 11
    L *= 11
    # 13
    L *= 13
    # 17
    L *= 17
    # 19
    L *= 19
    # 23
    L *= 23
    # 29
    L *= 29
    # 31
    L *= 31
    # 37
    L *= 37
    # 41
    L *= 41
    # 43
    L *= 43
    # 47
    L *= 47
    
    weights = [L // d for d in denominators]
    target = L - 1
    
    # Initial greedy solution
    # Sort by weight descending (smallest denominator first)
    indices = list(range(n))
    indices.sort(key=lambda i: weights[i], reverse=True)
    
    best_sum = 0
    best_set = set()
    
    curr_sum = 0
    curr_set = set()
    for i in indices:
        if curr_sum + weights[i] <= target:
            curr_sum += weights[i]
            curr_set.add(i)
    
    best_sum = curr_sum
    best_set = set(curr_set)
    
    start_time = time.time()
    time_limit = 55.0 # 5s margin
    
    # Precompute list of indices for random access
    all_indices = list(range(n))
    
    def try_improve(curr_set, curr_sum):
        nonlocal best_sum, best_set
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            
            # 1. Try adding items
            # To be fast, we can check all not in set
            for j in all_indices:
                if j not in curr_set:
                    if curr_sum + weights[j] <= target:
                        curr_sum += weights[j]
                        curr_set.add(j)
                        improved = True
                        if curr_sum > best_sum:
                            best_sum = curr_sum
                            best_set = set(curr_set)
            
            # 2. Try swapping items
            # Iterate over items in set
            # To avoid modifying list while iterating, make a copy of list
            set_list = list(curr_set)
            for i in set_list:
                # Check if removing i allows adding something better
                # Or just simple swap: remove i, add j if j > i and fits
                # But we want to maximize sum.
                # If we remove i, sum decreases by w_i.
                # We can add any j not in set such that new_sum <= target.
                # To improve, we need new_sum > curr_sum - w_i.
                # Ideally new_sum > curr_sum.
                # This requires w_j > w_i.
                # So we look for j not in set with w_j > w_i and curr_sum - w_i + w_j <= target.
                
                # Optimization: Only try swaps if there exists j with w_j > w_i
                # and curr_sum - w_i + w_j <= target.
                # This is a subset of the "not in set" items.
                
                # Let's just iterate all j not in set for swaps? 
                # N=47, so 47*47 ~ 2200 checks per pass. Very fast.
                for j in all_indices:
                    if j not in curr_set:
                        new_sum = curr_sum - weights[i] + weights[j]
                        if new_sum > curr_sum and new_sum <= target:
                            curr_set.remove(i)
                            curr_set.add(j)
                            curr_sum = new_sum
                            improved = True
                            if curr_sum > best_sum:
                                best_sum = curr_sum
                                best_set = set(curr_set)
                            break # Restart inner loop for this i? Or just continue?
                                   # Breaking out of j loop is fine, we'll pick up other i's or restart.
                                   # Actually, if we swap, curr_sum changes, so other swaps might be valid now.
                                   # Breaking the j loop allows us to re-evaluate other i's or j's in next 'improved' iteration.
    
    # Main Search Loop
    # We alternate between improving the best found and doing random restarts
    
    # First, improve the initial greedy solution
    try_improve(best_set, best_sum)
    
    # Now, random restarts
    # Strategy: Start with a random subset of the "large" weights, then greedily fill.
    # Or start empty and greedily add random items.
    
    while time.time() - start_time < time_limit:
        # Random Restart
        curr_set = set()
        curr_sum = 0
        
        # Randomly select a few large items to start?
        # Or just greedy fill with random order.
        
        # Shuffle indices
        rand_indices = all_indices.copy()
        random.shuffle(rand_indices)
        
        # Greedy fill in random order
        for i in rand_indices:
            if curr_sum + weights[i] <= target:
                curr_sum += weights[i]
                curr_set.add(i)
        
        # If this random greedy is better, update best
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_set = set(curr_set)
        
        # Try to improve this random solution
        # Limit time per restart? 
        # Just call try_improve. It has its own time check.
        # But if it takes too long on one bad solution, we waste time.
        # We can add a time check inside try_improve which we already have.
        
        try_improve(curr_set, curr_sum)
        
        # Also, periodically try to improve the BEST solution again
        # to see if we can squeeze more out of it.
        if random.random() < 0.1:
            try_improve(best_set, best_sum)

    # Convert best_set to list of indices
    return list(best_set)
