import time
import random
import sys

def _is_valid_set(S):
    """
    Check if S is a valid set.
    Returns True if valid.
    """
    if len(S) < 3:
        return True
        
    S = list(S)
    N = len(S)
    
    # For each element, check if it's the mean of any subset of others
    for i in range(N):
        a = S[i]
        others = S[:i] + S[i+1:]
        n_others = len(others)
        
        if n_others < 2:
            continue
            
        # DP for subset sums by size
        # max_sum = sum(others)
        # We use bitsets: dp[k] is an integer where bit s is set if sum s is achievable with k elements
        max_sum = sum(others)
        
        # Optimization: If max_sum is 0, impossible since distinct positive ints
        # We only care about sums that are multiples of 'a'
        # Target sums: 2*a, 3*a, ..., k*a where k <= n_others
        
        # Initialize dp
        # dp[0] = 1 << 0 (sum 0 with 0 elements)
        dp = [0] * (n_others + 1)
        dp[0] = 1  # bit 0 set
        
        for val in others:
            # Update dp in reverse to avoid reusing same element
            for k in range(n_others - 1, -1, -1):
                if dp[k] == 0:
                    continue
                # Shift by val
                shifted = dp[k] << val
                dp[k+1] |= shifted
                
        # Check if any dp[k] for k >= 2 has a bit set at position m*a
        # m ranges from 2 to n_others
        # Sum = m * a
        for k in range(2, n_others + 1):
            if dp[k] == 0:
                continue
            # Check multiples of a
            # Max sum is max_sum. So m*a <= max_sum => m <= max_sum // a
            max_m = max_sum // a
            # m must be at least 2
            # Also m corresponds to the subset size k? NO.
            # The condition is: Mean = a. Sum = k * a.
            # So we are looking for sum = k * a in dp[k].
            
            # Wait, the subset size IS k.
            # So we just need to check if bit (k * a) is set in dp[k].
            target_sum = k * a
            if target_sum <= max_sum:
                if (dp[k] >> target_sum) & 1:
                    return False
            else:
                # If target_sum > max_sum, no need to check further k?
                # No, for larger k, target_sum increases, but max_sum is fixed.
                # However, max_sum is sum of ALL others.
                # For specific k, the max achievable sum is sum of largest k elements.
                # But simple check target_sum <= max_sum is a necessary condition.
                # If k*a > max_sum, it's impossible.
                pass
                
    return True

def _fast_pair_check(S, x):
    """
    Quick check for pair violations involving x and S.
    Returns True if a pair violation is found.
    """
    S_set = set(S)
    
    # 1. x is mean of a, b in S
    # 2x = a + b
    for a in S:
        b = 2 * x - a
        if b in S_set and b != a:
            # Ensure a and b are distinct and in S
            # If b == a, then 2x = 2a => x=a, but x is new and distinct.
            return True
            
    # 2. Some a in S is mean of x, b in S
    # 2a = x + b => b = 2a - x
    for a in S:
        b = 2 * a - x
        if b in S_set and b != a and b != x:
            return True
            
    return False

def construct():
    start_time = time.time()
    deadline = 85.0
    
    best_set = []
    
    # Candidate pool
    candidates = list(range(1, 513))
    
    # Generate multiple orders to try
    orders = []
    
    # 1. Descending (picks large numbers first, which are less likely to be means of smaller ones? 
    #    Actually, large numbers are less likely to be means of *any* subset because the sum 
    #    would need to be large. But small numbers are more likely to be means of large pairs?
    #    Let's try various orders.)
    orders.append(candidates[::-1])
    orders.append(candidates[:])
    
    # 2. Interleaved
    inter = []
    lo, hi = 1, 512
    while lo <= hi:
        inter.append(hi)
        if lo != hi:
            inter.append(lo)
        lo += 1
        hi -= 1
    orders.append(inter)
    
    # 3. Random orders
    rng = random.Random(42)
    for _ in range(50):
        if time.time() - start_time > deadline - 5:
            break
        rand_order = candidates[:]
        rng.shuffle(rand_order)
        orders.append(rand_order)
        
    for order in orders:
        if time.time() - start_time > deadline - 5:
            break
            
        current_set = []
        
        for x in order:
            if time.time() - start_time > deadline - 2:
                break
                
            if len(current_set) >= 48:
                break
                
            # Fast pair check
            if _fast_pair_check(current_set, x):
                continue
                
            # Full check on current_set + [x]
            # To save time, we can just check the new element x against current_set
            # AND check if any existing element becomes a mean.
            # But _is_valid_set does the full check.
            # Optimization: Only run full check if pair check passes.
            # Since pair violations are common, this prunes many candidates.
            
            test_set = current_set + [x]
            
            # We can optimize _is_valid_set to only check the new element x 
            # AND check if existing elements are affected.
            # But for simplicity and correctness, run full check.
            # With N=48, it's ~0.1s. 
            # If we have 512 candidates, and 50% pass pair check, 256 * 0.1 = 25s. 
            # This might be too slow for 50 random orders.
            # We should limit the number of orders or optimize.
            
            # Let's try to be smarter.
            # Check if x is mean of subset of current_set.
            # Check if any y in current_set is mean of subset of current_set + [x].
            
            # For now, just use the full check but break early if it fails.
            if _is_valid_set(test_set):
                current_set.append(x)
                
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            
        # If we found 48, we can stop searching
        if len(best_set) >= 48:
            break
            
    # Final verification
    if _is_valid_set(best_set):
        return best_set
    else:
        # Fallback
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
