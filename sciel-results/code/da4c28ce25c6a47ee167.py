import time
import random
import sys

def construct():
    deadline = time.time() + 55.0
    
    # Upper bound on size is 11
    # We try to find a set of size 11, then 10, etc.
    
    def is_valid_extension(current_sums, x):
        """
        Check if adding x to the set with subset sums current_sums is valid.
        Returns True if valid, False otherwise.
        Also returns the new set of sums if valid.
        """
        # Generate new sums
        # New sums are x and x+s for s in current_sums
        new_sums = set()
        new_sums.add(x)
        for s in current_sums:
            ns = x + s
            if ns in current_sums:
                return False, None
            if ns in new_sums:
                return False, None
            new_sums.add(ns)
        
        # Check divisibility
        # 1. Any new_sum divisible by any old_sum?
        # 2. Any old_sum divisible by any new_sum?
        
        # Optimization: 
        # If min(new_sums) > max(current_sums), then case 2 is impossible.
        min_new = min(new_sums)
        max_old = max(current_sums)
        
        # Case 1: s_new % s_old == 0
        for ns in new_sums:
            for os in current_sums:
                if ns % os == 0:
                    return False, None
        
        # Case 2: s_old % s_new == 0
        if min_new <= max_old:
            for os in current_sums:
                for ns in new_sums:
                    if os % ns == 0:
                        return False, None
        
        return True, new_sums.union(current_sums)

    def backtrack(start_val, current_set, current_sums, target_size):
        """
        Try to build a set of size target_size.
        start_val: minimum value for the next element.
        current_set: list of elements so far.
        current_sums: set of subset sums so far.
        """
        if time.time() > deadline:
            return None
            
        if len(current_set) == target_size:
            return current_set[:]
            
        # Heuristic: Try values in a specific range
        # To ensure distinct sums, we generally need larger values.
        # But we can try a range.
        
        # Determine max possible value
        # Sum of remaining elements must be such that total sum <= 256 * (target_size - len(current_set))
        # Actually, each element <= 256.
        
        # Let's try candidates from start_val to 256.
        # To improve search, we can randomize the order or use a heuristic.
        # Heuristic: Larger elements are less likely to cause collisions with existing sums?
        # Actually, larger elements make it *easier* to satisfy distinctness if we are superincreasing, 
        # but we are not restricted to superincreasing.
        
        # Let's try a mix: some small, some large.
        # For now, iterate from start_val to 256.
        
        for x in range(start_val, 257):
            if time.time() > deadline:
                return None
                
            valid, new_sums = is_valid_extension(current_sums, x)
            if valid:
                res = backtrack(x + 1, current_set + [x], new_sums, target_size)
                if res:
                    return res
        return None

    # Try sizes from 11 down to 1
    # We allocate more time to larger sizes? Or just try them in order.
    # If we find a 11, we are done.
    # If we fail 11, we try 10.
    
    # To avoid spending all time on 11 if it's impossible, we can limit the depth or time per size.
    # But backtracking with pruning might be fast.
    
    # Let's add a restart mechanism with randomization if simple backtracking fails.
    
    best_set = []
    
    # Try deterministic backtracking for size 11
    # Start with 1? Or 2?
    # Starting with 1 is risky (1 divides everything).
    # So 1 cannot be in the set? 
    # If 1 is in the set, subset sum 1 exists. Any other subset sum s must not be divisible by 1. 
    # But every integer is divisible by 1. So if 1 is in the set, NO other subset sum can exist?
    # Wait. "No two distinct resulting values may divide one another."
    # If 1 is a sum, and s is another sum, 1 divides s. So this is invalid.
    # Therefore, 1 cannot be in the set unless the set has only 1 element? 
    # If set is {1}, sums = {1}. Only one sum. No pair of distinct sums. Valid.
    # If set is {1, 2}, sums = {1, 2, 3}. 1 divides 2, 1 divides 3. Invalid.
    # So if size > 1, 1 cannot be in the set.
    
    # Similarly, if 2 is in the set, sum 2 exists. No other sum can be even? 
    # And no other sum can divide 2? (So no sum 1, which is already excluded).
    # So all other sums must be odd and not divide 2 (which is true for odd > 2).
    # But if we have {2, 3}, sums {2, 3, 5}. 2 and 3 don't divide each other. 2 and 5 don't. 3 and 5 don't.
    # But wait, 2+3=5.
    # Is {2, 3} valid?
    # Sums: 2, 3, 5.
    # Pairs: (2,3), (2,5), (3,5).
    # 2|3? No. 3|2? No.
    # 2|5? No. 5|2? No.
    # 3|5? No. 5|3? No.
    # So {2, 3} is valid. Size 2.
    
    # Let's start backtracking from 2.
    
    # We will try to find a set of size 11.
    # To speed up, we can use a heuristic for the first few elements.
    
    # Let's run backtracking for target_size 11.
    # If it returns None, try 10, etc.
    
    # To prevent hanging, we check deadline.
    
    for target_size in range(11, 0, -1):
        # Randomize start?
        # Let's try a few random restarts for the first element if deterministic fails quickly?
        # No, deterministic backtracking with good pruning is usually best.
        
        # Heuristic: The first element should not be too small.
        # If a1 is small, it divides many things.
        # Let's try starting from 2.
        
        res = backtrack(2, [], {0}, target_size) 
        # Note: current_sums should contain 0? 
        # My is_valid_extension assumes current_sums are non-empty sums?
        # "Form every NONEMPTY subset sum".
        # So initial sums set should be empty?
        # If current_set is empty, current_sums is empty.
        # Adding x: new_sums = {x}.
        # Check divisibility: no old sums. Valid.
        # So start with current_sums = set().
        
        # Let's re-run with correct initialization.
        
        if res:
            return res
        
    # Fallback: return a known valid set
    return [2, 3]

# Wait, I need to fix the `backtrack` call initialization.
# In the code above, I passed {0}. That's wrong.
# Let's rewrite the construct function properly.
