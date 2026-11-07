import time
import random
import sys

# EVOLVE-BLOCK-START
P = {'n': 256, 'maximum_size': 30}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best_size = 0
    best_set = []
    
    # Helper to check validity
    def is_valid(s):
        # s is a list of distinct ints
        k = len(s)
        if k == 0:
            return True
        # Generate all non-empty subset sums
        # Using a set to handle duplicates (though distinct elements usually imply distinct sums if superincreasing, 
        # but not generally. The problem says "Equal sum values are counted once" but 
        # "Two different subsets with the same sum still make the set invalid".
        # So we MUST check for duplicate sums first.
        
        # Efficient subset sum generation
        sums = set()
        # Start with 0 (empty set) to make iteration easier, then remove 0
        current_sums = {0}
        for x in s:
            new_sums = set()
            for u in current_sums:
                new_sums.add(u + x)
            current_sums.update(new_sums)
        
        # Remove 0
        current_sums.discard(0)
        
        # Check for duplicates: if len(current_sums) != 2**k - 1, there are duplicates
        # Wait, 2**k - 1 is the number of non-empty subsets.
        # If any two subsets have the same sum, the size of the set of sums will be less than 2**k - 1.
        if len(current_sums) != (2**k - 1):
            return False
            
        # Check divisibility
        # Sort sums to potentially prune? Not strictly needed for the multiple check.
        # For each sum s, check if any multiple 2s, 3s... is in the set.
        for s_val in current_sums:
            # Optimization: if s_val is large, few multiples.
            # Max sum is sum(s)
            max_sum = sum(s)
            multiple = 2 * s_val
            while multiple <= max_sum:
                if multiple in current_sums:
                    return False
                multiple += s_val
        return True

    # Strategy: Randomized Greedy with Bias towards Larger Numbers
    # We try to find a set of size k.
    # We start with a moderate k and try to go higher.
    # Or just run greedy and see what size we get.
    
    # Let's try to maximize size.
    # We'll run multiple iterations.
    
    candidates = list(range(1, 257))
    
    # To improve chances, we can sort candidates in a way that favors "safe" numbers.
    # Large numbers are generally safer.
    # Let's try a mix.
    
    max_attempts = 10000
    attempt = 0
    
    while time.time() < deadline and attempt < max_attempts:
        attempt += 1
        
        # Randomize the order of candidates
        # Bias: shuffle, but maybe keep larger numbers more likely to be picked early?
        # Actually, just shuffling is fine for diversity.
        random.shuffle(candidates)
        
        current_set = []
        
        for c in candidates:
            # Check if adding c keeps the set valid
            # Optimization: Only check if len(current_set) + 1 > best_size
            # But we need to build the set to know if it's valid.
            
            # Heuristic: If current_set is already larger than best_size, 
            # we can stop early if we can't add more? No, we need to check validity.
            
            # Try adding c
            # To speed up, we can do a quick check before full validity check?
            # Maybe check if c divides any existing sum or is divided by any?
            # That's not sufficient because sums of multiple elements matter.
            
            # Just do the full check.
            # But generating subset sums for every candidate is expensive.
            # We can maintain the set of sums incrementally.
            
            # Let's maintain current_sums
            if not hasattr(construct, '_cache'):
                pass # No cache, simple approach first
            
            # Incremental sum update
            if len(current_set) == 0:
                new_sums = {c}
            else:
                # We need the previous sums. 
                # This approach requires storing the sums for the current_set.
                # Let's refactor to maintain sums.
                pass
            
            # The incremental approach is better.
            # Let's restart the loop logic with incremental sums.
            break # Break to use the incremental version below
        
    # Incremental Version
    best_size = 0
    best_set = []
    
    # We will try to build sets of increasing size or just greedily build and see.
    # Greedy build:
    for _ in range(10000): # Iterations
        if time.time() > deadline:
            break
            
        random.shuffle(candidates)
        current_set = []
        current_sums = set() # Contains sums of current_set
        
        for c in candidates:
            # Check if we can add c
            # New sums will be current_sums U {x+c for x in current_sums} U {c}
            # Actually, new_sums = current_sums | {x+c for x in current_sums} | {c}
            # Wait, if current_sums is empty, new is {c}.
            
            # Check for duplicate sums
            # Check for divisibility
            
            # Optimization: 
            # 1. Check if c is in current_sums? No, c is a new element.
            # 2. Check if any new sum divides an existing sum or vice versa.
            
            # Let's generate the new sums set.
            new_sums_set = set(current_sums)
            # Add c
            new_sums_set.add(c)
            # Add x+c
            for x in current_sums:
                new_sums_set.add(x + c)
                
            # Check distinctness
            # Number of subsets in new set is 2 * (2^k - 1) + 1 = 2^(k+1) - 1
            # If len(new_sums_set) != 2^(len(current_set)+1) - 1, there are duplicates.
            expected_size = (1 << (len(current_set) + 1)) - 1
            if len(new_sums_set) != expected_size:
                continue
                
            # Check divisibility
            valid = True
            max_sum = sum(current_set) + c
            # We only need to check pairs where one is in new_sums_set.
            # Since current_sums was valid, we only need to check:
            # 1. Does c divide any existing sum?
            # 2. Does any existing sum divide c?
            # 3. Does c divide any new sum (x+c)?
            # 4. Does any new sum (x+c) divide c? (Impossible if x>0)
            # 5. Does any existing sum divide any new sum?
            # 6. Does any new sum divide any existing sum?
            
            # This is complex. Just checking all pairs in new_sums_set is safer and fast enough for small k.
            # But k can be up to ~15. 2^15 = 32k sums.
            # Pairwise check is 32k^2 = 10^9. Too slow.
            
            # Use the multiple check method.
            for s_val in new_sums_set:
                multiple = 2 * s_val
                while multiple <= max_sum:
                    if multiple in new_sums_set:
                        valid = False
                        break
                    multiple += s_val
                if not valid:
                    break
            
            if valid:
                current_set.append(c)
                current_sums = new_sums_set
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = list(current_set)
                    
                # If we reached a good size, we can stop this trial early?
                # Or continue to see if we can get larger.
                # Since candidates are shuffled, we might find a larger set.
                # But if we are stuck, we can break.
                # Let's just finish the iteration.
        
        # Local Search?
        # Try to swap elements to improve?
        # For now, rely on random restarts.
        
    return best_set

# EVOLVE-BLOCK-END
