import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Fallback
    best_set = [31, 32, 34]
    
    def is_valid_incremental(old_sums, x):
        """
        old_sums: set of valid subset sums of current set (excluding 0)
        x: new element
        Returns (new_sums_set, True) if valid, else (None, False)
        """
        # Generate new sums
        # s_new = s_old + x for s_old in old_sums
        # Plus x itself
        new_sums_list = [s + x for s in old_sums]
        new_sums_list.append(x)
        
        # Check for duplicates in new_sums_list
        new_sums_set = set(new_sums_list)
        if len(new_sums_set) != len(new_sums_list):
            return None, False
            
        # Check for duplicates with old_sums
        if new_sums_set & old_sums:
            return None, False
            
        # Check divisibility
        # We need to ensure no a, b in (old_sums U new_sums_set) with a != b and a|b.
        # Since old_sums is valid, we only check pairs involving at least one element from new_sums_set.
        
        max_possible_sum = 7680 # 30 * 256
        
        for s_new in new_sums_set:
            # Check multiples of s_new
            mult = 2 * s_new
            while mult <= max_possible_sum:
                if mult in old_sums or mult in new_sums_set:
                    return None, False
                mult += s_new
                
            # Check divisors of s_new
            # Find all divisors d < s_new
            # If d in old_sums or d in new_sums_set, return False
            # Note: 1 is a divisor.
            
            # Efficient divisor check:
            # Iterate i from 1 to int(sqrt(s_new))
            i = 1
            while i * i <= s_new:
                if s_new % i == 0:
                    d1 = i
                    d2 = s_new // i
                    
                    # Check d1
                    if d1 < s_new:
                        if d1 in old_sums or d1 in new_sums_set:
                            return None, False
                    else:
                        # d1 == s_new, skip (we don't check self)
                        pass
                        
                    # Check d2
                    if d2 < s_new:
                        if d2 in old_sums or d2 in new_sums_set:
                            return None, False
                    else:
                        pass
                        
                i += 1
                
        return new_sums_set, True

    def greedy_construct(order):
        current_set = []
        current_sums = set() # Empty set of sums
        
        for x in order:
            if time.time() > deadline:
                break
                
            new_sums, valid = is_valid_incremental(current_sums, x)
            if valid:
                current_set.append(x)
                current_sums = new_sums
                
        return current_set

    # Run multiple random greedy attempts
    # We can also try to sort the order to help greedy?
    # Random order is good for exploration.
    
    while time.time() < deadline:
        order = list(range(1, 257))
        random.shuffle(order)
        
        # Optional: Try a few different strategies
        # 1. Pure random
        res = greedy_construct(order)
        
        if len(res) > len(best_set):
            best_set = res
            
        # 2. Maybe try sorted?
        # Sorted might get stuck on small numbers early.
        # Random is better.
        
    return best_set
