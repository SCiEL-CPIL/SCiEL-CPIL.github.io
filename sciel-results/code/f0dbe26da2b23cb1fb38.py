import time
import sys

# Increase recursion limit just in case, though depth is small
sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    time_limit = 55.0  # Leave 5s buffer for the evaluator's overhead
    
    best_seq = []
    best_len = 0
    
    # State:
    # seq: list of chosen numbers
    # sums: set of all consecutive block sums so far
    # last_val: last number in seq
    # suffix_sums: list of sums of consecutive blocks ending at the last element of seq
    #             e.g., if seq=[2,3], suffix_sums=[3, 5]
    
    def dfs(seq, sums, last_val, suffix_sums):
        nonlocal best_seq, best_len
        
        # Check time
        if time.time() - start_time > time_limit:
            return
        
        # Update best
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq.copy()
            
        # Pruning: Check if we can possibly extend to length > best_len
        # Actually, we want to find the max length.
        # Let's try to extend as far as possible.
        
        # Pruning Bound:
        # If we are at depth k, can we reach length N?
        # We don't know N yet, but we know we want to beat best_len.
        # Let target_len = best_len + 1.
        # If we can't even reach best_len + 1, we can stop this branch.
        
        k = len(seq)
        if k == 0:
            # We can always start, so no prune here unless time is up.
            pass
        else:
            # Check if we can extend to at least best_len + 1
            target_len = best_len + 1
            if k < target_len:
                # Calculate min possible total sum for length target_len
                # Current sum S_k
                S_k = sum(seq) # This is O(k). We can maintain S_k as a parameter to make it O(1).
                # Let's optimize by passing S_k.
                
                # Min remaining sum for m = target_len - k elements
                m = target_len - k
                # Smallest m integers > last_val are last_val+1 ... last_val+m
                # Sum = m*last_val + m*(m+1)//2
                min_remaining_sum = m * last_val + m * (m + 1) // 2
                min_total_sum = S_k + min_remaining_sum
                
                # Condition: min_total_sum - a_1 + 1 >= target_len * (target_len + 1) // 2
                a_1 = seq[0]
                required_range = target_len * (target_len + 1) // 2
                if min_total_sum - a_1 + 1 < required_range:
                    return

        for x in range(last_val + 1, 257):
            # Calculate new suffix sums
            new_suffix_sums = [x]
            for s in suffix_sums:
                new_suffix_sums.append(x + s)
                
            # Check for conflicts
            conflict = False
            for ns in new_suffix_sums:
                if ns in sums:
                    conflict = True
                    break
            
            if conflict:
                continue
                
            new_sums = sums.union(new_suffix_sums)
            # Recurse
            dfs(seq + [x], new_sums, x, new_suffix_sums)
            
            # If we found a solution with length >= some target, we could stop?
            # No, we want to maximize.

    # Initial call
    dfs([], set(), 0, [])
    
    return best_seq
