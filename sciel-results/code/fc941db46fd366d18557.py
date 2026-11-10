import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    def is_valid_extension(seq, used_sums, x):
        """
        Check if x can be appended to seq.
        seq is a list of integers.
        used_sums is a set of integers representing all current consecutive block sums.
        x is the new integer to append.
        Returns the new set of sums if valid, None otherwise.
        """
        n = len(seq)
        # New sums ending at x
        # Sum of [x]
        # Sum of [seq[-1], x]
        # ...
        # Sum of [seq[0], ..., seq[-1], x]
        
        # We can compute these efficiently using prefix sums or just accumulating
        # But we need to check against used_sums.
        
        # Let's just compute the new sums
        new_sums = set()
        current_sum = x
        new_sums.add(current_sum)
        if current_sum in used_sums:
            return None
        for i in range(n - 1, -1, -1):
            current_sum += seq[i]
            new_sums.add(current_sum)
            if current_sum in used_sums:
                return None
        
        # Also need to check if any new sum collides with another new sum?
        # No, the new sums are all distinct because they are suffix sums of the new sequence ending at x.
        # Wait, are they?
        # Sums are: x, a_n+x, a_{n-1}+a_n+x, ...
        # Since a_i > 0, these are strictly increasing. So they are distinct from each other.
        # So we only need to check against used_sums.
        
        return used_sums | new_sums

    def backtrack(seq, used_sums, last_val, depth):
        nonlocal best_seq, best_len, deadline
        
        if time.time() > deadline:
            return
        
        if len(seq) > best_len:
            best_seq = seq[:]
            best_len = len(seq)
            
        # If we've reached max possible length (256), stop
        if len(seq) == 256:
            return
            
        # Pruning: if we can't beat best_len, stop
        # Max possible additional elements is 256 - last_val
        if len(seq) + (256 - last_val) <= best_len:
            return
            
        # Try adding next elements
        # To improve search, we can try a mix of small and large numbers
        # But strictly increasing, so x > last_val
        
        # Heuristic: try values that are "spread out"
        # Or just iterate
        for x in range(last_val + 1, 257):
            if time.time() > deadline:
                return
            
            new_sums = is_valid_extension(seq, used_sums, x)
            if new_sums is not None:
                seq.append(x)
                backtrack(seq, new_sums, x, depth + 1)
                seq.pop()

    best_seq = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    best_len = len(best_seq)
    
    # Start backtracking
    # We can start from an empty sequence or a small prefix
    # Starting from empty allows finding the optimal start
    # But it's a large search space.
    # Let's start with a few good prefixes
    
    # Prefix 1: [1]
    used_init = {1}
    backtrack([1], used_init, 1, 0)
    
    # Prefix 2: [2]
    if time.time() < deadline - 5:
        used_init = {2}
        backtrack([2], used_init, 2, 0)
        
    # Prefix 3: [1, 2]
    if time.time() < deadline - 5:
        used_init = {1, 2, 3}
        backtrack([1, 2], used_init, 2, 0)

    return best_seq
