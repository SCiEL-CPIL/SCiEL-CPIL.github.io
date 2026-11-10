import time
import sys

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for the 60s limit
    
    # Initialize with a known valid sequence (powers of 2) to guarantee a baseline score
    # 1, 2, 4, 8, 16, 32, 64, 128, 256 is valid and has length 9.
    best_seq = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    # We will use a DFS to try to find a longer sequence.
    # To speed it up, we'll use in-place modifications and nonlocal variables.
    
    current_seq = []
    current_sums = set()
    suffix_sums = []
    
    def dfs_inplace():
        nonlocal best_seq
        nonlocal start_time
        nonlocal time_limit
        nonlocal suffix_sums
        
        if time.time() - start_time > time_limit:
            return
        
        # Update best if current is longer
        if len(current_seq) > len(best_seq):
            best_seq = current_seq[:]
            
        last_val = current_seq[-1] if current_seq else 0
        
        if last_val >= 256:
            return
            
        # Pruning: Even if we add the smallest possible next numbers (last_val+1, last_val+2...),
        # can we beat the current best length?
        # Max possible length extension is limited by the range.
        # A loose bound: if len(current_seq) + (256 - last_val) <= len(best_seq), we might still prune?
        # Actually, we can add at most 1 element per integer value.
        # So max additional elements is 256 - last_val.
        if len(current_seq) + (256 - last_val) <= len(best_seq):
            return

        for x in range(last_val + 1, 257):
            # Calculate new sums
            # x
            # x + suffix_sums[0]
            # ...
            
            # Check x
            if x in current_sums:
                continue
                
            # Check x + s
            valid = True
            new_sums = [x]
            for s in suffix_sums:
                ns = x + s
                new_sums.append(ns)
                if ns in current_sums:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # All good, add x
            current_seq.append(x)
            for ns in new_sums:
                current_sums.add(ns)
            # Update suffix_sums
            # New suffix sums are [x] + [x+s for s in suffix_sums]
            old_suffix_sums = suffix_sums[:]
            suffix_sums = [x] + [x + s for s in suffix_sums]
            
            dfs_inplace()
            
            # Undo
            current_seq.pop()
            for ns in new_sums:
                current_sums.remove(ns)
            suffix_sums = old_suffix_sums

    dfs_inplace()
    
    return best_seq
