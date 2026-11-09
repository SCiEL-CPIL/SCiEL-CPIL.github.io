import time
import random

def construct():
    """
    Finds a strictly increasing sequence of integers in [1, 256] 
    such that sums of all nonempty consecutive blocks are distinct.
    Maximizes sequence length.
    """
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    best_seq = []
    
    # Use a fixed seed for reproducibility in case the evaluator runs multiple times?
    # Or random? The task says "Python standard library... available".
    # Usually, deterministic is safer unless specified.
    # However, randomized search is often better for this.
    # I will use a fixed seed to ensure consistent behavior if needed, 
    # but try a few different seeds if time permits?
    # Let's stick to a single robust run with randomization.
    random.seed(42)
    
    def dfs(seq, last_val, prefix_sums, used_diffs):
        nonlocal best_seq, deadline
        
        if time.time() > deadline:
            return
        
        if len(seq) > len(best_seq):
            best_seq = seq.copy()
            
        # If we have used up the range, stop
        if last_val >= 256:
            return
            
        # Generate candidates
        # Heuristic: Try candidates in a random order to explore different regions
        candidates = list(range(last_val + 1, 257))
        random.shuffle(candidates)
        
        for val in candidates:
            if time.time() > deadline:
                return
            
            new_prefix = prefix_sums[-1] + val
            
            # Check Sidon condition: new differences must not exist in used_diffs
            new_diffs = []
            valid = True
            for i in range(len(prefix_sums)):
                d = new_prefix - prefix_sums[i]
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if valid:
                # Recurse
                seq.append(val)
                prefix_sums.append(new_prefix)
                used_diffs.update(new_diffs)
                
                dfs(seq, val, prefix_sums, used_diffs)
                
                # Backtrack
                seq.pop()
                prefix_sums.pop()
                used_diffs.difference_update(new_diffs)
                
                # If we found a very long sequence, we might stop early?
                # No, keep searching for longer.
                
    # Initial state:
    # seq: [1]
    # last_val: 1
    # prefix_sums: [0, 1]
    # used_diffs: {1}
    
    dfs([1], 1, [0, 1], {1})
    
    return best_seq
