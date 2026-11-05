import time

def construct():
    """
    Finds a strictly increasing sequence of integers in [1, 256] such that
    the sums of all nonempty consecutive blocks are distinct.
    Maximizes the length of the sequence.
    
    This is equivalent to finding a Golomb Ruler with strictly increasing marks
    (differences between consecutive marks), where each mark difference is <= 256.
    """
    start_time = time.time()
    # 85 seconds limit to be safe within the 90s wall-clock limit
    deadline = start_time + 85.0
    
    best_seq = []
    best_len = 0
    
    def dfs(seq, prefix, diffs):
        nonlocal best_seq, best_len
        
        # Update best if we found a longer sequence
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
        
        # Check time
        if time.time() > deadline:
            return
        
        last_a = seq[-1] if seq else 0
        last_p = prefix[-1]
        
        # Pruning: 
        # The maximum number of additional elements we can add is bounded by 
        # the number of integers in (last_a, 256].
        # If current length + max possible additions <= best_len, we can prune.
        max_additional = 256 - last_a
        if len(seq) + max_additional <= best_len:
            return

        # Try all possible next values for a
        for a in range(last_a + 1, 257):
            new_p = last_p + a
            
            # Calculate new differences: new_p - p for all p in prefix
            # Check if any of these differences already exist in 'diffs'
            valid = True
            new_diffs = []
            for p in prefix:
                d = new_p - p
                if d in diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
            
            # If valid, add new differences to the set
            for d in new_diffs:
                diffs.add(d)
            
            # Recurse
            dfs(seq + [a], prefix + [new_p], diffs)
            
            # Backtrack: remove new differences
            for d in new_diffs:
                diffs.remove(d)
            
            # Optional: If we just extended to a new best length, 
            # we might want to prioritize smaller 'a' to allow more steps?
            # The loop already tries smaller 'a' first.
            # We continue to try larger 'a' in case they lead to longer sequences 
            # (e.g., if small 'a' leads to a dead end quickly, but large 'a' 
            # leaves room for more steps? Unlikely for Golomb rulers, but possible).
            # However, smaller 'a' generally allows for more elements.
            # We can break if we found a solution of length best_len and 
            # we are sure we can't beat it? 
            # The pruning above handles the "can't beat it" case based on count.
            # But structurally, a different path might allow more steps.
            # So we must explore all.

    # Initial state: empty sequence, prefix sum 0, no differences
    dfs([], [0], set())
    
    return best_seq
