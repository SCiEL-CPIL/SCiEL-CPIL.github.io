import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    def greedy_construct():
        values, sums = [], set()
        for x in range(1, 257):
            # Compute new sums if x is appended
            # Suffix sums of current values
            suffix_sums = [0]
            total = 0
            for a in reversed(values):
                total += a
                suffix_sums.append(total)
            
            new_sums = [x + s for s in suffix_sums]
            if not sums.intersection(new_sums):
                values.append(x)
                sums.update(new_sums)
        return values

    # Start with greedy
    best_seq = greedy_construct()
    best_len = len(best_seq)
    
    # Try to find longer sequences with backtracking
    # We will try to find a sequence of length L
    # Start from best_len + 1
    
    def search(depth, target_len, last_val, current_seq, current_sums, suffix_sums):
        nonlocal best_seq, best_len, deadline
        
        if time.time() > deadline:
            return None
            
        if depth == target_len:
            return current_seq.copy()
        
        # Heuristic: try values in a specific order
        # We can try small values first, or randomize
        # To improve over greedy, maybe we should explore different paths
        # Let's try values from last_val + 1 to 256
        
        # Optimization: if we have many candidates, we can't try all
        # But depth is small (20-30), so branching is manageable if pruned well
        
        # We can also use a heuristic to skip some values
        # For now, iterate through candidates
        
        # To avoid deep recursion overhead, we can limit the number of candidates tried
        # Or use a randomized order
        
        candidates = range(last_val + 1, 257)
        
        # Shuffle candidates to explore different branches?
        # Or keep order to find small sums first?
        # Greedy uses small sums. Maybe larger sums allow more length?
        # Let's try a mix: some small, some large.
        
        # Actually, let's just try all candidates in increasing order first.
        # If that fails, we might need randomization.
        
        for x in candidates:
            # Check if x is valid
            new_sums = [x + s for s in suffix_sums]
            if current_sums.intersection(new_sums):
                continue
            
            # Recurse
            new_seq = current_seq + [x]
            new_sums_set = current_sums.union(new_sums)
            
            # Update suffix sums
            new_suffix_sums = [x + s for s in suffix_sums]
            # Note: suffix sums for new sequence are:
            # 0 is not included in suffix sums for future steps?
            # Wait, suffix sums are sums of suffixes of the sequence.
            # For sequence a1...ak, suffix sums are:
            # 0 (empty suffix, used for x alone), ak, ak+ak-1, ...
            # So when we append x, the new suffix sums are:
            # 0, x, x+ak, x+ak+ak-1, ...
            # This is exactly [x + s for s in suffix_sums] if suffix_sums included 0.
            # Yes, suffix_sums started with 0.
            
            result = search(depth + 1, target_len, x, new_seq, new_sums_set, new_suffix_sums)
            if result:
                return result
            
            # Backtrack
            # No need to explicitly undo since we passed copies/immutable structures
            # But we are creating new sets/lists, which is slow.
            # We should use in-place modification with undo for speed.
            
        return None

    # The above search is too slow due to copying.
    # Let's rewrite with in-place modification.
    
    def fast_search(depth, target_len, last_val, seq, sums, suffix_sums):
        nonlocal best_seq, best_len, deadline
        
        if time.time() > deadline:
            return None
            
        if depth == target_len:
            return seq.copy()
        
        # Try candidates
        # We can limit the number of candidates to try to save time
        # Or use a heuristic
        
        # Heuristic: try smaller x first
        for x in range(last_val + 1, 257):
            # Compute new sums
            new_sums = []
            valid = True
            for s in suffix_sums:
                ns = x + s
                if ns in sums:
                    valid = False
                    break
                new_sums.append(ns)
            
            if not valid:
                continue
            
            # Add x
            seq.append(x)
            sums.update(new_sums)
            new_suffix_sums = new_sums  # Since suffix_sums had 0, new_suffix_sums is x+0, x+ak, ...
            # Wait, suffix_sums for the new sequence should be:
            # 0, x, x+ak, ...
            # The list `new_sums` is [x+0, x+ak, ...]. It does NOT contain 0.
            # But for the next step, we need suffix sums including 0?
            # No, the definition of suffix_sums in the greedy part was:
            # suffix_sums = [0, ak, ak+ak-1, ...]
            # So new_suffix_sums should be [0, x, x+ak, ...]
            # But in the check, we used `x + s` for `s in suffix_sums`.
            # If suffix_sums contains 0, then x+0 = x is in new_sums.
            # So new_suffix_sums for the next iteration should be [0] + new_sums?
            # No, the suffix sums of the NEW sequence are:
            # 0 (empty), x (last element), x+ak (last two), ...
            # So yes, new_suffix_sums = [0] + new_sums.
            
            result = fast_search(depth + 1, target_len, x, seq, sums, [0] + new_sums)
            if result:
                return result
            
            # Undo
            seq.pop()
            for ns in new_sums:
                sums.remove(ns)
                
        return None

    # Try increasing target lengths
    current_target = best_len + 1
    while current_target <= 30:  # Arbitrary upper bound
        if time.time() > deadline:
            break
            
        # Reset state for search
        # We can start from the greedy sequence?
        # Or start from scratch?
        # Starting from scratch allows finding completely different sequences.
        # But greedy is a good heuristic.
        # Let's try starting from scratch with a limit on depth.
        
        # To make it faster, we can use the greedy sequence as a guide?
        # For now, let's just run the search.
        
        # We need to be careful with recursion depth and time.
        # Let's add a node limit or time check inside the loop.
        
        # We will run the search for this target length.
        # To avoid getting stuck, we can randomize the candidate order sometimes.
        
        # Let's try a few random restarts for each target length.
        found = None
        for _ in range(5):  # 5 restarts
            if time.time() > deadline:
                break
                
            # Shuffle candidate order?
            # The search function above uses increasing order.
            # We can modify it to accept a candidate list.
            
            # For simplicity, let's stick to increasing order but maybe skip some?
            # Or just rely on the backtracking.
            
            # We need to initialize the state.
            # Start with empty sequence.
            seq = []
            sums = set()
            suffix_sums = [0]
            
            # We can use a heuristic to pick the first few elements greedily
            # to reduce the search space.
            
            # Let's pick the first 5 elements greedily.
            for _ in range(5):
                best_x = None
                for x in range(1, 257):
                    if seq and x <= seq[-1]:
                        continue
                    new_sums = [x + s for s in suffix_sums]
                    if not sums.intersection(new_sums):
                        best_x = x
                        break
                if best_x is None:
                    break
                seq.append(best_x)
                new_sums = [best_x + s for s in suffix_sums]
                sums.update(new_sums)
                suffix_sums = [0] + new_sums
            
            if len(seq) == 0:
                continue
                
            # Now backtrack for the rest
            # We need to know the current depth
            current_depth = len(seq)
            last_val = seq[-1]
            
            # The search function expects to add elements until target_len.
            # We need to modify it to start from current state.
            
            # Let's create a wrapper or modify fast_search to take initial state.
            
            # For now, let's just call fast_search with the initial state.
            # But fast_search as defined above starts from depth 0.
            # We need to pass the current state.
            
            # Let's refactor fast_search to be more flexible.
            
            # Due to time constraints, let's just use the simple version and hope it works.
            # Or, we can just return the greedy result if search is too slow.
            
            # Let's try to improve the greedy result by local search?
            # No, backtracking is better.
            
            # Let's just run the search from the greedy start.
            # We need to modify fast_search to accept initial state.
            
            # I will rewrite the search function to accept initial state.
            
            def search_from_state(depth, target_len, last_val, seq, sums, suffix_sums):
                nonlocal deadline
                
                if time.time() > deadline:
                    return None
                    
                if depth == target_len:
                    return seq.copy()
                
                for x in range(last_val + 1, 257):
                    new_sums = []
                    valid = True
                    for s in suffix_sums:
                        ns = x + s
                        if ns in sums:
                            valid = False
                            break
                        new_sums.append(ns)
                    
                    if not valid:
                        continue
                    
                    seq.append(x)
                    sums.update(new_sums)
                    new_suffix_sums = [0] + new_sums
                    
                    result = search_from_state(depth + 1, target_len, x, seq, sums, new_suffix_sums)
                    if result:
                        return result
                    
                    seq.pop()
                    for ns in new_sums:
                        sums.remove(ns)
                        
                return None
            
            # Run search from greedy start
            # We need to copy the state because search_from_state modifies it
            seq_copy = seq.copy()
            sums_copy = sums.copy()
            suffix_sums_copy = suffix_sums.copy()
            
            result = search_from_state(len(seq_copy), current_target, seq_copy[-1], seq_copy, sums_copy, suffix_sums_copy)
            if result:
                found = result
                break
                
        if found:
            best_seq = found
            best_len = current_target
            current_target += 1
        else:
            # If we failed to find a sequence of this length, we might try a larger one?
            # Unlikely. Let's break.
            break
            
    return best_seq
