import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 35s buffer for safety

    def is_time_up():
        return time.time() > deadline

    # Hardcoded baseline: length 8 sequence
    base_seq = [1, 2, 5, 11, 23, 47, 95, 191]
    best_seq = base_seq[:]
    best_len = len(base_seq)

    def check_valid(seq):
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    # Verify baseline
    if not check_valid(base_seq):
        # Fallback to a simple valid sequence
        best_seq = [1, 2, 4, 8, 16, 32, 64, 128]
        best_len = len(best_seq)

    # Forward-checking backtracking solver
    # State:
    #   seq: list of chosen numbers
    #   prefix_sums: list of prefix sums, prefix_sums[i] = sum(seq[:i+1])
    #   forbidden_sums: set of all sums of consecutive blocks in seq
    #     These are all differences prefix_sums[j] - prefix_sums[i] for j > i
    #   last: last chosen number
    
    # When adding x to seq:
    #   new_prefix = prefix_sums[-1] + x (if seq non-empty) or x
    #   new sums are: new_prefix - prefix_sums[i] for i in range(len(prefix_sums))
    #   i.e., x, x + seq[-1], x + seq[-1] + seq[-2], ...
    #   Also need x > last for strictly increasing

    def dfs(seq, prefix_sums, forbidden_sums, last, target_len):
        nonlocal best_seq, best_len

        if is_time_up():
            return False

        if len(seq) == target_len:
            if len(seq) > best_len:
                best_seq = seq[:]
                best_len = target_len
            return True

        n = len(seq)
        
        # Generate candidates: last+1 to 256
        # Heuristic: try smaller values first to stay in range longer
        for x in range(last + 1, 257):
            if is_time_up():
                return False

            # Compute new prefix sum
            new_prefix = prefix_sums[-1] + x if prefix_sums else x

            # Check all new block sums ending at x
            # These are: new_prefix - prefix_sums[i] for i in range(len(prefix_sums))
            # Plus x itself if we consider the block [x]
            # Actually, the blocks ending at x are:
            #   [x] -> sum = x
            #   [seq[-1], x] -> sum = seq[-1] + x
            #   [seq[-2], seq[-1], x] -> sum = seq[-2] + seq[-1] + x
            #   ...
            #   [seq[0], ..., x] -> sum = new_prefix
            # 
            # In terms of prefix sums:
            #   Block from index i to n-1 (0-indexed in new seq) has sum = new_prefix - prefix_sums[i-1]
            #   where prefix_sums is the OLD prefix sums (before adding x)
            #   For the block starting at position 0 (the whole new seq), sum = new_prefix
            #   For block starting at position k (1 <= k <= n), sum = new_prefix - prefix_sums[k-1]
            #   For block starting at position n (just x), sum = x = new_prefix - 0

            # So new sums are: new_prefix - p for p in [0] + prefix_sums
            # i.e., new_prefix, new_prefix - prefix_sums[0], new_prefix - prefix_sums[1], ...

            valid = True
            new_sums = []
            
            # Block consisting of just x
            if x in forbidden_sums:
                valid = False
            else:
                new_sums.append(x)
                
            # Blocks ending at x that include previous elements
            for p in prefix_sums:
                s = new_prefix - p
                if s in forbidden_sums or s in new_sums:
                    valid = False
                    break
                new_sums.append(s)

            if not valid:
                continue

            # All new sums are valid. Recurse.
            # We need to add new_sums to forbidden_sums
            old_forbidden_size = len(forbidden_sums)
            forbidden_sums.update(new_sums)
            
            new_prefix_sums = prefix_sums + [new_prefix]
            
            if dfs(seq + [x], new_prefix_sums, forbidden_sums, x, target_len):
                return True
            
            # Backtrack
            forbidden_sums.difference_update(new_sums)

        return False

    # Try to find longer sequences
    # Start from target_len = best_len + 1
    # Use a time-limited search for each target length
    
    for target_len in range(best_len + 1, 20):
        if is_time_up():
            break
        
        # Reset state for this search
        # Start with empty sequence
        initial_prefix_sums = []
        initial_forbidden_sums = set()
        initial_last = 0
        
        found = dfs([], initial_prefix_sums, initial_forbidden_sums, initial_last, target_len)
        
        if found:
            # Found a valid sequence of this length
            # Continue to try for longer lengths
            continue
        else:
            if is_time_up():
                break
            else:
                # Search exhausted for this length (unlikely but possible with pruning)
                # If we truly exhausted, then no longer sequence exists
                # But we might have hit a timeout in the middle
                # Since we can't easily distinguish, and if target_len is small,
                # it's more likely we exhausted. For larger target_len, timeout is more likely.
                # Given the nature of the problem, if length L is impossible, L+1 is too.
                # So break.
                break

    return best_seq
