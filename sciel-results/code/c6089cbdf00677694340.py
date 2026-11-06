import time
import sys
import random

def construct():
        start_time = time.time()
        deadline = start_time + 80.0 # Leave 10s buffer

        def is_time_up():
            return time.time() > deadline

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

        # 1. Hardcoded Baseline (Length 8)
        # This is the greedy "smallest possible" sequence
        base_seq = [1, 2, 5, 11, 23, 47, 95, 191]
        best_seq = base_seq
        best_len = len(base_seq)

        # 2. Try to extend or find longer sequences via Backtracking
        # We search for sequences of length L from best_len + 1 upwards.
        
        def dfs(seq, sums, last, target_len, depth):
            nonlocal best_seq, best_len, deadline
            
            if is_time_up():
                return False
            
            if len(seq) == target_len:
                # Found a valid sequence of target_len
                best_seq = seq[:]
                best_len = target_len
                return True

            # Pruning: Not enough numbers left
            remaining_needed = target_len - len(seq)
            if 256 - last < remaining_needed:
                return False

            # Heuristic: Try numbers in a specific order
            # Start from last+1 up to 256
            # To speed up, we might want to try "gaps" that are large to avoid collisions?
            # Or small to stay in range?
            # Let's try increasing order.
            
            # Optimization: Calculate new sums efficiently
            # new_sums are sums of blocks ending at x
            # x, x+last, x+last+prev...
            
            for x in range(last + 1, 257):
                # Quick check: x itself must not be in sums
                if x in sums:
                    continue
                
                # Calculate all new sums
                new_sums = [x]
                total = x
                for a in reversed(seq):
                    total += a
                    new_sums.append(total)
                    if total in sums:
                        break # Early exit if collision found
                else:
                    # No collision found in the loop
                    if not any(s in sums for s in new_sums):
                        # Update sums
                        old_len = len(sums)
                        sums.update(new_sums)
                        
                        # Recurse
                        if dfs(seq + [x], sums, x, target_len, depth + 1):
                            return True
                        
                        # Backtrack
                        # Since we added exactly len(new_sums) new items, and they were all unique and not in sums,
                        # we can remove them.
                        for s in new_sums:
                            sums.remove(s)

            return False

        # Try to find longer sequences
        # We know length 8 is possible.
        # Try 9, 10, 11, 12...
        
        # To avoid getting stuck in a deep search for L=12 that yields nothing,
        # we can set a time limit per target length or use a global deadline.
        # Global deadline is safer.
        
        # However, DFS for L=9 might take a while if the first few branches are bad.
        # We can add a "node count" limit or just rely on time.
        
        # Let's try L from best_len + 1 to 15
        for target_len in range(best_len + 1, 16):
            if is_time_up():
                break
            
            # Reset sums for new search
            initial_sums = set()
            # We can start from empty or from a prefix of a known good sequence?
            # Starting from empty is general but slow.
            # Starting from [1] is a good heuristic.
            
            # Let's try starting from [1] to reduce branching factor slightly
            # Actually, starting from empty allows finding sequences not starting with 1.
            # But 1 is almost always in optimal sequences.
            
            # Let's use a randomized restart strategy if the first attempt fails quickly?
            # No, let's just run the DFS.
            
            # To make DFS faster, we can use a heuristic ordering.
            # Try x values that are "far" from existing sums?
            
            # Let's just run the standard DFS.
            # If it takes too long for L=9, we might miss L=10.
            # But usually, if L=9 is possible, it's found quickly.
            
            found = dfs([], initial_sums, 0, target_len, 0)
            
            if found:
                # We found a sequence of this length.
                # We can continue to try for longer lengths.
                continue
            else:
                # If we didn't find it, and time is up, stop.
                # If we didn't find it and time is NOT up, it means we exhausted the search space?
                # Unlikely for L=9+ unless the search space is pruned heavily.
                # If the search space is exhausted, no solution exists for this length.
                # Since valid lengths are contiguous, if L is impossible, L+1 is impossible.
                # So we can break.
                # BUT, did we exhaust it or just timeout?
                # dfs returns False on timeout as well.
                # We need to distinguish.
                
                if is_time_up():
                    break
                else:
                    # Exhausted search space for this length.
                    break

        return best_seq
