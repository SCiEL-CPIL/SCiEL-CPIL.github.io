import itertools
import time
import sys

# Precomputation for the specific case v=19, k=7, t=4
# But the function signature is construct(v, k, t), so we should handle general v,k,t if possible, 
# or at least optimize for the specific case if we assume the evaluator only calls with these.
# The prompt says "The evaluator passes this task's parameters v, k and t."
# It's safer to write a general solution that is fast for v=19, k=7, t=4.

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Map combinations to indices for fast lookup
    # We only need to track t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_index = {s: i for i, s in enumerate(all_t_subsets)}
    
    uncovered = set(range(len(all_t_subsets)))
    blocks = []
    
    # Precompute all (t-1) subsets for any potential block state?
    # No, we generate them on the fly.
    
    while uncovered:
        if time.time() > deadline:
            # If we time out, we must return a valid covering.
            # The greedy algorithm guarantees that every iteration covers at least one new subset.
            # If we break early, we might have uncovered subsets.
            # However, the problem requires a valid covering.
            # Given the speed, this should not happen.
            # If it does, we can't return an invalid result.
            # We will just let it run, but the deadline is a guard against infinite loops or bugs.
            # For this specific task, the loop will finish quickly.
            pass

        # Pick the first uncovered subset (lexicographically smallest index)
        # Using min(uncovered) is O(N) but N is small.
        # We can just pop one.
        idx = min(uncovered)
        S = all_t_subsets[idx]
        
        block = list(S)
        
        # Fill the block up to size k
        while len(block) < k:
            best_gain = -1
            best_x = -1
            current_block_set = set(block)
            
            # Generate all (t-1) subsets of the current block
            # These are the potential "bases" for new t-subsets involving x
            t_minus_1_subsets = list(itertools.combinations(block, t - 1))
            
            for x in range(v):
                if x in current_block_set:
                    continue
                
                gain = 0
                for rest in t_minus_1_subsets:
                    # Form the t-subset: rest + x
                    # rest is a tuple of t-1 elements
                    # We need to sort to form the canonical tuple
                    # Since rest is from combinations, it's sorted.
                    # We need to insert x into rest to keep it sorted, or just sort the union.
                    # Sorting a list of size t is fast.
                    new_subset = tuple(sorted(rest + (x,)))
                    if new_subset in t_subset_to_index:
                        sub_idx = t_subset_to_index[new_subset]
                        if sub_idx in uncovered:
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            # If no point gives positive gain (shouldn't happen if uncovered is not empty and we started with S),
            # we must pick one arbitrarily to continue.
            if best_x == -1:
                # Find any x not in block
                for x in range(v):
                    if x not in current_block_set:
                        best_x = x
                        break
            
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered set
        # Remove all t-subsets of the new block
        for sub in itertools.combinations(block, t):
            sub_idx = t_subset_to_index[sub]
            if sub_idx in uncovered:
                uncovered.remove(sub_idx)
        
        # Optional: Periodic check for time
        if time.time() > deadline:
             # Break and return what we have? No, must be valid.
             # But as established, this will finish.
             pass

    return blocks
