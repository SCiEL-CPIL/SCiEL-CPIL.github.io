import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s buffer for the evaluator's 60s/90s limit
    
    # 1. Greedy Baseline
    # This provides a good initial best_len and best_seq for pruning.
    values, sums = [], set()
    for x in range(1, 257):
        tails = []
        total = x
        tails.append(total)
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    
    best_seq = values[:]
    best_len = len(values)
    
    # If greedy already found a very long sequence, we might not need much search.
    # But let's try to improve.
    
    # 2. DFS to maximize length
    # We use a bitset for sums to make conflict checking O(1) per sum.
    # Max sum is 256 * 256 = 65536.
    
    def dfs(vals, sum_mask, last_val, current_len):
        nonlocal best_seq, best_len, deadline
        
        # Time check
        if time.time() > deadline:
            return False
        
        # Update best if we found a longer sequence
        if current_len > best_len:
            best_len = current_len
            best_seq = vals[:]
            
        # Pruning: Can we possibly beat best_len?
        # Max additional elements we can add is (256 - last_val).
        if current_len + (256 - last_val) <= best_len:
            return True # No need to explore further in this branch
        
        # Try next values
        # Heuristic: Try smaller values first? Or larger?
        # Smaller values leave more room for larger values later, but create smaller sums.
        # Larger values create larger sums, which might be less likely to conflict with existing small sums.
        # However, we are strictly increasing.
        # Let's iterate from last_val + 1 to 256.
        
        for x in range(last_val + 1, 257):
            # Check if x can be appended
            # New sums are x, x+vals[-1], x+vals[-1]+vals[-2], ...
            
            # Check x
            if (sum_mask >> x) & 1:
                continue
                
            # Check other suffix sums
            valid = True
            t = x
            # Iterate through existing values in reverse to build suffix sums
            # vals is [v1, v2, ..., vk]
            # Suffixes ending at vk: vk, vk+v_{k-1}, ...
            # We need to check x + suffix
            for v in reversed(vals):
                t += v
                if (sum_mask >> t) & 1:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Valid candidate x
            # Update mask
            new_mask = sum_mask | (1 << x)
            t = x
            for v in reversed(vals):
                t += v
                new_mask |= (1 << t)
                
            # Recurse
            # To avoid copying the list every time, we can append/pop
            vals.append(x)
            dfs(vals, new_mask, x, current_len + 1)
            vals.pop()
            
            # If we found a solution that is as long as the theoretical max for this branch?
            # Theoretical max for this branch is (256 - last_val) + current_len
            # If best_len >= current_len + (256 - last_val), we can break?
            # Yes, if we've already achieved the maximum possible length for this branch,
            # no other x in this branch can yield a longer sequence.
            if best_len >= current_len + (256 - last_val):
                return True

        return True

    # Start DFS
    # We can start from the greedy solution to save time?
    # Or start from scratch?
    # Starting from scratch explores more.
    # However, the greedy solution is a good lower bound.
    # Let's start from an empty sequence.
    
    # To speed up, we can try to start the DFS with the first few elements of the greedy solution?
    # No, greedy might be suboptimal in early choices.
    # But 256 is small.
    
    # Optimization: Order of x.
    # If we pick a small x, we have more options for x_next.
    # If we pick a large x, we have fewer options.
    # Usually, "left-deep" search (small x first) finds long sequences faster?
    # Or does it get stuck in local minima?
    # Given the pruning `current_len + (256 - last_val) <= best_len`,
    # if we pick a large x early, `256 - last_val` becomes small, pruning heavily.
    # So small x first is better for finding *long* sequences.
    
    dfs([], 0, 0, 0)
    
    return best_seq

construct()
