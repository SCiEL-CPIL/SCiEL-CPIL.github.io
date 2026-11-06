import sys

def construct():
    # Step 1: Greedy baseline
    values_greedy = []
    sums_greedy = set()
    for x in range(1, 257):
        tails = []
        total = 0
        # Calculate sums ending at x
        # x
        # x + last
        # x + last + second_last
        # ...
        # x + sum(all)
        current_sum = 0
        new_sums = []
        for a in reversed([x] + values_greedy):
            current_sum += a
            new_sums.append(current_sum)
        
        if not sums_greedy.intersection(new_sums):
            values_greedy.append(x)
            sums_greedy.update(new_sums)
    
    best_len = len(values_greedy)
    best_seq = values_greedy
    
    # Step 2: Backtracking to see if we can do better
    # We try to find a sequence longer than best_len
    # DFS state: current_seq, current_sums
    # We can prune if len(current_seq) + (256 - current_seq[-1]) <= best_len is not a good bound because we can skip numbers.
    # However, the number of elements is bounded by 256.
    # Let's use a simple DFS with a depth limit of best_len + 1 initially, then increase.
    # Or just DFS to the end, keeping track of the best found.
    
    # To make DFS efficient, we order candidates.
    # Trying smaller candidates first is generally good for packing.
    
    def dfs(current_seq, current_sums):
        nonlocal best_len, best_seq
        
        if len(current_seq) > best_len:
            best_len = len(current_seq)
            best_seq = current_seq[:]
            
        if len(current_seq) == 256:
            return
            
        last_val = current_seq[-1] if current_seq else 0
        
        # Optimization: If we have already found a sequence of length L, 
        # and we are at depth D, we can only improve if D + (256 - last_val) > best_len?
        # Not strictly, because we can skip values. But if the remaining values are few, it's a hint.
        
        for x in range(last_val + 1, 257):
            # Calculate new sums ending at x
            new_sums = []
            current_sum = 0
            # Append x to end
            # Sums: x, x+last, x+last+prev, ...
            for a in reversed([x] + current_seq):
                current_sum += a
                new_sums.append(current_sum)
                
            # Check collision
            collision = False
            for s in new_sums:
                if s in current_sums:
                    collision = True
                    break
            
            if not collision:
                # Recurse
                new_sums_set = set(new_sums)
                dfs(current_seq + [x], current_sums.union(new_sums_set))
                
                # Early termination if we found a very long sequence? 
                # Unlikely to exceed 20-30 given N=256.
                
    # Start DFS
    # We can start with empty sequence, or use the greedy sequence as a starting point?
    # Starting from empty is safer to ensure we don't miss a non-greedy prefix that leads to a longer tail.
    # However, the search space is $2^{256}$, which is huge.
    # We need strong pruning.
    
    # Pruning heuristic:
    # The number of subarray sums for a sequence of length k is k(k+1)/2.
    # The maximum sum is approx sum(seq).
    # If the number of used sums is close to the maximum possible distinct sums in the range [1, max_sum], we are stuck.
    
    # Let's refine the DFS with a better pruning:
    # If len(current_seq) + (256 - last_val) <= best_len, we can't beat the best even if we pick all remaining numbers.
    # This is a valid bound because we must pick strictly increasing integers.
    
    def dfs_optimized(current_seq, current_sums):
        nonlocal best_len, best_seq
        
        if len(current_seq) > best_len:
            best_len = len(current_seq)
            best_seq = current_seq[:]
            
        if len(current_seq) == 256:
            return
            
        last_val = current_seq[-1] if current_seq else 0
        
        # Pruning: Max possible length from here
        max_possible_additional = 256 - last_val
        if len(current_seq) + max_possible_additional <= best_len:
            return

        for x in range(last_val + 1, 257):
            new_sums = []
            current_sum = 0
            for a in reversed([x] + current_seq):
                current_sum += a
                new_sums.append(current_sum)
                
            collision = False
            for s in new_sums:
                if s in current_sums:
                    collision = True
                    break
            
            if not collision:
                dfs_optimized(current_seq + [x], current_sums.union(set(new_sums)))
                
    dfs_optimized([], set())
    
    return best_seq
