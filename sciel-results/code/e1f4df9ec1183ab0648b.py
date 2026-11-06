import time
import sys

def construct():
    deadline = time.time() + 85  # Safety margin
    
    def greedy_baseline():
        """Find a quick baseline sequence using greedy strategy."""
        seq = []
        sums = set()
        for x in range(1, 257):
            # Check if x can be added
            # New sums are x + suffix_sums of current seq
            new_sums = [x]
            total = x
            for a in reversed(seq):
                total += a
                new_sums.append(total)
            
            conflict = False
            for s in new_sums:
                if s in sums:
                    conflict = True
                    break
            
            if not conflict:
                seq.append(x)
                sums.update(new_sums)
        return seq

    def dfs(seq, sums, last, target_len):
        """
        Depth First Search to find a sequence of exactly target_len.
        Returns the sequence if found, None otherwise.
        """
        if time.time() > deadline:
            return None
        
        current_len = len(seq)
        if current_len == target_len:
            return list(seq)
        
        # Pruning: Not enough integers left
        if last + (target_len - current_len) > 256:
            return None
            
        # Pruning: Heuristic - if the gap is too small?
        # We need target_len - current_len more elements, all > last.
        # Smallest possible next elements are last+1, last+2...
        # If even the smallest possible choices lead to conflict, we might prune.
        # But checking that is expensive.
        
        # Iterate candidates
        # Smaller x is better (leaves more room)
        for x in range(last + 1, 257):
            if time.time() > deadline:
                return None
            
            # Compute new suffix sums
            # s_k = x + sum(seq[-k:])
            new_suffixes = [x]
            total = x
            # Optimization: Precompute suffix sums of current seq?
            # Since we are in a loop, doing this O(N) for each x is O(N^2) per level.
            # N is small (< 20), so it's fine.
            for a in reversed(seq):
                total += a
                new_suffixes.append(total)
            
            # Check conflicts
            conflict = False
            for s in new_suffixes:
                if s in sums:
                    conflict = True
                    break
            
            if conflict:
                continue
            
            # Recurse
            new_sums = sums | set(new_suffixes)
            seq.append(x)
            result = dfs(seq, new_sums, x, target_len)
            seq.pop()
            
            if result:
                return result
        
        return None

    # Get baseline
    baseline = greedy_baseline()
    base_len = len(baseline)
    
    best_seq = baseline
    best_len = base_len
    
    # Try to find longer sequences
    # Start from a reasonable upper bound.
    # Max possible length? 
    # Number of sums N = L(L+1)/2.
    # Max sum approx 256*L.
    # L=20 -> 210 sums. Range 5120. Feasible.
    # L=25 -> 325 sums. Range 6400. Feasible.
    # Let's try up to 25.
    
    # We will iterate target lengths downwards.
    # We give each target a time budget.
    # If we find one, we return it immediately as it's the longest possible (since we go down).
    # Wait, if we go 25, 24, 23... and 25 fails, 24 might succeed.
    # We want the MAX length. So yes, start high.
    
    # However, if 25 is very hard, we might waste time.
    # Let's estimate max possible.
    # The greedy baseline is a good lower bound.
    # Let's start from base_len + 5 and go up? No, we want max.
    # Let's start from 25 down to base_len.
    
    # To avoid wasting too much time on impossible high lengths,
    # we can check a simple bound.
    # But let's just run with time limits.
    
    # Time management:
    # Total time ~ 85s.
    # If we try 10 lengths, we get ~8s each.
    
    for target_len in range(25, base_len - 1, -1):
        if time.time() > deadline - 2:
            break
            
        # Set a sub-deadline for this specific search
        # We want to spend more time on higher lengths?
        # Or just let the global deadline handle it.
        # If we use global deadline, and 25 takes 80s, we don't try 24.
        # That's bad if 25 is impossible and 24 is easy.
        
        # So we need a time limit per target.
        # Let's allocate time based on remaining budget.
        remaining_targets = 25 - target_len + 1
        time_per_target = (deadline - time.time()) / remaining_targets
        # Minimum 1 second per target to avoid infinite loops on easy ones?
        # No, if it's easy, it finishes fast.
        
        # We need a way to stop DFS early.
        # We can pass a local deadline.
        
        local_deadline = time.time() + time_per_target
        
        # Modify dfs to accept a deadline?
        # Or just use the global one and break?
        # The current dfs uses global `deadline`.
        # If I change `deadline` inside the loop, it affects the function.
        # Since `deadline` is a closure variable, I can update it.
        
        old_deadline = deadline
        deadline = local_deadline
        
        result = dfs([], set(), 0, target_len)
        
        deadline = old_deadline # Restore
        
        if result:
            best_seq = result
            best_len = target_len
            break # Found the longest, stop searching
            
    return best_seq
