import time
import sys
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    TIMEOUT = 55.0  # Safety margin for 60s limit
    
    def get_all_sums(seq):
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
        return sums

    def can_append(seq, x, sums_set, suffix_sums):
        # Check if appending x to seq creates duplicate consecutive sums
        # New sums are x, x+seq[-1], x+seq[-1]+seq[-2], ...
        # These are x + s for s in [0] + suffix_sums
        # Since seq is increasing positive, suffix_sums are strictly decreasing
        # so new sums are strictly increasing - no internal duplicates
        new_sums = [x + s for s in [0] + suffix_sums]
        return not any(s in sums_set for s in new_sums)

    def get_new_sums(seq, x, suffix_sums):
        return [x + s for s in [0] + suffix_sums]

    def greedy_build(prefer_large=False):
        vals = []
        sums_set = set()
        suffix_sums = []
        
        candidates = range(1, 257)
        if prefer_large:
            candidates = range(256, 0, -1)
        
        for x in candidates:
            if vals and x <= vals[-1]:
                continue
            
            new_sums = get_new_sums(vals, x, suffix_sums)
            
            if not any(s in sums_set for s in new_sums):
                vals.append(x)
                sums_set.update(new_sums)
                suffix_sums = new_sums
        
        return vals

    # 1. Greedy initialization - try both ascending and descending
    best_seq = greedy_build(prefer_large=False)
    best_len = len(best_seq)
    
    alt_seq = greedy_build(prefer_large=True)
    if len(alt_seq) > best_len:
        best_seq = alt_seq
        best_len = len(alt_seq)
    
    # 2. Randomized local search to improve
    # Strategy: try to insert elements at various positions, or replace elements
    def try_improve(seq, sums_set, suffix_sums):
        # Try to extend the sequence by finding valid elements to append
        # Also try to find better elements by local search
        
        n = len(seq)
        if n == 0:
            return seq, sums_set, suffix_sums
        
        # First, try to extend greedily from the current end
        last_val = seq[-1]
        for x in range(last_val + 1, 257):
            if can_append(seq, x, sums_set, suffix_sums):
                new_sums = get_new_sums(seq, x, suffix_sums)
                seq.append(x)
                sums_set.update(new_sums)
                suffix_sums = new_sums
                break
        
        return seq, sums_set, suffix_sums

    def local_search(seq, max_iterations=1000):
        # Try to improve by replacing elements or inserting new ones
        # This is a simplified local search
        
        current_seq = list(seq)
        current_sums = get_all_sums(current_seq)
        current_suffix = []
        total = 0
        for a in reversed(current_seq):
            total += a
            current_suffix.append(total)
        current_suffix.reverse()
        
        best_local = current_seq
        best_local_len = len(current_seq)
        
        for iteration in range(max_iterations):
            if time.time() - start_time > TIMEOUT - 5:
                break
            
            improved = False
            n = len(current_seq)
            
            # Try to insert a new element at a random position
            if n < 256:
                # Find a valid position to insert
                pos = random.randint(0, n)
                min_val = current_seq[pos - 1] + 1 if pos > 0 else 1
                max_val = current_seq[pos] - 1 if pos < n else 256
                
                if min_val <= max_val:
                    # Try a random value in range
                    x = random.randint(min_val, max_val)
                    
                    # Check if inserting x at pos is valid
                    # We need to compute new sums
                    new_seq = current_seq[:pos] + [x] + current_seq[pos:]
                    
                    # Quick check: verify all consecutive sums are distinct
                    new_sums = get_all_sums(new_seq)
                    if len(new_sums) == len(new_seq) * (len(new_seq) + 1) // 2:
                        current_seq = new_seq
                        current_sums = new_sums
                        # Recompute suffix sums
                        current_suffix = []
                        total = 0
                        for a in reversed(current_seq):
                            total += a
                            current_suffix.append(total)
                        current_suffix.reverse()
                        
                        if len(current_seq) > best_local_len:
                            best_local = list(current_seq)
                            best_local_len = len(current_seq)
                        improved = True
            
            # Try to replace an element with a larger one
            if not improved and n > 0:
                pos = random.randint(0, n - 1)
                old_val = current_seq[pos]
                
                # Try replacing with a larger value
                min_new = old_val + 1
                max_new = current_seq[pos + 1] - 1 if pos + 1 < n else 256
                
                if min_new <= max_new:
                    x = random.randint(min_new, max_new)
                    new_seq = current_seq.copy()
                    new_seq[pos] = x
                    
                    # Check validity
                    new_sums = get_all_sums(new_seq)
                    if len(new_sums) == len(new_seq) * (len(new_seq) + 1) // 2:
                        current_seq = new_seq
                        current_sums = new_sums
                        current_suffix = []
                        total = 0
                        for a in reversed(current_seq):
                            total += a
                            current_suffix.append(total)
                        current_suffix.reverse()
                        improved = True
        
        return best_local

    # Run local search multiple times with different random seeds
    random.seed(42)
    for attempt in range(10):
        if time.time() - start_time > TIMEOUT - 10:
            break
        
        # Start from best_seq and try to improve
        seq_copy = list(best_seq)
        improved = local_search(seq_copy, max_iterations=500)
        
        if len(improved) > best_len:
            best_seq = improved
            best_len = len(best_seq)
    
    # 3. DFS from the best found so far
    # Use DFS to try to extend beyond current best
    def dfs_extend(seq, sums_set, suffix_sums, depth):
        nonlocal best_seq, best_len, start_time, TIMEOUT
        
        if time.time() - start_time > TIMEOUT:
            return False
        
        if len(seq) > best_len:
            best_seq = list(seq)
            best_len = len(seq)
        
        if seq:
            max_possible = len(seq) + (256 - seq[-1])
        else:
            max_possible = 256
            
        if max_possible <= best_len:
            return False

        last_val = seq[-1] if seq else 0
        
        # Try next values, but limit to reasonable range to avoid explosion
        # Focus on values that are likely to be valid
        for x in range(last_val + 1, min(last_val + 50, 257)):
            if can_append(seq, x, sums_set, suffix_sums):
                new_sums = get_new_sums(seq, x, suffix_sums)
                new_sums_set = sums_set.union(new_sums)
                new_suffix_sums = new_sums
                
                seq.append(x)
                dfs_extend(seq, new_sums_set, new_suffix_sums, depth + 1)
                seq.pop()
                
                if time.time() - start_time > TIMEOUT:
                    return True

        return False

    # Compute sums and suffixes for best_seq
    best_sums = get_all_sums(best_seq)
    best_suffix = []
    total = 0
    for a in reversed(best_seq):
        total += a
        best_suffix.append(total)
    best_suffix.reverse()
    
    # Try DFS extension
    dfs_extend(list(best_seq), best_sums, best_suffix, 0)
    
    # Final greedy extension
    final_seq = list(best_seq)
    final_sums = get_all_sums(final_seq)
    final_suffix = []
    total = 0
    for a in reversed(final_seq):
        total += a
        final_suffix.append(total)
    final_suffix.reverse()
    
    final_seq, final_sums, final_suffix = try_improve(final_seq, final_sums, final_suffix)
    
    if len(final_seq) > best_len:
        best_seq = final_seq
        best_len = len(best_seq)
    
    return best_seq
# EVOLVE-BLOCK-END
