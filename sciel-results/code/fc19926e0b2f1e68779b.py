import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # Initial valid sequence: Powers of 2 guarantee distinct sums
    # [1, 2, 4, 8, 16, 32, 64, 128] -> Length 8
    # Let's verify: Sums of consecutive blocks of powers of 2 are distinct.
    # Actually, powers of 2 are a Sidon-like set for consecutive sums?
    # Sum of a[i..j] = 2^(i+1) + ... + 2^(j+1) = 2^(j+2) - 2^(i+1).
    # Since binary representation is unique, these sums are distinct.
    # Max power of 2 in [1, 256] is 128. So [1, 2, 4, 8, 16, 32, 64, 128] is valid.
    # Can we add 256? 256 > 128.
    # New sums involving 256: 256, 256+128=384, ...
    # Are any of these equal to existing sums?
    # Existing sums are between 1 and 255.
    # 256 is not in existing sums.
    # So [1, 2, 4, 8, 16, 32, 64, 128, 256] is valid?
    # Let's check: 256 is distinct from 1..255.
    # Sums including 256 are >= 256.
    # So yes, length 9 is a valid baseline.
    
    best_seq = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    best_len = len(best_seq)
    
    # Precompute all possible consecutive sums for the baseline?
    # No, we will do backtracking from a slightly smaller start or refine.
    # Actually, backtracking from scratch is safer to find longer sequences.
    # But starting from a good seed helps.
    
    # Let's try a backtracking search that starts with a small prefix and tries to extend.
    # To avoid exponential explosion, we use heuristics and time limits.
    
    # Strategy: DFS with backtracking.
    # State: current sequence, set of used sums, current length.
    # We want to maximize length.
    # We can prune if current_length + (256 - last_element) < best_len.
    
    # To make it efficient, we will iterate through candidates in increasing order?
    # Or random order?
    # Increasing order is good for finding valid sequences quickly.
    
    # Let's implement a recursive DFS.
    # To avoid recursion depth issues, we'll use an iterative approach or increase limit.
    sys.setrecursionlimit(10000)
    
    def compute_sums(seq):
        sums = set()
        n = len(seq)
        for i in range(n):
            total = 0
            for j in range(i, n):
                total += seq[j]
                sums.add(total)
        return sums
    
    # Initial baseline sums
    current_sums = compute_sums(best_seq)
    
    # We will try to improve upon the baseline.
    # We'll do a search that starts with the baseline and tries to insert/modify?
    # No, DFS from scratch is more robust for finding the *global* max if we prune well.
    # But DFS from scratch might be too slow.
    # Let's use the baseline as the initial "best" and run a limited DFS to see if we can beat it.
    
    # We will perform a DFS that explores sequences starting with small numbers.
    # We'll use a "branch and bound" style.
    
    def dfs(seq, sums, last_val, depth):
        nonlocal best_seq, best_len, deadline
        
        if time.time() > deadline:
            return
        
        if len(seq) > best_len:
            best_seq = seq[:]
            best_len = len(seq)
        
        # Pruning: If we can't possibly beat best_len, stop.
        # Max possible extension: We can add at most (256 - last_val) numbers if we pick consecutive integers?
        # Actually, we can pick any integers > last_val.
        # The maximum number of additional elements we can add is bounded by the number of available integers > last_val.
        # That is 256 - last_val.
        if len(seq) + (256 - last_val) <= best_len:
            return
            
        # Try adding next element
        # To find long sequences, we should probably try smaller candidates first?
        # Smaller candidates leave more room for larger candidates later?
        # Not necessarily. But let's try increasing order.
        
        next_start = last_val + 1
        if next_start > 256:
            return
            
        for x in range(next_start, 257):
            if time.time() > deadline:
                return
            
            # Check if x can be added
            # New sums: x, x+seq[-1], x+seq[-1]+seq[-2], ...
            # We need to check if any new sum is in 'sums'.
            
            # Optimization:
            # The new sums are strictly increasing: x < x+seq[-1] < ...
            # We just need to check if any of these values are in the existing 'sums' set.
            
            valid = True
            new_sums_to_add = []
            total = x
            if total in sums:
                valid = False
            else:
                new_sums_to_add.append(total)
                
            if valid:
                for a in reversed(seq):
                    total += a
                    if total in sums:
                        valid = False
                        break
                    new_sums_to_add.append(total)
            
            if not valid:
                continue
                
            # Recursive step
            # To speed up, we can use a set for sums.
            # But copying the set at every step is expensive O(N^2) per step.
            # Total steps in DFS can be large.
            # For N~20, N^2=400. 400 * 20 = 8000 ops per node.
            # If we explore 10,000 nodes, 80M ops. Might be tight in Python.
            
            # Alternative: Pass a list of sums and use a global set with add/remove?
            # Or just use the set copy for simplicity and rely on pruning.
            
            new_sums = sums.union(new_sums_to_add)
            
            dfs(seq + [x], new_sums, x, depth + 1)
            
            # Backtracking is implicit in the recursive call structure if we don't modify in place.
            # Since we create new lists and sets, no explicit undo is needed.
            
            # Heuristic: If we found a sequence of length >= best_len, we might stop early?
            # No, we want to maximize.
            
            # Early termination for this branch:
            # If we've already found a sequence of length L, and we are at depth D,
            # and D + (256 - x) <= L, we can prune further branches for this x?
            # This is already handled at the start of dfs.
            
            # To improve search, we can try randomizing the order of x?
            # If we always go left (smallest x), we might find one long sequence and stop.
            # But maybe a different choice of x leads to a longer one.
            # Let's try a few random restarts or randomized DFS.
            
            # For now, let's just do the DFS.
            
            # To avoid getting stuck in one deep path, we can limit the depth or use beam search?
            # No, DFS is fine with time limit.
            
            # Break if we've updated best_len and are now "safe"?
            # No, keep searching.
            
            # To prevent exploring too many "bad" small x's, we can skip some?
            # Let's stick to simple DFS first.
            
            # If the time limit is close, we should stop.
            if time.time() > deadline:
                return

    # Start DFS from empty sequence?
    # Or from the baseline?
    # Starting from empty allows finding potentially better structures.
    # But it might take a long time to reach length 9.
    # Let's start from a short prefix to guide the search.
    # Prefix: [1]
    # Sums: {1}
    
    # Actually, let's try multiple starting points or just start from scratch.
    # Starting from scratch with x=1 is the most natural.
    
    # To ensure we find the baseline quickly, we can check if the baseline is valid (it is).
    # We'll run DFS from [1].
    
    # One issue: DFS from [1] might get stuck in a local optimum of length 8 or 9.
    # The baseline is 9. If DFS finds 9, it continues.
    
    # Let's run DFS.
    # To make it more robust, we can run it with different random seeds for candidate ordering?
    # For now, deterministic increasing order.
    
    # We'll also try a randomized version if the deterministic one finishes quickly.
    
    # Run DFS
    initial_seq = [1]
    initial_sums = {1}
    dfs(initial_seq, initial_sums, 1, 1)
    
    # If we have time, try a randomized DFS to escape local optima?
    # The current DFS explores all possibilities (pruned) in increasing order.
    # This is exhaustive within the time limit.
    
    # Let's add a randomized restart mechanism if time permits.
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        # Try a few randomized DFS runs
        for _ in range(5):
            if time.time() > deadline:
                break
            
            # Randomize the order of candidates?
            # We can't easily randomize the range in the for loop without overhead.
            # Instead, let's pick a random starting prefix?
            # Or just run the same DFS again? No, it's deterministic.
            
            # Let's try a "Greedy + Local Search" hybrid if DFS didn't find anything better?
            # But DFS is stronger.
            
            # Let's try a different heuristic:
            # Randomly pick the next x from a shuffled list of valid candidates?
            # This requires knowing valid candidates, which is expensive.
            
            # Let's just stick with the result from DFS.
            pass

    return best_seq

# EVOLVE-BLOCK-END
