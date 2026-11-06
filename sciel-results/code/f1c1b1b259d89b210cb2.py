import time
import random

def _all_block_sums(seq):
    """Compute all nonempty consecutive block sums of seq."""
    sums = set()
    n = len(seq)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.add(s)
    return sums

def _check_valid(seq):
    """Check that seq is strictly increasing, all elements in [1,256], and all consecutive block sums distinct."""
    if not seq:
        return True
    for i in range(1, len(seq)):
        if seq[i] <= seq[i-1]:
            return False
    if any(x < 1 or x > 256 for x in seq):
        return False
    sums = _all_block_sums(seq)
    n = len(seq)
    expected = n * (n + 1) // 2
    if len(sums) != expected:
        return False
    return True

def _greedy_build(start_value, max_time=2.0, strategy="largest"):
    """
    Build a sequence starting with start_value using a greedy strategy.
    strategy: "smallest", "largest"
    """
    start_time = time.time()
    seq = [start_value]
    sums_set = {start_value}
    last = start_value
    
    while True:
        if time.time() - start_time > max_time:
            break
        
        n = len(seq)
        # Compute suffix sums for conflict check
        suffix_sums = [0] * (n + 1)
        s = 0
        for k in range(n - 1, -1, -1):
            s += seq[k]
            suffix_sums[k] = s
        suffix_sums[n] = 0
        
        valid_next = []
        for x in range(last + 1, 257):
            conflict = False
            for k in range(n + 1):
                if suffix_sums[k] + x in sums_set:
                    conflict = True
                    break
            if not conflict:
                valid_next.append(x)
        
        if not valid_next:
            break
        
        if strategy == "smallest":
            best_x = valid_next[0]
        elif strategy == "largest":
            best_x = valid_next[-1]
        else:
            best_x = random.choice(valid_next)
        
        seq.append(best_x)
        # Update sums
        for k in range(n + 1):
            sums_set.add(suffix_sums[k] + best_x)
        last = best_x
        
    return seq

def _can_reach_length(N, S, v, target_L):
    """
    Check if it's possible to extend a sequence of length N, sum S, last value v
    to reach length target_L, such that the total sum is sufficient to hold
    target_L(target_L+1)/2 distinct sums.
    """
    if N >= target_L:
        return True
    
    k_needed = target_L - N
    # We need to add k_needed elements.
    # The minimum sum of these k_needed elements (strictly increasing, starting > v)
    # is (v+1) + (v+2) + ... + (v+k_needed)
    min_add_sum = k_needed * v + k_needed * (k_needed + 1) // 2
    max_possible_total_sum = S + min_add_sum
    
    # The target number of distinct sums is target_L * (target_L + 1) // 2
    required_sum_capacity = target_L * (target_L + 1) // 2
    
    # If even the minimal extension doesn't provide enough sum range, we can't reach target_L
    if max_possible_total_sum < required_sum_capacity:
        return False
    
    return True

def construct():
    """
    Main entry point. Returns a strictly increasing sequence in [1,256] with distinct consecutive-block sums.
    Uses Pruned DFS with Sum Capacity Bound.
    """
    total_start = time.time()
    deadline = total_start + 50.0  # Safety margin for 60s limit
    
    best_seq = []
    best_len = 0
    
    # 1. Initial Greedy Passes to get a good lower bound
    # Try starting from 1, 2, 3... with different strategies
    # "Largest" strategy often yields higher total sums, which helps with the sum capacity bound.
    for start_val in range(1, 10):
        if time.time() > deadline - 5:
            break
        for strategy in ["largest", "smallest"]:
            seq = _greedy_build(start_val, max_time=1.0, strategy=strategy)
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = seq[:]

    # 2. Pruned DFS
    # State: (seq, sums_set, last_val, total_sum)
    # Pruning:
    # 1. Time
    # 2. Length potential: len(seq) + (256 - last_val) <= best_len
    # 3. Sum Capacity: 
    #    To reach length L, we need TotalSum >= L(L+1)/2.
    #    Current sum is S. We can add at most k elements.
    #    Min additional sum for k elements starting after v is k*v + k(k+1)/2.
    #    If S + k*v + k(k+1)/2 < L(L+1)/2, we can't reach L.
    #    We want to beat best_len, so we check if we can reach best_len + 1.
    
    def dfs(seq, sums_set, last_val, total_sum):
        nonlocal best_seq, best_len
        
        # Time check
        if time.time() > deadline:
            return
        
        current_len = len(seq)
        
        # Update best
        if current_len > best_len:
            best_len = current_len
            best_seq = seq[:]
            
        # Pruning 1: Max possible length based on available values
        if current_len + (256 - last_val) <= best_len:
            return
            
        # Pruning 2: Sum Capacity Bound
        # Can we even reach best_len + 1?
        if not _can_reach_length(current_len, total_sum, last_val, best_len + 1):
            return
            
        n = current_len
        # Compute suffix sums
        suffix_sums = [0] * (n + 1)
        s = 0
        for k in range(n - 1, -1, -1):
            s += seq[k]
            suffix_sums[k] = s
        suffix_sums[n] = 0
        
        # Generate candidates
        # Heuristic: Try larger values first to increase total sum capacity faster?
        # Or smaller values to pack more elements?
        # Let's try descending order (largest first) to expand sum space.
        
        candidates = range(last_val + 1, 257)
        
        # To diversify, we could shuffle, but for maximizing length, 
        # systematic search is often better.
        # Let's try descending first.
        for x in reversed(list(candidates)):
            # Check validity
            conflict = False
            for k in range(n + 1):
                if suffix_sums[k] + x in sums_set:
                    conflict = True
                    break
            
            if conflict:
                continue
            
            # Recurse
            new_seq = seq + [x]
            new_sums = sums_set.copy()
            for k in range(n + 1):
                new_sums.add(suffix_sums[k] + x)
            
            dfs(new_seq, new_sums, x, total_sum + x)
            
            # Early exit if time is up
            if time.time() > deadline:
                return

    # Start DFS from empty sequence
    if time.time() < deadline - 5:
        dfs([], set(), 0, 0)
        
    # Verify result
    if not _check_valid(best_seq):
        # Fallback
        best_seq = [1, 2, 5]
        
    return best_seq
