import time
import random
import bisect

def _compute_block_sums(candidate):
    """Compute all non-empty consecutive block sums for a candidate sequence."""
    sums = set()
    n = len(candidate)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += candidate[j]
            sums.add(s)
    return sums

def _is_valid_candidate(candidate):
    """Check if all consecutive block sums are distinct."""
    sums = set()
    n = len(candidate)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += candidate[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _greedy_build(order):
    """Greedy build: iterate through order, add x if it maintains validity."""
    selected = []
    for x in order:
        candidate = selected + [x]
        candidate.sort()
        if _is_valid_candidate(candidate):
            selected = candidate
    return selected

def _try_insert(selected, x):
    """Try to insert x into sorted selected and check validity."""
    pos = bisect.bisect_left(selected, x)
    if pos < len(selected) and selected[pos] == x:
        return None
    new_candidate = selected[:pos] + [x] + selected[pos:]
    if _is_valid_candidate(new_candidate):
        return new_candidate
    return None

def _local_search(selected, available, deadline):
    """Local search: try to insert available numbers into the sequence."""
    current = selected[:]
    available_set = set(available)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        avail_list = list(available_set)
        random.shuffle(avail_list)
        for x in avail_list:
            if x in current:
                continue
            new_seq = _try_insert(current, x)
            if new_seq:
                current = new_seq
                available_set.remove(x)
                improved = True
    return current

def _dfs_backtrack(selected, available, max_depth, deadline, memo=None):
    """
    DFS backtracking search.
    selected: current sorted sequence
    available: set of numbers not yet used
    max_depth: target length to reach
    deadline: time limit
    memo: dictionary to memoize failed states (frozenset of block sums -> set of last elements tried)
    
    Returns the best sequence found (length >= len(selected)).
    """
    if memo is None:
        memo = {}
    
    if len(selected) >= max_depth:
        return selected[:]
    
    if time.time() > deadline:
        return selected[:]
    
    # Compute current block sums
    current_sums = _compute_block_sums(selected)
    last_val = selected[-1] if selected else 0
    
    # Memoization key: use a hashable representation of the sum set and last value
    # To keep memo small, we'll use a simplified key
    sums_tuple = tuple(sorted(current_sums))
    key = (last_val, sums_tuple)
    
    if key in memo:
        # If we've tried this state before and it didn't lead to a solution of sufficient length
        # We can skip it, but we need to be careful about what "failed" means
        # For simplicity, we'll just return early if we've seen this exact state
        return selected[:]
    
    # Try adding each available number
    best_result = selected[:]
    avail_list = sorted(available)
    
    for x in avail_list:
        if time.time() > deadline:
            break
        if x in selected:
            continue
        
        # Try inserting x at the end (since we want increasing sequence, x must be > last)
        # Actually, for strictly increasing, x must be > selected[-1] if selected is non-empty
        if selected and x <= selected[-1]:
            continue
        
        new_selected = selected + [x]
        new_sums = _compute_block_sums(new_selected)
        
        # Check if all sums are distinct
        if len(new_sums) == len(new_selected) * (len(new_selected) + 1) // 2:
            new_available = available - {x}
            result = _dfs_backtrack(new_selected, new_available, max_depth, deadline, memo)
            if len(result) > len(best_result):
                best_result = result
    
    memo[key] = True
    return best_result

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_seq = []
    
    # Baseline: greedy builds
    seq = _greedy_build(list(range(1, 257)))
    if len(seq) > len(best_seq):
        best_seq = seq
        
    seq = _greedy_build(list(range(256, 0, -1)))
    if len(seq) > len(best_seq):
        best_seq = seq
    
    # DFS backtracking from a good starting point
    # Start from the best greedy result and try to extend it
    if len(best_seq) > 0:
        available = set(range(1, 257)) - set(best_seq)
        # Try DFS with a target length slightly above current best
        target_len = len(best_seq) + 1
        dfs_result = _dfs_backtrack(best_seq, available, target_len, deadline)
        if len(dfs_result) > len(best_seq):
            best_seq = dfs_result
    
    # Random restarts with greedy + local search
    rng = random.Random(42)
    
    while time.time() < deadline:
        order = list(range(1, 257))
        rng.shuffle(order)
        
        seq = _greedy_build(order)
        
        if len(seq) > len(best_seq):
            best_seq = seq
        
        available = set(range(1, 257)) - set(seq)
        improved_seq = _local_search(seq, available, deadline)
        
        if len(improved_seq) > len(best_seq):
            best_seq = improved_seq
        
        # Try DFS extension from this improved sequence
        if len(improved_seq) > 0 and time.time() < deadline - 5:
            avail2 = set(range(1, 257)) - set(improved_seq)
            target2 = len(improved_seq) + 1
            dfs_res2 = _dfs_backtrack(improved_seq, avail2, target2, deadline)
            if len(dfs_res2) > len(best_seq):
                best_seq = dfs_res2
    
    return best_seq
