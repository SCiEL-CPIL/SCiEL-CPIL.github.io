"""Structural redesign: exhaustive backtracking with prefix-sum collision tracking."""

import time
import sys

# Safety deadline: return best result before 90s wall-clock kill.
_DEADLINE = 85.0
_START_TIME = time.time()


def _remaining_budget():
    """Return remaining seconds before deadline."""
    return _DEADLINE - (time.time() - _START_TIME)


def _compute_block_sums(seq):
    """
    Compute all non-empty consecutive block sums for a sequence.
    Returns a set of (start, end, sum) tuples where seq[start..end-1] sums to sum.
    """
    sums = set()
    n = len(seq)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.add((i, j, s))
    return sums


def _has_collision(sums_set, new_value, current_len):
    """
    Check if appending new_value to a sequence of length current_len creates a collision.
    sums_set contains (start, end, sum) for all consecutive blocks in the prefix.
    The new element at index current_len creates new blocks:
      - single element: (current_len, current_len+1, new_value)
      - blocks ending at current_len+1 that start earlier: (i, current_len+1, sum_of_seq[i..current_len])
    
    We need to check:
    1. new_value is not already in the set of all sums (as a standalone block)
    2. For each i in 0..current_len, the sum seq[i] + ... + seq[current_len-1] + new_value
       is not already in the set of sums.
    
    Actually, let's think differently. The new blocks are those ending at position current_len
    (the new element). These are:
      - block (current_len, current_len+1): sum = new_value
      - block (i, current_len+1) for i < current_len: sum = (sum of seq[i..current_len-1]) + new_value
    
    We need none of these new sums to collide with existing sums OR with each other.
    """
    n = current_len  # length of existing sequence
    
    # Collect new block sums
    new_sums = {}
    # Single element block
    new_sums[(n, n + 1)] = new_value
    
    # Blocks starting at i and ending at n (inclusive of new element)
    # We need to compute these sums. But we don't have the sequence here...
    # This approach requires the sequence. Let's refactor.
    
    return False  # Placeholder - will use different approach


def _try_append(seq, sums_set, candidate_value):
    """
    Try to append candidate_value to seq.
    seq: current sequence (list of ints)
    sums_set: set of all current consecutive block sums (just the sum values, not indices)
    candidate_value: int to try appending
    
    Returns:
      - True and updated sums_set if valid, or
      - False if collision detected
    """
    n = len(seq)
    
    # New blocks created by appending candidate_value:
    # 1. Single element: [candidate_value] -> sum = candidate_value
    # 2. For each i in 0..n-1: seq[i] + ... + seq[n-1] + candidate_value
    
    # Check single element
    if candidate_value in sums_set:
        return False, sums_set
    
    # Compute new block sums for blocks ending at the new position
    # We need suffix sums of the current sequence
    # suffix_sum[i] = seq[i] + seq[i+1] + ... + seq[n-1]
    
    # Build suffix sums
    suffix_sums = [0] * (n + 1)
    suffix_sums[n] = 0
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + seq[i]
    
    # New block sums:
    # Block (i, n) for i in 0..n: sum = suffix_sums[i] + candidate_value
    # Note: block (n, n+1) is just candidate_value (suffix_sums[n] = 0)
    
    new_block_sums = set()
    for i in range(n + 1):
        s = suffix_sums[i] + candidate_value
        new_block_sums.add(s)
    
    # Check for collisions with existing sums
    if sums_set & new_block_sums:
        return False, sums_set
    
    # Check for collisions among new block sums (they're in a set, so duplicates are handled)
    # But we need to ensure they're all distinct from each other - the set handles that.
    # Actually, if two different blocks have the same sum, that's a collision.
    # The set new_block_sums would have fewer elements than n+1 if there are duplicates.
    
    if len(new_block_sums) != n + 1:
        return False, sums_set
    
    # All good - return updated sums
    updated_sums = sums_set | new_block_sums
    return True, updated_sums


def _backtrack_search(max_length, deadline_check=None):
    """
    Exhaustive backtracking search for the longest valid sequence.
    
    Strategy: Try to build a sequence of length max_length by choosing values
    in increasing order. Use backtracking to explore the search space.
    
    Returns the longest valid sequence found.
    """
    best_seq = []
    best_len = 0
    
    # We'll search for sequences of length target_len, starting from high and going down
    # until we find one that works
    
    def dfs(seq, sums_set, next_min_val):
        nonlocal best_seq, best_len
        
        if deadline_check and deadline_check():
            return
        
        if len(seq) > best_len:
            best_seq = seq[:]
            best_len = len(seq)
        
        # Try to extend
        for v in range(next_min_val, 257):
            if deadline_check and deadline_check():
                return
            
            valid, new_sums = _try_append(seq, sums_set, v)
            if valid:
                seq.append(v)
                dfs(seq, new_sums, v + 1)
                seq.pop()
            
            # Pruning: if we've already found a sequence of length >= target, we can stop
            # early in some cases, but for now let's just explore
    
    # Start with empty sequence
    dfs([], set(), 1)
    
    return best_seq


def _greedy_build():
    """
    Greedy baseline: pick the smallest available value that doesn't create a collision.
    """
    values = []
    sums_set = set()
    
    for v in range(1, 257):
        valid, new_sums = _try_append(values, sums_set, v)
        if valid:
            values.append(v)
            sums_set = new_sums
    
    return values


def _solve():
    """
    Main solver: combine greedy baseline with targeted backtracking.
    """
    # First, get a good baseline with greedy
    best_seq = _greedy_build()
    best_len = len(best_seq)
    
    # Now try to improve via backtracking
    # Try to find sequences of length best_len + 1, best_len + 2, etc.
    
    # For each target length > best_len, run a bounded backtracking search
    for target_len in range(best_len + 1, 20):  # Cap at 20 to avoid infinite search
        if _remaining_budget() < 5.0:
            break
        
        found = _search_for_length(target_len)
        if found is not None and len(found) > best_len:
            best_seq = found
            best_len = len(found)
    
    return best_seq


def _search_for_length(target_len):
    """
    Search for a valid sequence of exactly target_len elements.
    Uses backtracking with aggressive pruning.
    """
    if _remaining_budget() < 2.0:
        return None
    
    seq = []
    sums_set = set()
    
    def dfs(next_min_val):
        nonlocal seq, sums_set
        
        if _remaining_budget() < 1.0:
            return None
        
        if len(seq) == target_len:
            return seq[:]
        
        remaining_slots = target_len - len(seq)
        
        # Pruning: need enough values left
        if next_min_val + remaining_slots - 1 > 256:
            return None
        
        # Try values in increasing order
        for v in range(next_min_val, 257):
            if _remaining_budget() < 1.0:
                return None
            
            # Pruning: not enough room for remaining elements
            if v + remaining_slots - 1 > 256:
                break
            
            valid, new_sums = _try_append(seq, sums_set, v)
            if valid:
                seq.append(v)
                old_sums = sums_set
                sums_set = new_sums
                
                result = dfs(v + 1)
                if result is not None:
                    return result
                
                seq.pop()
                sums_set = old_sums
        
        return None
    
    result = dfs(1)
    return result


def construct():
    """
    Return a strictly increasing sequence of integers in [1,256] with distinct
    consecutive block sums, maximizing sequence length.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    result = _solve()
    
    # Final validation
    if not result:
        result = _greedy_build()
    
    # Double-check validity
    n = len(result)
    for i in range(1, n):
        if result[i] <= result[i - 1]:
            result = _greedy_build()
            break
    for v in result:
        if v < 1 or v > 256:
            result = _greedy_build()
            break
    
    # Verify distinct block sums
    seen_sums = set()
    valid = True
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += result[j]
            if s in seen_sums:
                valid = False
                break
            seen_sums.add(s)
        if not valid:
            break
    
    if not valid:
        result = _greedy_build()
    
    return result
