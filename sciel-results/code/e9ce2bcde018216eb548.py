import time
import sys

# EVOLVE-BLOCK-START

def _is_valid_full(values):
    """Check if a full sequence has all distinct consecutive block sums."""
    n = len(values)
    seen = set()
    for i in range(n):
        total = 0
        for j in range(i, n):
            total += values[j]
            if total in seen:
                return False
            seen.add(total)
    return True

def _greedy_forward(values, max_val=256):
    """Greedy: pick smallest valid next element."""
    current = list(values)
    if not current:
        return []
    
    cur_sums = set()
    for i in range(len(current)):
        total = 0
        for j in range(i, len(current)):
            total += current[j]
            cur_sums.add(total)
            
    for x in range(current[-1] + 1, max_val + 1):
        tails = [x]
        total = x
        for a in reversed(current):
            total += a
            tails.append(total)
        if not cur_sums.intersection(tails):
            current.append(x)
            cur_sums.update(tails)
    return current

def _dfs_backtrack(L, max_val, deadline):
    """
    DFS with backtracking to find a sequence of length L.
    Uses a set of all consecutive block sums for pruning.
    """
    seq = []
    sums = set()
    
    def backtrack(idx, min_val):
        if time.time() > deadline:
            return False
        
        if idx == L:
            return True
            
        # We need to place elements at positions idx..L-1
        # The remaining elements must be strictly increasing and <= max_val
        # So the current element x must satisfy:
        # x >= min_val
        # x + (L - idx - 1) <= max_val  => x <= max_val - (L - idx - 1)
        
        max_possible = max_val - (L - idx - 1)
        if min_val > max_possible:
            return False
            
        # Try candidates. 
        # Heuristic: We want to keep numbers small to allow more room?
        # Or large? 
        # Usually, for "distinct sums", spreading out helps, but we are constrained by max_val.
        # Let's try from min_val upwards.
        
        # Optimization: If we have a lot of time, maybe try a few random ones?
        # No, keep it deterministic first.
        
        for x in range(min_val, max_possible + 1):
            if time.time() > deadline:
                return False
                
            # Check if x can be appended
            if not seq:
                new_sums = {x}
            else:
                # Compute tails: x, seq[-1]+x, seq[-2]+seq[-1]+x, ...
                tails = [x]
                total = x
                for a in reversed(seq):
                    total += a
                    tails.append(total)
                
                # Check collision
                # If any tail is in sums, invalid
                # Also, tails are strictly increasing, so no internal collision.
                if any(t in sums for t in tails):
                    continue
                
                new_sums = set(tails)
            
            # Tentatively add
            seq.append(x)
            old_sums_size = len(sums)
            sums.update(new_sums)
            
            if backtrack(idx + 1, x + 1):
                return True
            
            # Backtrack
            seq.pop()
            # Remove new_sums from sums
            # Since new_sums are disjoint from old sums (we checked), we can just remove them
            for s in new_sums:
                sums.discard(s)
                
        return False

    if backtrack(0, 1):
        return list(seq)
    return None

def _randomized_dfs(L, max_val, deadline, seed):
    """
    Randomized DFS with restarts.
    Tries different value ranges or orders.
    """
    import random
    rng = random.Random(seed)
    
    best_found = None
    
    # We will run several short searches
    # Or one search with randomized ordering?
    # Randomized ordering of candidates is tricky because domain is ordered.
    # Instead, we can randomize the starting point or the "bias".
    
    # Strategy: 
    # 1. Standard DFS from 1.
    # 2. DFS with a "gap" bias?
    
    # Let's stick to one robust DFS but with a better heuristic if possible.
    # Actually, the deterministic DFS might be fast enough if L is not too large.
    # For L=10, it might take a while.
    
    # Let's add a simple heuristic:
    # When choosing x, prefer values that are "central" or have specific properties?
    # Hard to say.
    
    # Let's just run the deterministic DFS.
    return _dfs_backtrack(L, max_val, deadline)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    best = []
    
    # 1. Get a baseline via greedy
    greedy_seq = _greedy_forward([])
    if len(greedy_seq) > len(best):
        best = list(greedy_seq)
    
    # 2. Try to improve with DFS
    # We know greedy gives length G.
    # Let's try to find length G+1, G+2, etc.
    
    current_len = len(best)
    
    # We can try to extend the greedy solution?
    # Greedy is already maximal for that prefix.
    # So we need a different prefix.
    
    # Let's iterate on target length
    # Start from current_len + 1
    # But DFS for L=12 might be slow.
    # Let's estimate.
    
    # To be safe, let's try lengths in increasing order?
    # Or decreasing?
    # If we try L=15 and it fails, we spent a lot of time.
    # If we try L=current_len+1, we might find it quickly.
    
    # Let's try L from current_len + 1 up to 15.
    # But if L is impossible, we waste time.
    
    # Better:
    # We don't know the max.
    # Let's try a binary search approach?
    # No, let's just try a few lengths.
    
    # Heuristic:
    # The max length is likely small (<= 12 or 13).
    # Let's try L = current_len + 1, + 2, ... up to a limit or timeout.
    
    # We need to be careful with time.
    # Let's allocate time based on remaining time.
    
    # For each L, give it a portion of the remaining time.
    # Or just run until deadline?
    
    # If we run DFS for L=12 and it takes 30s, we have 55s left.
    # Then we try L=13?
    
    # Let's try:
    # 1. Try L = current_len + 1 with 20s limit.
    # 2. If found, try L + 1 with remaining time.
    # 3. If not found, assume L is hard/impossible and stop?
    #    Or try a different strategy?
    
    # Let's implement a loop that increases L and allocates time.
    
    target_len = current_len + 1
    max_target = 15  # Arbitrary cap
    
    while target_len <= max_target and time.time() < deadline - 5:
        # Allocate time for this attempt
        # If we have 30s left, and target_len is 12, give it 10s?
        # If we find it, we have more time for 13.
        
        # Simple: give it 10% of remaining time or 5s, whichever is less?
        # No, if we have 50s, we can afford a longer search.
        
        # Let's use a fraction of remaining time.
        remaining = deadline - time.time()
        attempt_time = min(remaining - 2, 15.0)  # Max 15s per attempt?
        
        if attempt_time < 1:
            break
            
        # Run DFS with this time limit
        found_seq = _dfs_backtrack(target_len, 256, time.time() + attempt_time)
        
        if found_seq:
            if len(found_seq) > len(best):
                best = found_seq
                # Update current_len?
                # We found target_len.
                # Can we find target_len + 1?
                target_len += 1
            else:
                # Found a sequence but not better? Shouldn't happen if target_len > len(best)
                target_len += 1
        else:
            # Failed to find target_len.
            # Is it impossible?
            # We spent `attempt_time`.
            # If we have plenty of time, maybe retry?
            # Or give up on this L?
            
            # If we have > 20s left, maybe retry with more time?
            # But if it's impossible, more time won't help.
            # Let's assume if we fail with 15s, it's likely impossible or very hard.
            # Let's stop to save time for other strategies?
            # Or break.
            
            # Let's try one more time with slightly more time if we have it?
            # No, let's break to avoid wasting time on a likely impossible L.
            # However, it's possible the search space is just large.
            # Let's break.
            break
    
    # 3. Try to extend the best found with greedy?
    # If we found a sequence via DFS, it might not be extendable greedily?
    # Greedy extension requires the prefix to be such that greedy works.
    # But any valid sequence can be extended greedily if possible.
    # So let's try to extend `best`.
    
    if time.time() < deadline - 1:
        extended = _greedy_forward(best)
        if len(extended) > len(best):
            best = extended
    
    # 4. Random search for a quick win?
    # Maybe a random sequence of length len(best)+1?
    # Unlikely to find quickly.
    
    # Final validity check
    if not _is_valid_full(best):
        # Fallback to greedy
        best = _greedy_forward([])
        
    return best

# EVOLVE-BLOCK-END
