import time
import random
import sys
from functools import lru_cache

# EVOLVE-BLOCK-START
def _compute_sums(seq):
    """Compute all consecutive block sums for a sequence."""
    sums = set()
    n = len(seq)
    for i in range(n):
        total = 0
        for j in range(i, n):
            total += seq[j]
            sums.add(total)
    return sums

def _check_extension(seq, x, current_sums):
    """
    Efficiently check if appending x to seq creates any duplicate sums.
    Returns (is_valid, new_sums_set) if valid, or (False, None) if invalid.
    New sums are: x, x+seq[-1], x+seq[-1]+seq[-2], ...
    """
    if x in current_sums:
        return False, None
    
    new_sums = [x]
    total = x
    # Check tails
    for a in reversed(seq):
        total += a
        new_sums.append(total)
        if total in current_sums:
            return False, None
    
    # Check if any new sums collide with each other
    # Since seq is strictly increasing and positive, the tails are strictly increasing:
    # x < x+seq[-1] < x+seq[-1]+seq[-2] < ...
    # So new_sums are distinct internally. We only need to check against current_sums.
    # Wait, do we need to check new_sums against each other?
    # Yes, but as established, they are strictly increasing, so no internal collisions.
    
    # Update sums
    new_sum_set = current_sums.union(new_sums)
    return True, new_sum_set

def _backtrack_search(seq, current_sums, max_val=256, depth_limit=0, deadline=None, rng=None):
    """
    DFS backtracking to extend the sequence.
    Returns the longest valid sequence found.
    """
    best = list(seq)
    
    if deadline and time.time() > deadline:
        return best
    
    # Base case: if we hit depth limit or max_val, return current
    if len(seq) >= 256: # Impossible to exceed range
        return best
        
    if depth_limit > 0 and len(seq) > depth_limit:
        return best

    last_val = seq[-1] if seq else 0
    
    # Candidate range: last_val + 1 to max_val
    # To improve efficiency, we can prioritize candidates that are "far" from existing sums?
    # Or just iterate. Iteration order matters for finding long sequences.
    # Small steps might lead to dead ends quickly. Large steps are safer but limit length.
    # A mixed strategy: try a few small, then jump.
    
    candidates = list(range(last_val + 1, max_val + 1))
    
    # Shuffle candidates slightly to avoid deterministic traps, but keep bias towards smaller values
    # to maximize potential length? Actually, smaller values pack tighter, allowing more elements.
    # But smaller values create more sum collisions.
    # Let's try sorting by "safety" or just randomize the order of exploration.
    
    if rng:
        rng.shuffle(candidates)
    else:
        # Default: try smaller first, as they allow more room for future elements
        pass
        
    for x in candidates:
        if deadline and time.time() > deadline:
            break
            
        valid, new_sums = _check_extension(seq, x, current_sums)
        if valid:
            # Recurse
            next_seq = seq + [x]
            result = _backtrack_search(next_seq, new_sums, max_val, depth_limit - 1, deadline, rng)
            if len(result) > len(best):
                best = result
            # Early stop if we found a very long sequence?
            # If we found length L, we only care about L+1.
            # But we don't know the max possible.
            # Continue searching to see if we can do better.
            
    return best

def _greedy_with_lookahead(start_seq, max_val=256, deadline=None, lookahead_depth=3):
    """
    Greedy approach, but at each step, pick the candidate that leads to the
    longest extension via a short lookahead search.
    """
    seq = list(start_seq)
    if not seq:
        seq = [1]
        
    current_sums = _compute_sums(seq)
    rng = random.Random(42)
    
    while seq[-1] < max_val:
        if deadline and time.time() > deadline:
            break
            
        best_next = None
        best_extension_len = -1
        best_new_sums = None
        
        last_val = seq[-1]
        # Limit candidate range to keep lookahead fast
        # Trying all 256 candidates at each step with depth 3 is 256^3 ops? No, pruning helps.
        # Let's restrict candidates to a window to speed up.
        # Or just iterate all, but prune aggressively.
        
        for x in range(last_val + 1, max_val + 1):
            if deadline and time.time() > deadline:
                break
                
            valid, new_sums = _check_extension(seq, x, current_sums)
            if not valid:
                continue
            
            # Lookahead: try to extend this candidate
            # Use a shallow search
            # To save time, we can just check if it can be extended by 1 more step?
            # Or just pick the smallest valid x (greedy) but with a twist:
            # If we pick the smallest valid x, we maximize the number of remaining slots.
            # However, sometimes a slightly larger x avoids a collision that would kill the sequence later.
            
            # Simple heuristic: Pick the smallest valid x.
            # This is standard greedy. Why did the current program fail?
            # Probably because pure greedy gets stuck.
            # So, if we pick the smallest valid x, we should verify if it leads to a dead end immediately.
            # If it does, try the next smallest.
            
            # Let's implement "Greedy with 1-step lookahead":
            # Pick smallest valid x. Check if there exists ANY valid y > x for seq+[x].
            # If no y exists, reject x and try next x.
            
            can_extend = False
            for y in range(x + 1, max_val + 1):
                valid_y, _ = _check_extension(seq + [x], y, new_sums)
                if valid_y:
                    can_extend = True
                    break
            
            if can_extend:
                best_next = x
                best_new_sums = new_sums
                break # Take the first one that allows extension
            
            # If no x allows extension, we are stuck. Return current best.
            # But wait, maybe a larger x allows extension?
            # The loop above only breaks if it finds one. If it finishes the loop without breaking,
            # it means NO x in the range allowed a further extension.
            # In that case, we should accept the smallest valid x even if it doesn't allow extension?
            # No, if it doesn't allow extension, the sequence ends there.
            # If a larger x *did* allow extension, we would have found it?
            # No, the logic above:
            # for x:
            #   if valid:
            #     check if x can be extended
            #     if yes: take x and break
            #     if no: continue to next x
            # This finds the smallest x that is "not a dead end".
            
        if best_next is None:
            # No valid extension that allows further growth found.
            # Try to just append the smallest valid x to end the sequence?
            # Or just stop.
            # Let's try to append the smallest valid x regardless, to maximize length by 1.
            for x in range(last_val + 1, max_val + 1):
                valid, new_sums = _check_extension(seq, x, current_sums)
                if valid:
                    seq.append(x)
                    current_sums = new_sums
                    break
            break
            
        seq.append(best_next)
        current_sums = best_new_sums

    return seq

def _randomized_backtracking(max_val=256, deadline=None, num_restarts=20):
    """
    Run multiple randomized backtracking searches.
    """
    best = []
    rng = random.Random(123)
    
    for r in range(num_restarts):
        if deadline and time.time() > deadline:
            break
            
        # Start with a random small sequence
        start_len = rng.randint(2, 5)
        start_seq = []
        last = 0
        current_sums = set()
        
        for _ in range(start_len):
            # Pick a random value > last
            max_possible = max_val
            # Ensure we can pick
            if last + 1 > max_possible:
                break
            x = rng.randint(last + 1, min(last + 20, max_possible))
            valid, new_sums = _check_extension(start_seq, x, current_sums)
            if valid:
                start_seq.append(x)
                current_sums = new_sums
                last = x
            else:
                # Try a larger jump
                x = rng.randint(last + 10, max_possible)
                valid, new_sums = _check_extension(start_seq, x, current_sums)
                if valid:
                    start_seq.append(x)
                    current_sums = new_sums
                    last = x
                else:
                    break
        
        if not start_seq:
            start_seq = [1]
            current_sums = {1}

        # Run backtracking from this start
        # Limit depth to avoid explosion
        result = _backtrack_search(start_seq, current_sums, max_val, depth_limit=5, deadline=deadline, rng=rng)
        
        if len(result) > len(best):
            best = result
            
    return best

def construct():
    deadline = time.time() + 85  # 90s limit, 5s buffer
    
    best = []
    
    # 1. Greedy with 1-step lookahead (fast, good baseline)
    g_seq = _greedy_with_lookahead([1], max_val=256, deadline=deadline)
    if len(g_seq) > len(best):
        best = list(g_seq)
        
    # 2. Randomized Backtracking
    # This is the main "different strategy"
    rb_seq = _randomized_backtracking(max_val=256, deadline=deadline, num_restarts=50)
    if len(rb_seq) > len(best):
        best = list(rb_seq)
        
    # 3. Try to extend the best found using greedy
    if best:
        ext_seq = _greedy_with_lookahead(best, max_val=256, deadline=deadline)
        if len(ext_seq) > len(best):
            best = list(ext_seq)
            
    # 4. Final check and fallback
    if not best:
        best = [1]
        
    # Validate final result
    # (Optional: _is_valid_full check, but _check_extension guarantees it)
    
    return best

# EVOLVE-BLOCK-END
