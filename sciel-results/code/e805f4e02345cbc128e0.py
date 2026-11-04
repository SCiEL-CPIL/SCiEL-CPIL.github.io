import time
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # 1. Generate standard recursive superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
        
    # Convert to list of ints for easier manipulation
    S = list(word)
    
    # 2. Initialize counts
    # n is small (<=7), so tuples are fine
    counts = {}
    for i in range(len(S) - n + 1):
        p = tuple(S[i:i+n])
        counts[p] = counts.get(p, 0) + 1
        
    best_S = S[:]
    best_len = len(S)
    
    # Helper to check if removal at index i is valid
    def can_remove(S, counts, i, n):
        # Calculate affected old windows
        start_old = max(0, i - n + 1)
        end_old = min(i, len(S) - n)
        
        # Check if any count drops to 0
        for j in range(start_old, end_old + 1):
            p = tuple(S[j:j+n])
            if counts.get(p, 0) == 1:
                return False, None, None
        return True, start_old, end_old

    # Helper to perform removal
    def do_remove(S, counts, i, n, start_old, end_old):
        # Decrement counts for old
        for j in range(start_old, end_old + 1):
            p = tuple(S[j:j+n])
            counts[p] -= 1
            if counts[p] == 0:
                del counts[p]
        
        # Calculate and Increment new
        # New windows are formed by S[j:n] where j is in the straddling range
        # New length is len(S) - 1
        # New windows indices:
        start_new = max(0, i - n + 1)
        end_new = min(i - 1, len(S) - 1 - n)
        
        for k in range(start_new, end_new + 1):
            window_chars = []
            for idx in range(k, k+n):
                s_idx = idx if idx < i else idx + 1
                window_chars.append(S[s_idx])
            p_new = tuple(window_chars)
            counts[p_new] = counts.get(p_new, 0) + 1
        
        # Commit removal
        S.pop(i)

    # Helper to check validity of a swap
    def try_swap(S, counts, i, j, n):
        # Swap S[i] and S[j]
        S[i], S[j] = S[j], S[i]
        
        # Update counts
        # This is complex to do incrementally without full recalculation if n is large,
        # but n=7 is small.
        # Affected windows are those covering i or j.
        # It's safer to just recalculate counts for the whole string if we do many swaps?
        # No, we can update locally.
        
        # Windows affected by change at i: [i-n+1, i]
        # Windows affected by change at j: [j-n+1, j]
        
        # Let's just collect all affected window indices
        affected = set()
        for idx in range(max(0, i-n+1), min(i, len(S)-n) + 1):
            affected.add(idx)
        for idx in range(max(0, j-n+1), min(j, len(S)-n) + 1):
            affected.add(idx)
            
        # Decrement old
        for idx in affected:
            # We need the OLD value. But we already swapped.
            # So we must store old values or revert, check, then commit.
            pass
            
        # Better approach for swap:
        # 1. Store old chars
        # 2. Swap
        # 3. Calculate new windows
        # 4. Check if any perm count drops to 0 (considering the new windows and the fact that some old windows are gone)
        
        # Actually, simplest robust way for n=7:
        # Just recalculate the entire count dict?
        # Length ~5900. 5900 * 7 = 41300 ops.
        # Doing this for every swap attempt might be slow if we do thousands of swaps.
        # But we only do swaps when stuck.
        # Let's try to be smart.
        
        # Revert swap for now
        S[i], S[j] = S[j], S[i]
        return False # Placeholder, will implement in main loop logic

    stuck_passes = 0
    last_len = len(S)
    
    while time.time() < deadline:
        # Try a pass of removals
        # Randomize the order of indices to break symmetry
        indices = list(range(len(S)))
        random.shuffle(indices)
        
        removals_made = 0
        
        for i in indices:
            if time.time() > deadline:
                break
            if i >= len(S):
                continue
            
            valid, start_old, end_old = can_remove(S, counts, i, n)
            if valid:
                do_remove(S, counts, i, n, start_old, end_old)
                removals_made += 1
                # After removal, indices shift.
                # The `indices` list is now stale for indices > i.
                # We are iterating over a shuffled list.
                # If we removed i, any index > i in the list is now index-1.
                # Any index < i is unchanged.
                # Since we shuffled, we don't know the relative order easily.
                # It's safer to just break and restart the pass with a fresh shuffle if we remove?
                # Or, we can just let it continue, but we might miss some or hit invalid indices.
                # `if i >= len(S): continue` handles the length change.
                # But the index `i` in the list might now point to a different character.
                # This is acceptable for a heuristic search. We might re-check the same spot or skip one.
                # To be cleaner, if we remove, we can restart the inner loop?
                # No, that's expensive.
                # Let's just continue. The randomness helps.
        
        if removals_made > 0:
            best_S = S[:]
            best_len = len(S)
            stuck_passes = 0
        else:
            stuck_passes += 1
            # If stuck for a few passes, try a swap perturbation
            if stuck_passes % 10 == 0 and len(S) > 1:
                # Try a random swap
                i = random.randint(0, len(S) - 2)
                j = random.randint(i + 1, len(S) - 1)
                
                # Perform swap and check validity
                # We need to check if the swap breaks any permutation count.
                # Let's do a full validity check after swap.
                old_S = S[:]
                S[i], S[j] = S[j], S[i]
                
                # Recalculate counts to check validity?
                # Or just check the affected windows?
                # Let's use a helper to check if all perms are present.
                # Since we have `counts`, we can check if any count becomes 0.
                # But we need the new counts.
                
                # Efficient check:
                # 1. Identify affected windows (covering i or j)
                # 2. Decrement old counts for those windows (using old_S)
                # 3. Increment new counts for those windows (using new_S)
                # 4. Check if any count is 0.
                
                affected = set()
                for idx in range(max(0, i-n+1), min(i, len(old_S)-n) + 1):
                    affected.add(idx)
                for idx in range(max(0, j-n+1), min(j, len(old_S)-n) + 1):
                    affected.add(idx)
                    
                # Note: The set of window indices might shift if we consider the string structure,
                # but since we are just swapping values, the indices remain the same.
                
                temp_counts = counts.copy()
                valid_swap = True
                for idx in affected:
                    p_old = tuple(old_S[idx:idx+n])
                    temp_counts[p_old] -= 1
                    if temp_counts[p_old] == 0:
                        # Check if this perm appears in new windows?
                        # We haven't added new windows yet.
                        # We must add new windows first to see if they cover it.
                        pass
                    
                    p_new = tuple(S[idx:idx+n])
                    temp_counts[p_new] = temp_counts.get(p_new, 0) + 1
                    
                # Check if any perm is missing
                # We need to ensure no count is 0.
                # But we only updated `affected` windows.
                # If a perm was only in an affected window and is not in the new affected windows, it's gone.
                # If a perm was in an unaffected window, it's still there.
                
                # So we just need to check if any `temp_counts` value is 0?
                # No, `temp_counts` contains all perms.
                # If a perm was not in `affected`, its count is unchanged and > 0 (since we started valid).
                # If a perm was in `affected`, we updated it.
                # So we just need to check if any value in `temp_counts` is 0.
                # Wait, `temp_counts` might have keys that went to 0.
                
                for p, c in temp_counts.items():
                    if c == 0:
                        valid_swap = False
                        break
                        
                if valid_swap:
                    # Commit swap
                    S = old_S[:]
                    S[i], S[j] = S[j], S[i]
                    counts = temp_counts
                    # Reset stuck counter to allow more removal attempts
                    stuck_passes = 0
                else:
                    # Revert
                    S = old_S[:]
                    # counts remains unchanged (we used temp_counts)
        
        # If we've been stuck for a long time, maybe we should try a different strategy?
        # For now, just keep shuffling and swapping.
        
    return "".join(map(str, best_S))
