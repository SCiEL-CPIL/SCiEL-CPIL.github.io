<<<<<<< SEARCH
    # Local Search Strategy:
    # We repeatedly try to remove characters.
    # If we get stuck (no removals in a full pass), we perturb the string
    # by swapping two random characters or inserting a random character,
    # then try to remove again. This helps escape local minima.
    
    # We track the number of consecutive failed passes to decide when to perturb
    failed_passes = 0
    max_failed_passes_before_perturb = 3
    
    # Pre-compute factorial for validity check
    target_perms = math.factorial(n)
    
    while time.time() < deadline:
        # Attempt a full pass of removals from right to left
        # Removing from right to left ensures that indices < i are not affected
        # by the shift caused by removal at i, so we can continue scanning leftwards.
        removed_any = False
        i = len(S) - 1
        
        while i >= 0:
            if time.time() > deadline:
                break
            
            # Identify windows affected by removing S[i]
            # Old windows containing index i:
            # Start index j such that j <= i <= j+n-1  =>  i-n+1 <= j <= i
            # Also j must be a valid window start: 0 <= j <= len(S)-n
            start_old = max(0, i - n + 1)
            end_old = min(i, len(S) - n)
            
            valid_removal = True
            old_perms_to_dec = []
            
            # Check if any old window has count 1
            for j in range(start_old, end_old + 1):
                p = tuple(S[j:j+n])
                if counts.get(p, 0) == 1:
                    valid_removal = False
                    break
                old_perms_to_dec.append(p)
            
            if not valid_removal:
                i -= 1
                continue
            
            # If valid, perform the removal
            # 1. Decrement counts for old windows
            for p in old_perms_to_dec:
                counts[p] -= 1
                if counts[p] == 0:
                    del counts[p]
            
            # 2. Identify new windows created by the removal
            # The new string is S_new = S[:i] + S[i+1:]
            # A new window at index k in S_new straddles the gap if:
            # It uses some elements from S[:i] and some from S[i+1:]
            # Specifically, the window starts at k in S_new.
            # The elements are S_new[k], ..., S_new[k+n-1].
            # Mapping to S indices:
            # S_new[idx] = S[idx] if idx < i, else S[idx+1]
            
            # The straddling windows are those where the window start k in S_new
            # is such that the window covers index i in S.
            # In S_new, the "gap" is at position i.
            # Windows that cross this gap start at k in S_new where:
            # k < i and k+n-1 >= i  (in S_new coordinates)
            # Actually, it's easier to think in terms of S indices.
            # The new windows are formed by joining S[j] and S[j+1] for j < i and S[j+1] and S[j+2] for j >= i? No.
            
            # Let's derive the new windows explicitly.
            # New length L' = len(S) - 1.
            # New window at index k (0 <= k <= L'-n) consists of:
            # S_new[k : k+n]
            # If k+n <= i: window is S[k:k+n] (Unchanged, already counted)
            # If k >= i: window is S[k+1:k+1+n] (Unchanged content, shifted, already counted)
            # The only *new* combinations are those where the window includes characters from both sides of the removed index i.
            # This happens if k < i and k+n-1 >= i.
            # i.e., i - n + 1 <= k <= i - 1.
            
            start_new = max(0, i - n + 1)
            end_new = min(i - 1, (len(S) - 1) - n) # L' - n
            
            for k in range(start_new, end_new + 1):
                # Construct the new window tuple
                # It consists of:
                # S[k : i]  (length i-k)
                # S[i+1 : i+1 + (n - (i-k))] = S[i+1 : k+n+1]
                
                # Let's just build the tuple by mapping indices
                window_chars = []
                for idx in range(k, k+n):
                    s_idx = idx if idx < i else idx + 1
                    window_chars.append(S[s_idx])
                
                p_new = tuple(window_chars)
                counts[p_new] = counts.get(p_new, 0) + 1
            
            # Commit removal
            S.pop(i)
            removed_any = True
            best_S = S[:]
            best_len = len(S)
            
            # Do not decrement i, because the element that was at i+1 is now at i,
            # and we haven't checked it yet (or rather, the indices > i are shifted left,
            # but we are scanning downwards. The element now at i was previously at i+1.
            # We have already processed indices > i. So we should check the new i next.)
            # Wait, if we remove i, the element at i+1 moves to i.
            # We are scanning i downwards. We have already processed i+1, i+2, etc.
            # So the element now at i (which was i+1) has already been considered for removal?
            # No, when we were at i+1, we checked if removing S[i+1] was valid.
            # Now S[i+1] is at S[i]. The context has changed (S[i-1] is to the left, S[i+1] (old i+2) is to the right).
            # So we should check the new S[i] again. Thus, do not decrement i.
            
        if removed_any:
            failed_passes = 0
        else:
            failed_passes += 1
            
            # If stuck for a few passes, perturb
            if failed_passes >= max_failed_passes_before_perturb:
                # Check if we have enough time for a perturbation and rebuild
                if time.time() > deadline - 5.0:
                    break
                
                # Strategy: Swap two random characters
                # This is a simple move to break the local minimum.
                # We must ensure the swap doesn't make the string invalid immediately?
                # Actually, swaps can create invalid states (missing perms).
                # But we are going to try to remove characters next.
                # If the swap makes the string invalid, we won't be able to remove anything
                # that fixes it easily, and we might lose the valid state.
                
                # Safer perturbation: Try to insert a character? No, we want to shorten.
                # Better perturbation: Randomly remove a character even if it's "invalid" locally?
                # No, that breaks validity.
                
                # Let's try a different approach for perturbation:
                # Instead of random swaps which risk validity, let's try to "shuffle"
                # the order of attempts or use a different removal heuristic.
                
                # Alternative: If stuck, it means every character is critical to some unique window.
                # This often happens in the standard construction.
                # A known improvement for superpermutations is to identify "redundant" overlaps.
                
                # Given the complexity of safe perturbations, let's just restart the search
                # with a slightly different random seed or break if we've spent too much time.
                # However, since we are in a "stuck" state, simple random swaps are risky.
                
                # Let's try a "local swap" that preserves validity?
                # Too complex.
                
                # Let's just break the loop if we are stuck and close to deadline.
                # The current program already does decent removals.
                # The main gain comes from the initial greedy removal pass.
                
                # To avoid infinite loops of failed passes, we break.
                break
                
    # Final check: Ensure the returned string is valid.
    # If the local search somehow corrupted the string (shouldn't happen with valid removals),
    # we fall back to the original.
    if not is_valid(best_S, n):
        return "".join(map(str, word))
        
    return "".join(map(str, best_S))
=======
    # Local Search Strategy with Insertion-based perturbation:
    # Phase 1: Greedy removals from right to left
    # Phase 2: When stuck, try insertions to enable new removals (escape local minima)
    # Phase 3: When stuck, try adjacent swaps to restructure
    
    # We track the number of consecutive failed passes to decide when to perturb
    failed_passes = 0
    max_failed_passes_before_perturb = 2
    insert_attempts = 0
    max_insert_attempts = 50000
    
    # Pre-compute factorial for validity check
    target_perms = math.factorial(n)
    
    # Helper: try to remove character at index i, return True if successful
    def try_remove(S, counts, i, n):
        """Try to remove S[i]. Returns True if removal was performed."""
        # Identify windows affected by removing S[i]
        start_old = max(0, i - n + 1)
        end_old = min(i, len(S) - n)
        
        old_perms_to_dec = []
        for j in range(start_old, end_old + 1):
            p = tuple(S[j:j+n])
            if counts.get(p, 0) == 1:
                return False
            old_perms_to_dec.append(p)
        
        # Decrement counts for old windows
        for p in old_perms_to_dec:
            counts[p] -= 1
            if counts[p] == 0:
                del counts[p]
        
        # Identify new windows created by the removal
        start_new = max(0, i - n + 1)
        end_new = min(i - 1, (len(S) - 1) - n)
        
        for k in range(start_new, end_new + 1):
            window_chars = []
            for idx in range(k, k+n):
                s_idx = idx if idx < i else idx + 1
                window_chars.append(S[s_idx])
            p_new = tuple(window_chars)
            counts[p_new] = counts.get(p_new, 0) + 1
        
        # Commit removal
        S.pop(i)
        return True
    
    # Helper: try to insert character c at position pos, return True if successful
    def try_insert(S, counts, pos, c, n):
        """Try to insert c at position pos. Returns True if insertion was performed."""
        # This is always valid for our purposes since we're just adding characters
        # We'll track it and try to remove later
        S.insert(pos, c)
        
        # Update counts for windows affected by the insertion
        # New length L' = len(S) - 1 (before insertion) + 1 = len(S)
        # Windows that now exist due to insertion:
        # For window start k where the window includes the new character at pos:
        # k <= pos <= k+n-1  =>  pos-n+1 <= k <= pos
        # Also 0 <= k <= len(S)-n
        
        start_new = max(0, pos - n + 1)
        end_new = min(pos, len(S) - n)
        
        for k in range(start_new, end_new + 1):
            p = tuple(S[k:k+n])
            counts[p] = counts.get(p, 0) + 1
        
        return True
    
    # Main search loop
    while time.time() < deadline:
        # Phase 1: Attempt a full pass of removals from right to left
        removed_any = False
        i = len(S) - 1
        
        while i >= 0:
            if time.time() > deadline:
                break
            
            if try_remove(S, counts, i, n):
                removed_any = True
                best_S = S[:]
                best_len = len(S)
                # Do not decrement i - check the new element at position i
            else:
                i -= 1
        
        if removed_any:
            failed_passes = 0
            insert_attempts = 0
            continue
        
        # Stuck - try perturbation strategies
        failed_passes += 1
        
        if time.time() > deadline - 3.0:
            break
        
        if failed_passes >= max_failed_passes_before_perturb:
            # Strategy 1: Try random insertions to break local minimum
            # Insert a random character at a random position, then try to remove
            # This effectively performs a "move" operation
            
            success = False
            for _ in range(min(1000, max(0, deadline - time.time() - 1.0) * 100)):
                if time.time() > deadline - 2.0:
                    break
                
                # Pick a random position and character
                pos = random.randint(0, len(S) - 1)
                c = random.randint(1, n)
                
                # Save state before insertion
                old_len = len(S)
                
                # Perform insertion
                try_insert(S, counts, pos, c, n)
                
                # Now try to remove some characters to get back to a shorter or same length
                removals = 0
                for _ in range(5):
                    if time.time() > deadline - 2.0:
                        break
                    # Try removing the inserted character or nearby characters
                    remove_pos = random.randint(max(0, pos-1), min(len(S)-1, pos+1))
                    if try_remove(S, counts, remove_pos, n):
                        removals += 1
                        if len(S) < old_len:
                            best_S = S[:]
                            best_len = len(S)
                            success = True
                            break
                        elif len(S) == old_len:
                            # Same length but different structure - might enable new removals
                            best_S = S[:]
                            success = True
                            break
                
                if success:
                    break
                # If we couldn't improve, the insertion is still in place
                # Try to remove the inserted character to restore original state
                # Find where the inserted character is now (it may have shifted)
                # For simplicity, just continue - the next pass will handle it
            
            if success:
                failed_passes = 0
                continue
            
            # Strategy 2: Try adjacent swaps
            swap_success = False
            for _ in range(500):
                if time.time() > deadline - 2.0:
                    break
                
                if len(S) < 2:
                    break
                
                # Pick a random adjacent pair to swap
                i_swap = random.randint(0, len(S) - 2)
                
                # Save state
                old_S = S[:]
                old_counts = counts.copy()
                
                # Perform swap
                S[i_swap], S[i_swap + 1] = S[i_swap + 1], S[i_swap]
                
                # Rebuild counts (expensive but safe)
                counts_new = {}
                for j in range(len(S) - n + 1):
                    p = tuple(S[j:j+n])
                    counts_new[p] = counts_new.get(p, 0) + 1
                
                # Check if this swap enables any removals
                can_remove = False
                for j in range(len(S) - 1, -1, -1):
                    start_old = max(0, j - n + 1)
                    end_old = min(j, len(S) - n)
                    valid = True
                    for k in range(start_old, end_old + 1):
                        if counts_new.get(tuple(S[k:k+n]), 0) == 1:
                            valid = False
                            break
                    if valid:
                        can_remove = True
                        break
                
                if can_remove:
                    # Accept this swap
                    counts = counts_new
                    best_S = S[:]
                    best_len = len(S)
                    swap_success = True
                    break
                else:
                    # Revert swap
                    S = old_S
                    counts = old_counts
            
            if swap_success:
                failed_passes = 0
                continue
            
            # If all perturbation strategies failed, break
            break
    
    # Final check: Ensure the returned string is valid.
    if not is_valid(best_S, n):
        return "".join(map(str, word))
        
    return "".join(map(str, best_S))
>>>>>>> REPLACE
