import time
import random

def construct(n):
    start_time = time.time()
    # 150s limit. Use 145s for construction + search to be safe.
    deadline = start_time + 145.0

    # For small n, return known optimal constructions
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "12341231423124312"
    
    # Build the standard superpermutation via recursive insertion
    # This produces a valid superpermutation of length sum_{k=1}^n k!
    # For n=7, this is 1+2+6+24+120+720+5040 = 5913.
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        word = new_word
    
    if n <= 4:
        return "".join(map(str, word))
    
    target_count = 1
    for i in range(2, n + 1):
        target_count *= i
        
    current_word = list(word)
    
    # Precompute window counts
    # counts[window] = number of occurrences of this permutation in the word
    def compute_counts(w):
        counts = {}
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i+n])
            if len(set(window)) == n:
                counts[window] = counts.get(window, 0) + 1
        return counts

    perm_count = compute_counts(current_word)
    
    # Optimization Loop
    # We perform local search by removing characters.
    # A character at index i can be removed if no unique permutation is destroyed.
    # Specifically, if a permutation window appears exactly once in the word, 
    # and that window is affected by the removal (i.e., it disappears or changes),
    # we must check if the new word still contains that permutation elsewhere.
    
    # Efficient check:
    # Removing index i affects windows starting at i-n+1 to i.
    # Let's define affected_windows as the set of windows that will be removed or changed.
    # Actually, it's simpler to just check:
    # 1. Identify all windows in the current word that contain index i.
    # 2. For each such window, check if its count is 1.
    # 3. If any such window has count 1, we need to verify that the new word (after removal)
    #    still contains that specific window.
    #    Wait, if a window has count 1 and is removed, it's gone. 
    #    The only way it's still valid is if the "new" windows formed by the removal 
    #    include that same window? No, removing a character shifts things.
    #    If a unique window W is destroyed, the new word must contain W elsewhere.
    #    But W was unique, so it can't be elsewhere.
    #    Therefore, if a unique window is destroyed, the removal is INVALID.
    #    UNLESS the removal creates a new instance of W? 
    #    If W is unique, there is no other instance.
    #    So, if removing i destroys a unique window W, the new word has 0 instances of W?
    #    Not necessarily. The removal might create a new window that IS W.
    #    Example: W = 1234567. Word ...1234567...
    #    Remove 3: ...124567...
    #    Old windows involving 3: 1234567 (if it started there).
    #    New windows: 124567... (starts with 1).
    #    Does 124567... contain 1234567? No.
    #    So generally, if a unique window is destroyed, it's invalid.
    
    # However, we must be careful: "destroyed" means the specific tuple of characters 
    # no longer exists at that position. But it might appear at a NEW position due to shift?
    # No, the shift changes the content of windows.
    # If W is unique in the old word, and we remove a char from W, W is gone from that spot.
    # Can W appear elsewhere? No, it was unique.
    # Can W appear at a new spot created by the removal?
    # The new windows are formed by concatenating parts of the old word.
    # It is theoretically possible for a unique window to be "re-created" or preserved 
    # if the removal doesn't actually break the sequence? 
    # No, removing a char breaks the sequence.
    # So the rule is: **If removing index i causes any permutation window with count 1 to disappear, the move is invalid.**
    
    # Wait, is that sufficient?
    # What if removing i causes a window with count 2 to become count 1? That's fine.
    # What if removing i causes a window with count 1 to be replaced by a DIFFERENT window? 
    # The old window is gone. Since it was unique, it's lost. Invalid.
    
    # So the check is:
    # 1. Find all windows in `current_word` that include index i.
    # 2. For each such window, if `perm_count[window] == 1`, the removal is INVALID.
    # 3. If all such windows have count > 1, the removal is VALID.
    #    (We decrement counts for removed windows, increment for new ones? 
    #     Actually, if we just check validity, we don't need to update counts immediately 
    #     if we revert on failure. But we are committing to the move. 
    #     So we must update counts.)
    
    # Let's refine the update logic.
    # When we remove index i:
    # Old windows affected: start indices j in [i-n+1, i].
    #   For each j, window_old = current_word[j:j+n].
    #   These windows are REMOVED from the count.
    # New windows affected: 
    #   The word shrinks. Indices > i shift left by 1.
    #   New windows are formed at start indices j in [i-n+1, i-1] (in the new word's coordinate system? 
    #   No, let's look at the new word `new_word`).
    #   `new_word` has length L-1.
    #   Windows in `new_word` that include the "gap" (which is now closed) are those 
    #   starting at j such that the window covers the junction.
    #   The junction is between `new_word[i-1]` and `new_word[i]` (which is `current_word[i+1]`).
    #   So the new windows are those starting at j in [i-n+1, i-1] in the NEW word?
    #   Let's map indices.
    #   Old index i is removed.
    #   New index k corresponds to old index k if k < i, and old index k+1 if k >= i.
    #   A window in the new word starting at k covers new indices k..k+n-1.
    #   This window is "new" (i.e., different from old windows) if it spans the junction.
    #   It spans the junction if k <= i-1 and k+n-1 >= i.
    #   So k is in [i-n+1, i-1].
    #   For these k, we compute the window in `new_word` and add to counts.
    
    # Algorithm for one step:
    # 1. Candidate i.
    # 2. Check validity:
    #    For j in range(max(0, i-n+1), min(i, len(current_word)-n) + 1):
    #        w = tuple(current_word[j:j+n])
    #        if len(set(w)) == n and perm_count[w] == 1:
    #            return False (invalid)
    # 3. If valid, perform removal:
    #    - Decrement counts for old windows (j in range...)
    #    - Create new_word
    #    - Increment counts for new windows (k in range...)
    #    - Update current_word = new_word
    
    # This is O(n) per check. 
    # We can try many removals.
    
    # We will iterate through indices in random order or specific orders.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Randomize order to escape local minima / bias
        indices = list(range(len(current_word)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            if i >= len(current_word):
                continue
            
            # 1. Check validity
            valid = True
            old_start = max(0, i - n + 1)
            old_end = min(i, len(current_word) - n)
            
            # Store affected old windows for potential update
            affected_old = []
            
            for j in range(old_start, old_end + 1):
                window = tuple(current_word[j:j+n])
                if len(set(window)) == n:
                    affected_old.append((j, window))
                    if perm_count[window] == 1:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # 2. Perform removal
            new_word = current_word[:i] + current_word[i+1:]
            
            # Update counts for old windows (decrement)
            for j, window in affected_old:
                perm_count[window] -= 1
                if perm_count[window] == 0:
                    del perm_count[window]
            
            # Update counts for new windows (increment)
            # New windows start at k in [i-n+1, i-1] in the NEW word
            # Note: new_word length is len(current_word)-1
            new_start = max(0, i - n + 1)
            new_end = min(i - 1, len(new_word) - n)
            
            for k in range(new_start, new_end + 1):
                new_window = tuple(new_word[k:k+n])
                if len(set(new_window)) == n:
                    perm_count[new_window] = perm_count.get(new_window, 0) + 1
            
            current_word = new_word
            improved = True
            
            # If we got stuck, we might want to try a different strategy or restart?
            # For now, simple greedy removal.
            
    # Final verification
    # Ensure we haven't broken anything
    if time.time() < deadline - 2:
        seen = set()
        for i in range(len(current_word) - n + 1):
            window = tuple(current_word[i:i+n])
            if len(set(window)) == n:
                seen.add(window)
        if len(seen) != target_count:
            # Revert to original if invalid (should not happen with correct logic)
            current_word = list(word)
    
    return "".join(map(str, current_word))
