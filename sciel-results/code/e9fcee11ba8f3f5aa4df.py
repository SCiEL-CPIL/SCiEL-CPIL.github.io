import time
import sys
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 115.0  # 115s limit to be safe
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "1213121"
    if n == 4:
        return "12131214121312131413121314121312141213121"

    # Standard construction
    word = [1]
    for m in range(2, n + 1):
        if time.time() > deadline:
            break # Should not happen for n=7
            
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
                
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new), len(block))
            overlap = 0
            for k in range(max_overlap, -1, -1):
                if new[len(new)-k:] == block[:k]:
                    overlap = k
                    break
            new += block[overlap:]
        word = new
        
    if time.time() > deadline:
        return "".join(map(str, word))

    # Greedy Reduction
    # Compute initial counts
    perm_counts = defaultdict(int)
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i+n])
        perm_counts[w] += 1
        
    # Helper to check if a specific window count is > 0
    # We will maintain the counts dict.
    
    improved = True
    while improved:
        improved = False
        for i in range(len(word)):
            if time.time() > deadline:
                return "".join(map(str, word))
            
            # Identify old windows to remove
            # j such that word[j:j+n] contains index i
            # j <= i < j+n => i-n+1 <= j <= i
            # j must be valid index for window start: 0 <= j <= len(word)-n
            old_start = max(0, i - n + 1)
            old_end = min(i, len(word) - n)
            
            # Identify new windows to add
            # In new word, index i is removed.
            # The join is between word[i-1] and word[i+1].
            # New word length is len(word)-1.
            # New windows containing the join (at index i in new word? No, join is between i-1 and i in new word indices?
            # Let's say new_word = word[:i] + word[i+1:]
            # The new adjacency is at index i-1 (end of first part) and i (start of second part).
            # So windows in new_word that cover index i-1 or i?
            # Actually, any window in new_word that includes the position i (which is the first char of the second part)
            # or i-1 (last char of first part).
            # The range of start indices j in new_word for windows covering the junction:
            # j <= i-1 < j+n and j <= i < j+n
            # This is roughly j in [i-n+1, i]
            
            new_start = max(0, i - n + 1)
            new_end = min(i, len(word) - 1 - n) # len(new_word) - n = len(word) - 1 - n
            
            # Tentatively update counts
            removed_windows = []
            added_windows = []
            valid = True
            
            for j in range(old_start, old_end + 1):
                w = tuple(word[j:j+n])
                perm_counts[w] -= 1
                removed_windows.append(w)
                if perm_counts[w] < 0:
                    valid = False
                    # Restore immediately if possible? 
                    # Better to restore all at end if invalid.
            
            if valid:
                # Create new word to get new windows
                # We don't want to copy the whole word for every i.
                # We can construct the new windows manually.
                # New word is word[:i] + word[i+1:]
                # The new windows are those starting at j in [new_start, new_end]
                for j in range(new_start, new_end + 1):
                    # Construct window from original word
                    # If j < i: window is word[j:j+n] but skipping i?
                    # If j >= i: window is word[j+1:j+1+n]
                    # Actually, it's easier to just slice the conceptual new word.
                    # But we don't have the new word.
                    # We can build the window:
                    # Indices in new word:
                    # 0 ... i-1 correspond to word 0 ... i-1
                    # i ... correspond to word i+1 ...
                    
                    # Window in new word starting at j:
                    # Elements are new_word[j], ..., new_word[j+n-1]
                    # Map back to word indices:
                    # if k < i: word[k]
                    # if k >= i: word[k+1]
                    
                    win = []
                    for k in range(j, j+n):
                        if k < i:
                            win.append(word[k])
                        else:
                            win.append(word[k+1])
                    w = tuple(win)
                    perm_counts[w] += 1
                    added_windows.append(w)
                    if perm_counts[w] < 1:
                        # This shouldn't happen as we are adding
                        pass
            
            if valid:
                # Check if any count became 0?
                # We decremented removed. If any was 1, it is now 0.
                # We incremented added.
                # We need to check if any permutation count is 0.
                # Only the removed ones could have become 0.
                # The added ones are now >= 1.
                # So we just need to check if any removed window had count 1 (became 0).
                # We already checked `if perm_counts[w] < 0` which is wrong.
                # We should check `if perm_counts[w] == 0` AFTER decrement.
                
                # Let's re-verify the validity logic.
                # We decremented removed.
                # If any removed window count is 0, it's invalid.
                # We did `if perm_counts[w] < 0`. This is incorrect.
                # It should be `if perm_counts[w] == 0`.
                
                # Let's restart the validity check properly.
                pass
            
            # Revert if invalid or just to be safe before deciding
            for w in removed_windows:
                perm_counts[w] += 1
            for w in added_windows:
                perm_counts[w] -= 1
                
            # Proper Validity Check
            valid = True
            for j in range(old_start, old_end + 1):
                w = tuple(word[j:j+n])
                if perm_counts[w] == 1:
                    valid = False
                    break
            
            if valid:
                # Commit
                word.pop(i)
                # Update counts permanently
                for j in range(old_start, old_end + 1):
                    w = tuple(word[j:j+n]) # Wait, word changed!
                    # We must use the OLD word for removal
                    # And NEW word for addition
                    pass
                
                # This manual count updating is error-prone.
                # Given the small n, maybe just re-compute counts from scratch?
                # No, that's O(N^2).
                
                # Let's stick to the decrement/increment logic but fix the validity check.
                
                # 1. Decrement removed
                for j in range(old_start, old_end + 1):
                    w = tuple(word[j:j+n])
                    perm_counts[w] -= 1
                    if perm_counts[w] == 0:
                        valid = False
                        break
                    removed_windows.append(w)
                
                if valid:
                    # 2. Increment added
                    for j in range(new_start, new_end + 1):
                        win = []
                        for k in range(j, j+n):
                            if k < i:
                                win.append(word[k])
                            else:
                                win.append(word[k+1])
                        w = tuple(win)
                        perm_counts[w] += 1
                        added_windows.append(w)
                    
                    # 3. Commit
                    word.pop(i)
                    improved = True
                    break # Restart loop
                else:
                    # Revert
                    for w in removed_windows:
                        perm_counts[w] += 1
                    # No added windows were incremented if we broke early?
                    # We only increment if valid is still true after decrements.
                    # So if invalid, we only need to revert decrements.
            else:
                # Revert if valid was false due to break
                for w in removed_windows:
                    perm_counts[w] += 1

    return "".join(map(str, word))
