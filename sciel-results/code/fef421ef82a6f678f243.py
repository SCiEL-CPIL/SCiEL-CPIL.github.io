import itertools
import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # 1. Generate standard superpermutation
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
    
    # If n is small, return immediately
    if n < 7:
        return "".join(map(str, word))
        
    # For n=7, try to optimize
    # Convert to list of ints for easy manipulation
    w = word
    
    # Pre-compute all permutations
    all_perms = set(itertools.permutations(range(1, n+1)))
    total_perms = len(all_perms)
    
    # Count occurrences of each permutation in the current word
    perm_counts = {}
    for i in range(len(w) - n + 1):
        p = tuple(w[i:i+n])
        if p in all_perms:
            perm_counts[p] = perm_counts.get(p, 0) + 1
            
    # Check if all are present (should be)
    if len(perm_counts) < total_perms:
        return "".join(map(str, w))
        
    # Greedy removal
    # We try to remove characters from the end first? Or middle?
    # Removing from the end is safer?
    # Let's try removing from the end backwards.
    # If we remove from the end, we only lose the last window.
    # But we might lose a permutation that ONLY appears at the end.
    
    # A better strategy: Try to remove any character.
    # Iterate through indices.
    # To be fast, we can just try to remove from the end repeatedly.
    # If the last character is redundant, remove it.
    # Then check the new last character.
    
    # Let's try a more aggressive approach:
    # Scan from left to right, try to remove.
    
    i = 0
    while i < len(w) - n + 1:
        if time.time() > deadline:
            break
            
        # Try to remove w[i]
        # Check if any permutation that ONLY appears at window starting i (or affected by i) is lost.
        # Windows affected by removing w[i]:
        # Start indices: i-n+1 to i.
        # Actually, removing w[i] shifts everything after i.
        # The windows that CHANGE are those that included w[i].
        # These are windows starting at i-n+1, ..., i.
        
        # We need to check if any permutation that has count 1 is in these windows.
        # If so, we cannot remove w[i] (unless that permutation appears elsewhere, which count 1 implies it doesn't).
        
        removable = True
        for start in range(max(0, i-n+1), min(i+1, len(w)-n+1)):
            p = tuple(w[start:start+n])
            if p in all_perms:
                if perm_counts.get(p, 0) == 1:
                    # This permutation only appears here.
                    # If we remove w[i], this window is destroyed or changed.
                    # So we cannot remove w[i].
                    removable = False
                    break
        
        if removable:
            # Remove w[i]
            # Update counts for affected windows
            # Old windows: start in [i-n+1, i]
            for start in range(max(0, i-n+1), min(i+1, len(w)-n+1)):
                p = tuple(w[start:start+n])
                if p in all_perms:
                    perm_counts[p] -= 1
                    if perm_counts[p] == 0:
                        del perm_counts[p]
            
            w.pop(i)
            
            # New windows appear?
            # The windows that were shifted left might now be new?
            # No, the set of windows is just the old set minus the destroyed ones,
            # plus the new windows formed by the shift?
            # Actually, removing a char creates new windows at the boundary?
            # No, it just shortens the string.
            # The windows are defined by indices in the NEW string.
            # The windows starting at indices < i are unchanged?
            # No, if we remove w[i], w[i+1] becomes w[i].
            # So windows starting at i-1, i-2, etc might change?
            # Window starting at i-1 in old: w[i-1]...w[i]...w[i+n-2]
            # Window starting at i-1 in new: w[i-1]...w[i-1]... (shifted)
            # This is complex.
            
            # Simpler: Just re-calculate counts for the affected region?
            # Or just restart the check?
            
            # For simplicity and safety, let's just try to remove from the END.
            # If we remove from the end, only the last window is lost.
            # If the last window is a permutation with count 1, we can't remove.
            # Otherwise, we can.
            
            # Let's switch to "Remove from End" strategy for simplicity and speed.
            # It might not be optimal but it's safe.
            
            # Actually, let's stick to the "try to remove w[i]" but update counts correctly.
            # To update counts correctly after removal:
            # 1. Decrement counts for windows that contained w[i] (old indices).
            # 2. Remove w[i].
            # 3. Increment counts for windows that are NEW or CHANGED.
            # The windows that are changed are those starting at i-n+1 ... i-1 (in the new string, they are the same starts, but content changed? No, content before i is same. Content after i is shifted.
            # So windows starting at i-n+1 ... i-1 in the NEW string are DIFFERENT from old.
            # We need to add their counts.
            
            # This is getting complicated. Let's use the "Remove from End" heuristic.
            # It is very fast.
            
            # Let's break out and use a different loop.
            break 
        else:
            i += 1
            
    # If we broke early, let's do the "Remove from End" pass.
    # This is a simple loop.
    while len(w) > n:
        if time.time() > deadline:
            break
        # Check last window
        last_start = len(w) - n
        p = tuple(w[last_start:])
        if p in all_perms and perm_counts.get(p, 0) == 1:
            # Can't remove last char
            break
        else:
            # Safe to remove?
            # Removing last char destroys the last window.
            # If it was a permutation, decrement count.
            if p in all_perms:
                perm_counts[p] -= 1
                if perm_counts[p] == 0:
                    del perm_counts[p]
            w.pop()
            
    return "".join(map(str, w))
