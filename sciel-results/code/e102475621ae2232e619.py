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
        
    # Greedy removal with correct count updates
    # We will scan from left to right.
    # When we remove w[i], we update counts for the affected windows.
    # Affected old windows: starts i-n+1 ... i
    # Affected new windows: starts i-n+1 ... i-1 (in the new string, these windows are shifted/changed)
    
    i = 0
    while i < len(w) - n + 1:
        if time.time() > deadline:
            break
            
        # Check if w[i] is removable
        # A window is "critical" if its count is 1.
        # If any window containing w[i] is critical, we cannot remove w[i].
        removable = True
        for start in range(max(0, i-n+1), min(i+1, len(w)-n+1)):
            p = tuple(w[start:start+n])
            if p in all_perms:
                if perm_counts.get(p, 0) == 1:
                    removable = False
                    break
        
        if removable:
            # 1. Decrement counts for old windows containing w[i]
            # These are starts: max(0, i-n+1) to min(i, len(w)-n+1)
            old_starts = []
            for start in range(max(0, i-n+1), min(i+1, len(w)-n+1)):
                old_starts.append(start)
                p = tuple(w[start:start+n])
                if p in all_perms:
                    perm_counts[p] -= 1
                    if perm_counts[p] == 0:
                        del perm_counts[p]
            
            # 2. Remove w[i]
            w.pop(i)
            
            # 3. Increment counts for new windows
            # The windows that are "new" or "changed" in the new string are those starting at i-n+1 to i-1.
            # In the new string, the window starting at `start` (where start < i) is:
            # new_w[start : start+n]
            # This window was NOT present in the old string at this start index (it was different).
            # Wait, in the old string, the window starting at `start` was `w_old[start:start+n]`.
            # In the new string, the window starting at `start` is `w_new[start:start+n]`.
            # Since we removed `w[i]`, and `start < i`, the sequence `w_new[start:start+n]` consists of:
            # `w_old[start:i]` + `w_old[i+1:i+1+n-(i-start)]`
            # The old window at `start` was `w_old[start:i]` + `w_old[i:i+n-(i-start)]`.
            # So they are different.
            # We need to add the counts for these new windows.
            
            # Which starts should we check?
            # The starts are from max(0, i-n+1) to i-1.
            # Note: if i-n+1 < 0, we start at 0.
            # Also, we only care about windows that are valid permutations (in all_perms).
            
            new_starts = range(max(0, i-n+1), i)
            for start in new_starts:
                # Ensure start + n <= len(w) (new length)
                if start + n <= len(w):
                    p = tuple(w[start:start+n])
                    if p in all_perms:
                        perm_counts[p] = perm_counts.get(p, 0) + 1
        else:
            i += 1
            
    # "Remove from End" pass
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
