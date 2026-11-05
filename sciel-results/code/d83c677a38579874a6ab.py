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
    w = word[:]
    
    # Pre-compute all permutations
    all_perms = set(itertools.permutations(range(1, n+1)))
    total_perms = len(all_perms)
    
    # Build a helper to get the window at a given start index
    def get_window(start, w_list, n_val):
        if start < 0 or start + n_val > len(w_list):
            return None
        return tuple(w_list[start:start+n_val])
    
    # Count occurrences of each permutation in the current word
    perm_counts = {}
    for i in range(len(w) - n + 1):
        p = tuple(w[i:i+n])
        if p in all_perms:
            perm_counts[p] = perm_counts.get(p, 0) + 1
            
    # Check if all are present (should be)
    if len(perm_counts) < total_perms:
        return "".join(map(str, w))
        
    # Strategy: Try to remove characters from the end first (safest), 
    # then try removing from various positions.
    # Key insight: When we remove a character at position i, the windows affected are:
    # - Old windows starting at i-n+1 through i (these are destroyed/changed)
    # - New windows may form at the boundary
    # 
    # For simplicity and correctness, let's use a two-pass approach:
    # Pass 1: Remove from the end as much as possible
    # Pass 2: Try removing from middle positions (left to right)
    
    # Pass 1: Remove from end
    while len(w) > n:
        if time.time() > deadline:
            break
        last_start = len(w) - n
        p = tuple(w[last_start:])
        if p in all_perms and perm_counts.get(p, 0) == 1:
            break
        else:
            if p in all_perms:
                perm_counts[p] -= 1
                if perm_counts[p] == 0:
                    del perm_counts[p]
            w.pop()
    
    # Pass 2: Try removing from left to right
    # For each position i from 0 to len(w)-n, try to remove w[i]
    # We need to be careful about maintaining perm_counts consistency
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(w) - n + 1):
            if time.time() > deadline:
                break
            
            # Check if removing w[i] is safe
            # Windows that include w[i]: start indices from max(0, i-n+1) to min(i, len(w)-n)
            removable = True
            affected_starts = list(range(max(0, i-n+1), min(i, len(w)-n) + 1))
            
            for start in affected_starts:
                p = tuple(w[start:start+n])
                if p in all_perms:
                    if perm_counts.get(p, 0) == 1:
                        removable = False
                        break
            
            if removable:
                # Remove w[i] and update counts
                # First, decrement counts for old affected windows
                for start in affected_starts:
                    p = tuple(w[start:start+n])
                    if p in all_perms:
                        perm_counts[p] -= 1
                        if perm_counts[p] == 0:
                            del perm_counts[p]
                
                # Remove the character
                w.pop(i)
                
                # After removal, new windows may form. The windows starting at 
                # positions affected_starts (in the new string) need to be added back.
                # In the new string, the positions are the same for starts < i, 
                # but for starts >= i, they shift. Actually, since we removed position i,
                # windows starting at positions in affected_starts (which are all < i or == i)
                # now have different content. We need to add the new windows.
                
                # The new affected windows in the new string are at the same start positions
                # (since we removed position i, and all affected starts are <= i, 
                # the new windows at those starts are different)
                for start in affected_starts:
                    if start + n <= len(w):  # Make sure the window fits
                        p = tuple(w[start:start+n])
                        if p in all_perms:
                            perm_counts[p] = perm_counts.get(p, 0) + 1
                
                changed = True
                break  # Restart the scan from the beginning
    
    return "".join(map(str, w))
