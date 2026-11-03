import sys
from collections import defaultdict

def construct(n):
    # Classic construction
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
        
    L = len(word)
    if L <= n:
        return "".join(map(str, word))
        
    # Initialize counts
    perm_counts = defaultdict(int)
    window_perms = [None] * (L - n + 1)
    for j in range(L - n + 1):
        p = tuple(word[j:j+n])
        window_perms[j] = p
        perm_counts[p] += 1
        
    # Greedy deletion
    # We iterate i from 0 to L-1
    # If we delete i, we update counts
    deleted = [False] * L
    
    for i in range(L):
        if deleted[i]:
            continue
            
        # Check if deletion is valid
        # 1. Identify windows to remove
        remove_perms = []
        # Windows containing i: start j such that j <= i <= j+n-1
        # j in [i-n+1, i]
        start_min = max(0, i - n + 1)
        start_max = min(L - n, i)
        
        for j in range(start_min, start_max + 1):
            p = window_perms[j]
            remove_perms.append(p)
            
        # 2. Identify windows to add (bridging)
        add_perms = []
        # New windows crossing the gap at i
        # In the new word, the gap is at position i (if we consider 0-indexed new word, 
        # but it's easier to think in terms of original indices)
        # A new window is formed by original indices k..i-1 and i+1..k+n
        # The length must be n.
        # The start of the new window in the original word context is k.
        # The new window consists of word[k:i] + word[i+1:k+n+1]
        # We need k such that the window is valid in the new string.
        # The new string length is L-1.
        # The window starts at k' in new string.
        # If k < i, k' = k.
        # The window ends at k'+n-1 in new string.
        # This corresponds to original indices k..i-1 (length i-k) and i+1..k+n (length n-(i-k))
        # So the original end index is i+1 + (n-(i-k)) - 1 = n+k.
        # So we take word[k:i] + word[i+1:k+n+1]
        
        # What range of k?
        # The new window must exist in the new string.
        # New string length L-1.
        # Max start k' is L-1-n.
        # If k < i, k' = k. So k <= L-1-n.
        # If k >= i, k' = k-1. So k-1 <= L-1-n => k <= L-n.
        
        # Also, the window must cross the gap.
        # This means it includes indices < i and > i.
        # So k < i and k+n >= i+1 => k >= i-n+1.
        
        k_min = max(0, i - n + 1)
        k_max = min(L - n, i - 1) # k must be < i
        
        for k in range(k_min, k_max + 1):
            # Construct new perm
            # word[k:i] + word[i+1:k+n+1]
            # Check bounds
            if k+n+1 > L:
                continue
            new_p = tuple(word[k:i] + word[i+1:k+n+1])
            add_perms.append(new_p)
            
        # 3. Check validity
        valid = True
        # Apply removals
        for p in remove_perms:
            perm_counts[p] -= 1
            if perm_counts[p] < 0:
                valid = False
                break
        if valid:
            # Apply additions
            for p in add_perms:
                perm_counts[p] += 1
            
            # Check if any count is 0
            # We only need to check the ones we removed
            for p in remove_perms:
                if perm_counts[p] == 0:
                    valid = False
                    break
        
        if valid:
            deleted[i] = True
            # Keep the updated counts
        else:
            # Revert
            for p in remove_perms:
                perm_counts[p] += 1
            for p in add_perms:
                perm_counts[p] -= 1
            # Do not mark as deleted

    # Construct final word
    final_word = [str(w) for i, w in enumerate(word) if not deleted[i]]
    return "".join(final_word)
