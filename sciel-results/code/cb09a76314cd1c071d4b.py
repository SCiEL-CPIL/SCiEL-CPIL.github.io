import itertools

def construct(n):
    # Step 1: Generate the standard recursive superpermutation
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
        
    # Step 2: Compress using greedy deletion with incremental validity check
    if n <= 1:
        return "".join(map(str, word))
        
    # Precompute permutation IDs
    all_perms = list(itertools.permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    counts = [0] * num_perms
    unique_count = 0
    
    # Initialize counts for the initial word
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i + n])
        if w in perm_to_id:
            pid = perm_to_id[w]
            if counts[pid] == 0:
                unique_count += 1
            counts[pid] += 1
            
    # Greedy Deletion
    # We iterate until no more deletions are possible
    # To avoid O(L^2) in the worst case if we don't break, we break on first success
    # and restart the scan.
    
    while True:
        deleted = False
        L = len(word)
        for i in range(L):
            # Identify old windows covering i
            start = max(0, i - n + 1)
            end = min(i, L - n)
            
            old_ids = []
            # Remove old windows
            for j in range(start, end + 1):
                w = tuple(word[j:j + n])
                if w in perm_to_id:
                    pid = perm_to_id[w]
                    old_ids.append(pid)
                    counts[pid] -= 1
                    if counts[pid] == 0:
                        unique_count -= 1
                        
            # Create new word
            new_word = word[:i] + word[i + 1:]
            new_L = L - 1
            
            # Add new windows
            # The windows in new_word that are affected are at indices 
            # roughly corresponding to the old range, but shifted/changed.
            # Specifically, windows starting at j in new_word where j is in [start, end]
            # Note: The index i is removed.
            # Windows ending at i-1 in old word start at i-n.
            # Windows starting at i in old word start at i.
            # In new word, the character at i is gone.
            # The window that was at j in old word (j < i) is now at j in new word, but content changed?
            # No, if j + n - 1 < i, the window is unchanged.
            # If j <= i <= j + n - 1, the window is changed.
            # So we only need to re-evaluate windows that *cover* the deletion point i.
            # In the new word, these windows start at j such that j <= i-1 <= j+n-1?
            # Actually, it's safer to just re-evaluate the range [start, end] in the new word.
            
            new_ids = []
            for j in range(start, min(end, new_L - n) + 1):
                if j < 0: continue
                w = tuple(new_word[j:j + n])
                if w in perm_to_id:
                    pid = perm_to_id[w]
                    new_ids.append(pid)
                    if counts[pid] == 0:
                        unique_count += 1
                    counts[pid] += 1
                        
            # Check validity
            if unique_count == num_perms:
                # Accept deletion
                word = new_word
                deleted = True
                break
            else:
                # Revert
                for pid in new_ids:
                    counts[pid] -= 1
                    if counts[pid] == 0:
                        unique_count -= 1
                for pid in old_ids:
                    counts[pid] += 1
                    if counts[pid] == 1:
                        unique_count += 1
        
        if not deleted:
            break
            
    return "".join(map(str, word))
