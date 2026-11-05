import itertools
import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 145
    
    # 1. Generate standard superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    # 2. Optimization: Optimized Local Deletion Search
    all_perms = set(itertools.permutations(range(1, n + 1)))
    num_perms = len(all_perms)
    
    # Map permutations to indices for fast counting
    perm_list = list(all_perms)
    perm_to_idx = {p: i for i, p in enumerate(perm_list)}
    
    def get_counts(w):
        counts = [0] * num_perms
        for i in range(len(w) - n + 1):
            p = tuple(w[i:i+n])
            if p in perm_to_idx:
                counts[perm_to_idx[p]] += 1
        return counts

    counts = get_counts(word)
    
    def is_valid(counts):
        # Check if all counts are > 0
        for c in counts:
            if c == 0:
                return False
        return True
        
    if not is_valid(counts):
        return "".join(map(str, word))
        
    improved = True
    while improved:
        if time.time() > deadline - 10:
            break
        improved = False
        L = len(word)
        # Scan from right to left
        # We skip the last n-1 chars as they cannot be deleted without breaking the last window?
        # Actually, we can delete any char, but we must ensure validity.
        # We iterate i from L-2 down to 1.
        for i in range(L - 2, 0, -1):
            if time.time() > deadline - 5:
                return "".join(map(str, word))
            
            # Determine affected windows
            # Removed windows: starts j such that j <= i < j+n
            # j in [i-n+1, i]
            j_start_rem = max(0, i - n + 1)
            j_end_rem = min(i, L - n)
            
            # Check validity: no removed window can be the unique occurrence of its perm
            valid_deletion = True
            removed_perms = []
            for j in range(j_start_rem, j_end_rem + 1):
                p = tuple(word[j:j+n])
                idx = perm_to_idx.get(p, -1)
                if idx != -1:
                    if counts[idx] == 1:
                        valid_deletion = False
                        break
                    removed_perms.append((j, idx))
            
            if not valid_deletion:
                continue
            
            # If valid, perform deletion
            # 1. Update word
            word.pop(i)
            improved = True
            L = L - 1
            
            # 2. Update counts
            # Decrement removed
            for j, idx in removed_perms:
                counts[idx] -= 1
            
            # Increment new windows
            # New windows are those that span the gap.
            # In the new word, these are at indices j such that the window covers the gap.
            # The gap is at index i in the new word? No, the gap is closed.
            # The new windows are $W'_j$ where $j < i$ and $j+n > i$ (in old coordinates? No, new coordinates).
            # Let's find the new windows in the new word.
            # A window $W'_j$ is "new" if it is not just a shift of an old window.
            # $W'_j = W[j:i] + W[i+1:j+n+1]$ (old indices).
            # This happens if $j < i$ and $j+n > i$.
            # So $j \in [i-n+1, i-1]$.
            # Also $j \ge 0$ and $j+n \le L$ (new length).
            j_start_new = max(0, i - n + 1)
            j_end_new = min(i - 1, L - n)
            
            for j in range(j_start_new, j_end_new + 1):
                # Construct the new window
                # It consists of word[j:i] and word[i:j+n] (in new word)
                # Note: in new word, the part after i is shifted.
                # word_new[j:i] == word_old[j:i]
                # word_new[i:j+n] == word_old[i+1:j+n+1]
                new_window = tuple(word[j:i] + word[i:j+n])
                idx = perm_to_idx.get(new_window, -1)
                if idx != -1:
                    counts[idx] += 1

    return "".join(map(str, word))
