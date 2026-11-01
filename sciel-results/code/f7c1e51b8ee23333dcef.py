import itertools
import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # 1. Generate the standard superpermutation
    # This is the recursive construction: L(n) = n! + (n-1)! + ... + 1!
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract permutations of length m-1 from the current word
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap with the end of new_word
            overlap = 0
            max_overlap = min(len(new_word), len(block))
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        word = new_word
    
    # For n <= 4, this is optimal, no need to search
    if n <= 4:
        return "".join(map(str, word))
    
    # 2. Local Search: Greedy Deletion
    # Map permutations to IDs for fast counting
    all_perms = list(itertools.permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    def get_counts(w):
        counts = [0] * num_perms
        L = len(w)
        for i in range(L - n + 1):
            p = tuple(w[i:i+n])
            if p in perm_to_id:
                counts[perm_to_id[p]] += 1
            # Note: In a valid superpermutation, every window of length n 
            # that is a permutation is counted. 
            # The standard construction ensures all windows are permutations?
            # No, some windows might repeat symbols? 
            # Actually, the standard construction for superpermutations 
            # ensures that every permutation appears. 
            # It does NOT ensure that every window of length n is a permutation.
            # But we only care about the permutations.
            # So we just increment if it is a permutation.
        return counts

    # Initial counts
    counts = get_counts(word)
    
    # Verify initial validity
    if any(c == 0 for c in counts):
        # Fallback to standard if something went wrong
        return "".join(map(str, word))

    # Greedy deletion loop
    # We iterate until no single character can be removed.
    # To avoid getting stuck in a bad local minimum too early, 
    # we could randomize the order of checking indices, 
    # but deterministic left-to-right is safer for reproducibility.
    
    while time.time() < deadline:
        improved = False
        L = len(word)
        # Check each index
        for i in range(L):
            if time.time() > deadline:
                break
                
            # Check if deleting word[i] is valid
            # Windows affected are those starting at s where s <= i < s+n
            # i.e., s in [i-n+1, i]
            # We need to ensure that for every permutation p that appears in these windows,
            # its total count remains >= 1 after removing the occurrences in these windows.
            
            # Identify unique permutations in the affected windows
            # and how many times they appear in them.
            affected = {}
            for s in range(max(0, i - n + 1), min(i, L - n) + 1):
                p = tuple(word[s:s+n])
                if p in perm_to_id:
                    pid = perm_to_id[p]
                    affected[pid] = affected.get(pid, 0) + 1
            
            # Check validity
            valid = True
            for pid, cnt_in_window in affected.items():
                if counts[pid] - cnt_in_window < 1:
                    valid = False
                    break
            
            if valid:
                # Perform deletion
                word.pop(i)
                # Update counts
                # Instead of full recompute, we could update incrementally,
                # but full recompute is O(L) and L is small (~6000).
                # Let's recompute for simplicity and safety.
                counts = get_counts(word)
                improved = True
                # Restart the scan from the beginning as indices have shifted
                break
        
        if not improved:
            break

    return "".join(map(str, word))
