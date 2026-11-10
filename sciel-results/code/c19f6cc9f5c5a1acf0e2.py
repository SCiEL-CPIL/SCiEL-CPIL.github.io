import time
import itertools
from collections import defaultdict

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    
    start_time = time.time()
    deadline = start_time + 115  # safety margin

    # 1. Build standard superpermutation
    def _build_superperm(n):
        word = [1]
        for m in range(2, n + 1):
            # Extract all distinct (m-1)-permutations in order of appearance
            perms = []
            seen = set()
            for i in range(len(word) - (m - 1) + 1):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            
            # Build new word by inserting m into each permutation
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                if not new:
                    new = block
                else:
                    # Find maximum overlap
                    max_ov = min(len(new), len(block))
                    ov = 0
                    for o in range(max_ov, 0, -1):
                        if new[-o:] == block[:o]:
                            ov = o
                            break
                    new.extend(block[ov:])
            word = new
        return word

    word = _build_superperm(n)
    
    # 2. Local Search: Greedy Deletion
    # We represent the word as a list of integers
    # We maintain counts of all permutations
    
    L = len(word)
    if L < n:
        return "".join(map(str, word))
        
    # Initial count of permutations
    perm_counts = defaultdict(int)
    for i in range(L - n + 1):
        p = tuple(word[i:i+n])
        perm_counts[p] += 1
        
    # Total number of unique permutations should be n!
    # We don't strictly need to verify this if the generator is correct, 
    # but it's good for sanity.
    
    # Function to check if index i can be removed
    def can_remove(i):
        # Windows that will be destroyed are those covering i
        # Start indices: max(0, i-n+1) to min(i, L-n)
        start_min = max(0, i - n + 1)
        start_max = min(i, L - n)
        
        for start in range(start_min, start_max + 1):
            p = tuple(word[start:start+n])
            if perm_counts[p] == 1:
                return False
        return True

    # Function to perform removal at index i
    def remove_at(i):
        nonlocal L
        # Identify destroyed windows
        start_min = max(0, i - n + 1)
        start_max = min(i, L - n)
        destroyed = []
        for start in range(start_min, start_max + 1):
            p = tuple(word[start:start+n])
            destroyed.append(p)
            perm_counts[p] -= 1
            
        # Remove the character
        word.pop(i)
        L -= 1
        
        # Identify new windows created by the merge
        # The gap is at i. 
        # New windows are those that span the gap.
        # In the new list, the character at i-1 is now adjacent to i (which was i+1).
        # A window of length n spanning the gap must include the new junction.
        # The junction is between index i-1 and i in the new list.
        # Possible start indices for windows covering the junction:
        # The junction is at position i-1 (end of left part) and i (start of right part).
        # A window [j, j+n-1] covers the junction if j <= i-1 and j+n-1 >= i.
        # So i - n + 1 <= j <= i - 1.
        # Also j >= 0 and j <= L - n.
        
        new_start_min = max(0, i - n + 1)
        new_start_max = min(i - 1, L - n)
        
        for start in range(new_start_min, new_start_max + 1):
            p = tuple(word[start:start+n])
            perm_counts[p] += 1

    # Iterate until no more removals
    while True:
        if time.time() > deadline:
            break
            
        removed_any = False
        i = 0
        while i < L:
            if can_remove(i):
                remove_at(i)
                removed_any = True
                # After removal, the element at i is the one that was at i+1.
                # We should check the new element at i again? 
                # Or just continue to i+1? 
                # If we don't check i again, we might miss a removal if the new element at i 
                # combined with previous context allows removal.
                # However, usually we scan left to right. 
                # Let's restart the scan for simplicity and correctness, 
                # as L is small enough.
                break
            else:
                i += 1
                
        if not removed_any:
            break
            
    return "".join(map(str, word))
