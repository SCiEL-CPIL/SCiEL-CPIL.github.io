import time
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

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
            # Find max overlap
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    if n < 3:
        return "".join(map(str, word))

    # 2. Optimization via Greedy Deletion of Non-Essential Characters
    # We repeatedly remove characters that are not part of any "unique" permutation window.
    # A window is unique if that specific permutation appears exactly once in the current word.
    # If a character is in a unique window, it is essential for that permutation.
    # If a character is NOT in ANY unique window, it is safe to remove.
    
    while time.time() < deadline:
        L = len(word)
        if L < n:
            break
            
        # Count occurrences of each permutation
        perm_indices = {}
        
        # Step 1: Store indices for each permutation encountered.
        for i in range(L - n + 1):
            window = tuple(word[i:i+n])
            if len(set(window)) == n:
                if window in perm_indices:
                    perm_indices[window].append(i)
                else:
                    perm_indices[window] = [i]
                    
        # Step 2: Mark essential indices for unique perms
        essential_indices = [False] * L
        any_essential = False
        for perm, indices in perm_indices.items():
            if len(indices) == 1:
                idx = indices[0]
                any_essential = True
                for k in range(n):
                    essential_indices[idx + k] = True
        
        if not any_essential:
            # If no essential indices, the whole word is redundant? 
            # This shouldn't happen for a valid superperm unless it's very short.
            # Actually, if no perm is unique, it means every perm appears >= 2 times.
            # Then we can remove anything? No, we need to be careful.
            # But if every perm is covered by >= 2 windows, removing one char might drop some to 1.
            # The "no unique window" case means we can pick any char? 
            # Actually, if no window is unique, we can remove the first char and re-evaluate.
            # But let's just break if we can't find a "safe" non-essential char.
            # Wait, if essential_indices are all False, then ALL indices are non-essential.
            # So we can remove word[0].
            word = word[1:]
            continue

        # Find a non-essential index and remove it
        # Strategy: Try removing from the end first, as the tail of standard superperms is often redundant
        removed = False
        for i in range(L - 1, -1, -1):
            if not essential_indices[i]:
                word = word[:i] + word[i+1:]
                removed = True
                break
        
        if not removed:
            # No non-essential indices found. 
            # This means every index is part of at least one unique window.
            # We are stuck with this greedy heuristic. 
            # We could try a more complex local search (swaps, etc.) but time is limited.
            # For now, break.
            break

    return "".join(map(str, word))
