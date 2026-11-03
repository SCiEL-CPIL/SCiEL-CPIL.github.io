import time
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 115 # Safety margin

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
            best_overlap = 0
            max_check = min(len(new), len(block))
            for o in range(max_check, 0, -1):
                if new[-o:] == block[:o]:
                    best_overlap = o
                    break
            new += block[best_overlap:]
        word = new

    if n <= 6:
        return "".join(map(str, word))

    # 2. Setup for local search
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    # Precompute window IDs for the current word
    L = len(word)
    num_windows = L - n + 1
    window_ids = [0] * num_windows
    counts = [0] * num_perms
    
    for i in range(num_windows):
        p = tuple(word[i:i+n])
        pid = perm_to_id.get(p, -1)
        window_ids[i] = pid
        if pid >= 0:
            counts[pid] += 1
            
    # Track zero counts for fast validity check
    zero_count = sum(1 for c in counts if c == 0)
    
    if zero_count > 0:
        return "".join(map(str, word))

    # Local Search: Deletion with fast validity check
    def try_delete(i):
        """
        Attempts to delete word[i]. Returns True if successful.
        Modifies word and counts if successful.
        """
        nonlocal zero_count
        L = len(word)
        if L == n:
            return False
        
        # List of (pid, delta) to apply
        updates = []
        
        # 1. Decrement old windows
        # Windows affected: start k in [i-n+1, i]
        k_start = max(0, i - n + 1)
        k_end = min(i + 1, L - n + 1)
        for k in range(k_start, k_end):
            pid = window_ids[k]
            if pid >= 0:
                # Pre-check: if count is 1, it will become 0
                if counts[pid] == 1:
                    zero_count += 1
                counts[pid] -= 1
                updates.append((pid, -1))
        
        # 2. Increment new windows
        # New windows affected: start k in [i-n+1, i-1] in the NEW word
        # Note: The range of k in the new word is slightly different.
        # Old word length L. New word length L-1.
        # New windows start at k where new_word[k:k+n] is valid.
        # k ranges from 0 to (L-1)-n.
        # The deletion at i affects windows that span the gap.
        # In the new word, these are windows starting at k such that the window includes the position where i was removed.
        # This corresponds to new_word indices k in [i-n+1, i-1] roughly.
        
        nk_start = max(0, i - n + 1)
        nk_end = min(i, L - n)
        for k in range(nk_start, nk_end): # k is start index in NEW word
            if k < i:
                # Window spans the deletion point
                # new_word[k] = word[k] ... word[i-1], word[i+1] ...
                # We need to construct the tuple
                # Optimization: use slicing
                w1 = word[k:i]
                w2 = word[i+1:k+n+1]
                p = tuple(w1 + w2)
            else: # k >= i
                # new_word[k] = word[k+1]
                p = tuple(word[k+1:k+1+n])
                
            pid = perm_to_id.get(p)
            if pid is not None:
                if counts[pid] == 0:
                    zero_count -= 1
                counts[pid] += 1
                updates.append((pid, 1))
        
        # Check validity
        if zero_count == 0:
            word.pop(i)
            # Update window_ids efficiently
            # Windows before i-1 are unchanged
            # Windows from i-n+1 to i-1 need recomputation (they span the gap)
            # Windows after i are shifted left by 1
            new_L = L - 1
            new_num_windows = new_L - n + 1
            
            # Recompute affected windows
            for k in range(max(0, i - n + 1), min(i, new_num_windows)):
                p = tuple(word[k:k+n])
                window_ids[k] = perm_to_id.get(p, -1)
            
            # Shift remaining window_ids
            for k in range(i, new_num_windows):
                window_ids[k] = window_ids[k+1]
            window_ids.pop()
            
            return True
        else:
            # Revert
            for pid, delta in updates:
                counts[pid] -= delta
            zero_count = sum(1 for c in counts if c == 0)
            return False

    # Iterate deletion passes
    improved = True
    while improved:
        improved = False
        # Left to right
        i = 0
        while i < len(word):
            if time.time() > deadline:
                return "".join(map(str, word))
            if try_delete(i):
                improved = True
            else:
                i += 1
        # Right to left
        i = len(word) - 1
        while i >= 0:
            if time.time() > deadline:
                return "".join(map(str, word))
            if try_delete(i):
                improved = True
            else:
                i -= 1

    # 3. Local Search: Swap (Optimized)
    def try_swap(i):
        nonlocal zero_count
        L = len(word)
        if i >= L - 1:
            return False
            
        updates = []
        
        # Decrement old windows
        k_start = max(0, i - n + 1)
        k_end = min(i + 2, L - n + 1)
        for k in range(k_start, k_end):
            pid = window_ids[k]
            if pid >= 0:
                if counts[pid] == 1:
                    zero_count += 1
                counts[pid] -= 1
                updates.append((pid, -1))
        
        # Perform swap
        word[i], word[i+1] = word[i+1], word[i]
        
        # Increment new windows
        for k in range(k_start, k_end):
            p = tuple(word[k:k+n])
            pid = perm_to_id.get(p)
            if pid is not None:
                if counts[pid] == 0:
                    zero_count -= 1
                counts[pid] += 1
                updates.append((pid, 1))
                window_ids[k] = pid
        
        if zero_count == 0:
            # Valid swap. Now try to delete to see if it helps.
            saved_word = word[:]
            saved_zero = zero_count
            saved_window_ids = window_ids[:]
            
            # Try delete i
            if try_delete(i):
                return True
            # Try delete i+1 (index might have shifted if delete i succeeded, but we returned)
            # If delete i failed, word is unchanged (post-swap).
            # try_delete reverts itself if it fails.
            # So state is post-swap.
            if try_delete(i+1):
                return True
            
            # No deletion possible. Revert swap.
            word[:] = saved_word
            zero_count = saved_zero
            window_ids[:] = saved_window_ids
            # Revert counts? try_delete reverted its own changes.
            # The swap changes are still applied. We need to revert them.
            # The `updates` list contains the swap changes.
            for pid, delta in updates:
                counts[pid] -= delta
            # Recalculate zero_count
            zero_count = sum(1 for c in counts if c == 0)
            return False
        else:
            # Invalid swap. Revert.
            word[i], word[i+1] = word[i+1], word[i]
            for pid, delta in updates:
                counts[pid] -= delta
            zero_count = sum(1 for c in counts if c == 0)
            # Restore window_ids for affected positions
            for k in range(k_start, k_end):
                p = tuple(word[k:k+n])
                window_ids[k] = perm_to_id.get(p, -1)
            return False

    # Iterate swap passes
    improved = True
    while improved:
        improved = False
        i = 0
        while i < len(word) - 1:
            if time.time() > deadline:
                return "".join(map(str, word))
            if try_swap(i):
                improved = True
            else:
                i += 1

    return "".join(map(str, word))
