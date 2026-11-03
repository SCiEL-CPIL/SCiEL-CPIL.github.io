import time
import random

def construct(n):
    if n == 1:
        return "1"
    
    # Generate standard superpermutation
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
            overlap = 0
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    if n <= 6:
        return "".join(map(str, word))
        
    # Local search for n >= 7
    start_time = time.time()
    max_time = 110.0 # Leave 10s buffer
    
    # Precompute initial permutation counts
    perm_counts = {}
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i+n])
        perm_counts[p] = perm_counts.get(p, 0) + 1
        
    best_word = word[:]
    best_len = len(word)
    
    # Helper to check and apply swap
    def try_swap(i):
        if i >= len(word) - 1:
            return False
        # Affected windows: start j in [i-n+1, i+1]
        min_j = max(0, i - n + 1)
        max_j = min(i + 1, len(word) - n)
        
        old_perms = []
        for j in range(min_j, max_j + 1):
            old_perms.append((j, tuple(word[j:j+n])))
            
        # Perform swap
        word[i], word[i+1] = word[i+1], word[i]
        
        # Check new windows
        for j in range(min_j, max_j + 1):
            new_p = tuple(word[j:j+n])
            old_p = old_perms[j - min_j][1]
            if old_p != new_p:
                # Decrement old
                perm_counts[old_p] -= 1
                if perm_counts[old_p] == 0:
                    # Revert
                    word[i], word[i+1] = word[i+1], word[i]
                    perm_counts[old_p] += 1
                    return False
                # Increment new
                perm_counts[new_p] = perm_counts.get(new_p, 0) + 1
                
        return True

    # Helper to check and apply deletion
    def try_delete(i):
        if i < 0 or i >= len(word):
            return False
        # Affected windows: start j in [i-n+1, i]
        min_j = max(0, i - n + 1)
        max_j = min(i, len(word) - n)
        
        old_perms = []
        for j in range(min_j, max_j + 1):
            old_perms.append((j, tuple(word[j:j+n])))
            
        # Check if any old perm has count 1
        for j, p in old_perms:
            if perm_counts[p] <= 1:
                return False
                
        # Perform deletion
        word.pop(i)
        
        # Update counts
        for j, p in old_perms:
            perm_counts[p] -= 1
            
        # New windows formed across the gap
        # Gap is between i-1 and i in new word (old i+1 is now at i)
        new_min_j = max(0, i - n + 1)
        new_max_j = min(i - 1, len(word) - n)
        
        for j in range(new_min_j, new_max_j + 1):
            new_p = tuple(word[j:j+n])
            perm_counts[new_p] = perm_counts.get(new_p, 0) + 1
            
        return True

    # Main loop
    random.seed(42)
    stuck_count = 0
    
    while time.time() - start_time < max_time:
        if len(word) == best_len:
            # Try to find a move
            move_made = False
            # Try deletions first (greedy scan)
            i = 0
            while i < len(word):
                if try_delete(i):
                    move_made = True
                    # Don't increment i, re-check i (now old i+1)
                    if len(word) < best_len:
                        best_word = word[:]
                        best_len = len(word)
                    stuck_count = 0
                else:
                    i += 1
            if move_made:
                continue
                
            # Try random swaps
            # Pick a random index
            i = random.randint(0, len(word) - 2)
            if try_swap(i):
                move_made = True
                stuck_count = 0
            else:
                stuck_count += 1
                # If stuck too many times, restart from best
                if stuck_count > 1000:
                    word = best_word[:]
                    # Recount perms
                    perm_counts.clear()
                    for i in range(len(word) - n + 1):
                        p = tuple(word[i:i+n])
                        perm_counts[p] = perm_counts.get(p, 0) + 1
                    stuck_count = 0
        else:
            # This shouldn't happen if we update best_len correctly
            best_word = word[:]
            best_len = len(word)
            
    return "".join(map(str, best_word))
