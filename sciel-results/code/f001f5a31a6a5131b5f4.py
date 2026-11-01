import time
import random
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 145.0
    
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    # Build standard superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        word = new_word
    
    if n <= 4:
        return "".join(map(str, word))
    
    target_count = 1
    for i in range(2, n + 1):
        target_count *= i
    
    current_word = list(word)
    L = len(current_word)
    
    # Build window count map and position-to-windows mapping
    def build_maps(w, n):
        counts = defaultdict(int)
        pos_windows = defaultdict(list)  # pos -> list of (window_start, window_tuple)
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i+n])
            if len(set(window)) == n:
                counts[window] += 1
                for k in range(n):
                    pos_windows[i + k].append((i, window))
        return counts, pos_windows
    
    perm_count, pos_windows = build_maps(current_word, n)
    
    def try_remove(i):
        """Try to remove position i. Returns (success, affected_old, affected_new, removed_symbol, new_word)"""
        L_cur = len(current_word)
        if i < 0 or i >= L_cur:
            return False, [], [], None, None
        
        # Old windows containing position i
        old_start = max(0, i - n + 1)
        old_end = min(i, L_cur - n)
        
        affected_old = []
        for j in range(old_start, old_end + 1):
            window = tuple(current_word[j:j+n])
            if len(set(window)) == n:
                affected_old.append((j, window))
        
        # Check if any old window would be lost
        for j, window in affected_old:
            if perm_count[window] <= 1:
                # Check if this window reappears in new windows
                new_start = max(0, i - n + 1)
                new_end = min(i - 1, L_cur - n - 1)
                found = False
                for nj in range(new_start, new_end + 1):
                    new_window = tuple(current_word[nj:i] + current_word[i+1:nj+n+1])
                    if new_window == window:
                        found = True
                        break
                if not found:
                    return False, [], [], None, None
        
        # Compute new windows
        affected_new = []
        new_start = max(0, i - n + 1)
        new_end = min(i - 1, L_cur - n - 1)
        for j in range(new_start, new_end + 1):
            new_window = tuple(current_word[j:i] + current_word[i+1:j+n+1])
            if len(set(new_window)) == n:
                affected_new.append((j, new_window))
        
        removed_symbol = current_word[i]
        new_word = current_word[:i] + current_word[i+1:]
        return True, affected_old, affected_new, removed_symbol, new_word
    
    def apply_removal(i, affected_old, affected_new, new_word):
        nonlocal current_word, perm_count, pos_windows
        for j, window in affected_old:
            perm_count[window] -= 1
            if perm_count[window] == 0:
                del perm_count[window]
        for j, window in affected_new:
            perm_count[window] = perm_count.get(window, 0) + 1
        current_word = new_word
        # Rebuild pos_windows (expensive but necessary for correctness)
        # Only rebuild periodically or do incremental update
        pass
    
    def revert_removal(i, removed_symbol, affected_old, affected_new, old_word):
        nonlocal current_word, perm_count
        for j, window in affected_old:
            perm_count[window] = perm_count.get(window, 0) + 1
        for j, window in affected_new:
            perm_count[window] -= 1
            if perm_count[window] == 0:
                del perm_count[window]
        current_word = old_word
    
    # Main loop: repeated deletion passes with randomized order
    improved = True
    pass_count = 0
    
    while improved and time.time() < deadline:
        improved = False
        pass_count += 1
        
        L_cur = len(current_word)
        indices = list(range(L_cur))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline - 5:
                break
            if i >= len(current_word):
                continue
            
            success, affected_old, affected_new, removed_symbol, new_word = try_remove(i)
            if not success:
                continue
            
            old_word = current_word
            apply_removal(i, affected_old, affected_new, new_word)
            improved = True
    
    # Final verification
    if time.time() < deadline - 2:
        seen = set()
        for i in range(len(current_word) - n + 1):
            window = tuple(current_word[i:i+n])
            if len(set(window)) == n:
                seen.add(window)
        if len(seen) != target_count:
            current_word = list(word)
    
    return "".join(map(str, current_word))
