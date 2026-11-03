# EVOLVE-BLOCK-START
import time
import sys
from itertools import permutations

# Precompute permutation index for n=7 (and smaller)
_PERM_INDEX = {}
_PERM_LIST = []
_PERM_INV = []

def _build_perm_tables(n):
    global _PERM_INDEX, _PERM_LIST, _PERM_INV
    _PERM_INDEX = {}
    _PERM_LIST = []
    _PERM_INV = []
    idx = 0
    for p in permutations(range(1, n + 1)):
        _PERM_INDEX[p] = idx
        _PERM_LIST.append(p)
        _PERM_INV.append(p)
        idx += 1

def _classic_superperm(n):
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
    return word

def _factorial(n):
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r

def _is_valid_superperm(word, n):
    if len(word) < n:
        return False
    total_perms = n * _factorial(n)
    seen = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n:
            seen.add(window)
    return len(seen) == total_perms

def _fast_delete_pass_optimized(word, n, time_limit, start_time):
    """
    Optimized deletion pass using incremental updates.
    A character at position i can be deleted if no valid permutation window
    that contains position i is the ONLY occurrence of that permutation.
    
    We maintain:
    - perm_count: list of counts for each permutation index
    - window_perm: list where window_perm[s] is the perm index for window starting at s, or -1 if invalid
    """
    word = list(word)
    L = len(word)
    total_perms = n * _factorial(n)
    
    if L < n:
        return word

    # Initialize window_perm and perm_count
    window_perm = [-1] * (L - n + 1)
    perm_count = [0] * total_perms
    
    for s in range(L - n + 1):
        window_tuple = tuple(word[s:s + n])
        if len(set(window_tuple)) == n:
            idx = _PERM_INDEX.get(window_tuple, -1)
            if idx >= 0:
                window_perm[s] = idx
                perm_count[idx] += 1

    # Helper to check if deletion at i is safe
    def can_delete(i):
        # Windows containing i are those starting at s such that s <= i <= s+n-1
        # => i-n+1 <= s <= i
        s_min = max(0, i - n + 1)
        s_max = min(i, L - n)
        
        for s in range(s_min, s_max + 1):
            if window_perm[s] >= 0:
                if perm_count[window_perm[s]] == 1:
                    return False
        return True

    # Helper to perform deletion and update state
    def do_delete(i):
        nonlocal L, window_perm, perm_count, word
        
        # 1. Identify affected windows in the OLD array
        # Windows starting at s in [i-n+1, i] are affected.
        # Some of these will disappear (if they contained the deleted char and are no longer valid or shifted)
        # Actually, it's easier to think about which windows in the NEW array correspond to which in the OLD.
        
        # Let's update counts by removing contributions from old windows that are "broken" or "shifted".
        # A window starting at s in the old array:
        # - If s < i: it shifts to s in the new array? No.
        #   Old window: word[s ... s+n-1]
        #   If i is in [s, s+n-1], the window content changes.
        
        # Let's just recalculate the counts for the affected range of windows.
        # The affected windows in the OLD array are those with start index s in [i-n+1, i].
        # Let's decrement their counts.
        s_min_old = max(0, i - n + 1)
        s_max_old = min(i, L - n)
        
        for s in range(s_min_old, s_max_old + 1):
            if window_perm[s] >= 0:
                perm_count[window_perm[s]] -= 1
                window_perm[s] = -1 # Mark as invalid for now
        
        # Delete the character
        del word[i]
        L -= 1
        
        # Now, we need to recalculate the windows in the NEW array that are affected.
        # The deletion at i affects windows in the NEW array starting at s_new in [i-n+1, i].
        # Note: indices >= i in the new array correspond to indices > i in the old array.
        # Indices < i in the new array correspond to indices < i in the old array.
        
        # The windows that need to be re-evaluated in the NEW array are those starting at s_new
        # such that the window [s_new, s_new+n-1] overlaps with the deletion point i.
        # In the new array, the deletion point is effectively at i (the char after i-1 is now at i).
        # The windows affected are those where s_new <= i <= s_new+n-1.
        # => i-n+1 <= s_new <= i.
        
        s_min_new = max(0, i - n + 1)
        s_max_new = min(i, L - n)
        
        for s in range(s_min_new, s_max_new + 1):
            window_tuple = tuple(word[s:s + n])
            if len(set(window_tuple)) == n:
                idx = _PERM_INDEX.get(window_tuple, -1)
                if idx >= 0:
                    window_perm[s] = idx
                    perm_count[idx] += 1
                else:
                    window_perm[s] = -1
            else:
                window_perm[s] = -1

    improved = True
    direction = 1 # 1 for forward, -1 for backward
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        
        # Determine range to scan
        if direction == 1:
            scan_range = range(L)
        else:
            scan_range = range(L - 1, -1, -1)
            
        for i in scan_range:
            if time.time() - start_time > time_limit:
                break
            if can_delete(i):
                do_delete(i)
                improved = True
                # After a deletion, the indices shift. 
                # If we are scanning forward, the next index to check is the same i (which now holds the next element).
                # If we are scanning backward, the next index to check is i-1? 
                # Actually, it's safer to restart the scan or just continue.
                # Continuing is faster. 
                # If forward: we deleted i. The element now at i was at i+1. We should check i again? 
                # Yes, because the new element at i might also be deletable.
                # So we don't increment i if we are in a for loop that auto-increments.
                # Python for loops don't work well with this. We use a while loop with manual index.
                pass # Placeholder, logic below
                
        # To handle the index shifting correctly, we replace the for loop with a while loop
        # Let's re-implement the inner loop properly.
        # The above for loop was illustrative. Let's do it right.
        
        # We will break out of the while loop if no improvement, or time out.
        # The inner loop needs to be a while loop to handle index adjustment.
        
        # Let's restart the inner logic with a while loop for index management
        # (This block is effectively the body of the 'while improved' loop, rewritten)
        
        # Wait, I already entered the 'for i in scan_range'. 
        # I should structure the code to use a while loop for the index 'i'.
        
        # Let's refactor the inner part of the 'while improved' block.
        
    # The previous implementation inside the for loop was flawed for index adjustment.
    # Let's rewrite the function body properly using a while loop for the index.
    
    return word

# Actually, let's just write the clean version of _fast_delete_pass_optimized directly.
# The draft above had a flaw in the loop control.

def _fast_delete_pass_optimized(word, n, time_limit, start_time):
    word = list(word)
    L = len(word)
    total_perms = n * _factorial(n)
    
    if L < n:
        return word

    window_perm = [-1] * (L - n + 1)
    perm_count = [0] * total_perms
    
    for s in range(L - n + 1):
        window_tuple = tuple(word[s:s + n])
        if len(set(window_tuple)) == n:
            idx = _PERM_INDEX.get(window_tuple, -1)
            if idx >= 0:
                window_perm[s] = idx
                perm_count[idx] += 1

    def can_delete(i):
        s_min = max(0, i - n + 1)
        s_max = min(i, L - n)
        for s in range(s_min, s_max + 1):
            if window_perm[s] >= 0:
                if perm_count[window_perm[s]] == 1:
                    return False
        return True

    def do_delete(i):
        nonlocal L, window_perm, perm_count, word
        
        # Decrement counts for old windows affected
        s_min_old = max(0, i - n + 1)
        s_max_old = min(i, L - n)
        
        for s in range(s_min_old, s_max_old + 1):
            if window_perm[s] >= 0:
                perm_count[window_perm[s]] -= 1
                window_perm[s] = -1
        
        del word[i]
        L -= 1
        
        # Recalculate new windows affected
        s_min_new = max(0, i - n + 1)
        s_max_new = min(i, L - n)
        
        for s in range(s_min_new, s_max_new + 1):
            window_tuple = tuple(word[s:s + n])
            if len(set(window_tuple)) == n:
                idx = _PERM_INDEX.get(window_tuple, -1)
                if idx >= 0:
                    window_perm[s] = idx
                    perm_count[idx] += 1
                else:
                    window_perm[s] = -1
            else:
                window_perm[s] = -1

    improved = True
    # Alternate directions to escape local minima
    direction = 1 
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        
        if direction == 1:
            i = 0
            while i < L:
                if time.time() - start_time > time_limit:
                    return word
                if can_delete(i):
                    do_delete(i)
                    improved = True
                    # Stay at the same index i to check the new element at this position
                else:
                    i += 1
        else:
            i = L - 1
            while i >= 0:
                if time.time() - start_time > time_limit:
                    return word
                if can_delete(i):
                    do_delete(i)
                    improved = True
                    # Stay at the same index i? 
                    # If we delete i, the element at i-1 shifts to i? No.
                    # If we delete i, the element at i+1 shifts to i.
                    # We are scanning backwards.
                    # If we delete i, the next candidate to check is i (which now holds the old i+1).
                    # So we stay at i.
                else:
                    i -= 1
        
        # Toggle direction
        direction *= -1

    return word

def construct(n):
    start_time = time.time()
    time_limit = 140  # Leave 10s safety margin for 150s limit
    
    # Build permutation tables
    _build_perm_tables(n)
    
    # Get classic superpermutation
    word = _classic_superperm(n)
    
    # Try to remove redundant characters using optimized fast deletion
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 5:
        word = _fast_delete_pass_optimized(word, n, remaining_time, start_time)
    
    # Final validation
    if not _is_valid_superperm(word, n):
        # Fallback to classic
        word = _classic_superperm(n)
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
