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
    # Use factorial number system for faster indexing if needed, 
    # but for n=7, tuple dict is fast enough for 5040 entries.
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

def _fast_delete_pass(word, n, time_limit, start_time):
    """
    Fast deletion pass using global sweep.
    1. Count occurrences of each permutation in all valid windows.
    2. Mark positions that are part of any window with count == 1 as 'protected'.
    3. Delete all un-protected positions.
    4. Repeat until no more deletions or time limit.
    """
    word = list(word)
    total_perms = n * _factorial(n)
    
    while True:
        if time.time() - start_time > time_limit:
            break
        
        L = len(word)
        if L < n:
            break
            
        # Build window IDs and counts
        num_windows = L - n + 1
        window_id = [-1] * num_windows
        perm_count = [0] * total_perms
        
        for i in range(num_windows):
            window_tuple = tuple(word[i:i + n])
            # Check if valid permutation (all distinct)
            # For n=7, this is a quick check. 
            # We can use a bitmask or just set. 
            # Since n is small, set is fine.
            if len(set(window_tuple)) == n:
                idx = _PERM_INDEX.get(window_tuple, -1)
                if idx >= 0:
                    window_id[i] = idx
                    perm_count[idx] += 1
        
        # Determine protected positions
        is_protected = [False] * L
        
        # Mark positions in windows with count == 1
        for i in range(num_windows):
            if window_id[i] >= 0:
                if perm_count[window_id[i]] == 1:
                    # Mark all positions in this window
                    for j in range(i, i + n):
                        is_protected[j] = True
        
        # Check if any position is unprotected
        deletions = []
        for i in range(L):
            if not is_protected[i]:
                deletions.append(i)
                
        if not deletions:
            break
            
        # Perform deletions from right to left to maintain index validity
        for i in reversed(deletions):
            del word[i]
            
        # If no deletions were made (should not happen if deletions list was non-empty), break
        if len(deletions) == 0:
            break

    return word

def construct(n):
    start_time = time.time()
    time_limit = 140  # Leave 10s safety margin for 150s limit
    
    # Build permutation tables
    _build_perm_tables(n)
    
    # Get classic superpermutation
    word = _classic_superperm(n)
    
    # Try to remove redundant characters using fast deletion
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 5:
        word = _fast_delete_pass(word, n, remaining_time, start_time)
    
    # Final validation
    if not _is_valid_superperm(word, n):
        # Fallback to classic
        word = _classic_superperm(n)
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
