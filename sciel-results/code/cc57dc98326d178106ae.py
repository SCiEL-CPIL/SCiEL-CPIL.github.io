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

def _fast_delete_pass(word, n, time_limit, start_time):
    """
    Fast deletion pass using permutation frequency counting.
    A character at position i can be deleted if no valid permutation window
    that contains position i is the ONLY occurrence of that permutation.
    """
    word = list(word)
    total_perms = n * _factorial(n)
    
    # Build initial permutation frequency table
    window_perm = [-1] * (len(word) - n + 1)
    perm_count = [0] * total_perms
    
    for i in range(len(word) - n + 1):
        window_tuple = tuple(word[i:i + n])
        if len(set(window_tuple)) == n:
            idx = _PERM_INDEX.get(window_tuple, -1)
            if idx >= 0:
                window_perm[i] = idx
                perm_count[idx] += 1
    
    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        # Try positions in reverse order for diversity
        for i in range(len(word) - 1, -1, -1):
            if time.time() - start_time > time_limit:
                break
            # Find all valid permutation windows containing position i
            s_min = max(0, i - n + 1)
            s_max = min(i, len(word) - n)
            critical = False
            for s in range(s_min, s_max + 1):
                if window_perm[s] >= 0:
                    if perm_count[window_perm[s]] == 1:
                        critical = True
                        break
            if not critical:
                # Delete position i
                del word[i]
                
                # Rebuild window_perm and perm_count from scratch
                window_perm = [-1] * (len(word) - n + 1)
                perm_count = [0] * total_perms
                for s in range(len(word) - n + 1):
                    window_tuple = tuple(word[s:s + n])
                    if len(set(window_tuple)) == n:
                        idx = _PERM_INDEX.get(window_tuple, -1)
                        if idx >= 0:
                            window_perm[s] = idx
                            perm_count[idx] += 1
                
                improved = True
                break  # Restart scan from beginning after a removal
    
    return word

def _try_swap_pass(word, n, time_limit, start_time):
    """
    Try swapping adjacent characters to see if we can maintain validity
    while keeping the same length (this doesn't help reduce length, but
    might enable further deletions).
    """
    word = list(word)
    total_perms = n * _factorial(n)
    
    for _ in range(100):  # Limit number of swap attempts
        if time.time() - start_time > time_limit:
            break
        
        # Try a random adjacent swap
        if len(word) < 2:
            break
        i = (int(time.time() * 1000000) % (len(word) - 1))
        
        # Check if swap is valid
        word[i], word[i + 1] = word[i + 1], word[i]
        
        # Quick check: does the word still contain all permutations?
        seen = set()
        for s in range(len(word) - n + 1):
            window = tuple(word[s:s + n])
            if len(set(window)) == n:
                seen.add(window)
        
        if len(seen) == total_perms:
            # Keep the swap
            pass
        else:
            # Revert
            word[i], word[i + 1] = word[i + 1], word[i]
    
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
    
    # Try swap pass to potentially enable more deletions
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 10:
        word = _try_swap_pass(word, n, remaining_time, start_time)
    
    # Final validation
    if not _is_valid_superperm(word, n):
        # Fallback to classic
        word = _classic_superperm(n)
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
