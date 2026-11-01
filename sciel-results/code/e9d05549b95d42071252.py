"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from itertools import permutations

# EVOLVE-BLOCK-START
def _classic_superperm(n):
    """Classic recursive construction of length 1! + 2! + ... + n!."""
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
            for o in range(min(len(new), len(block)), -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def _is_valid_superperm(word, n):
    """Check if word contains all n-permutations as consecutive symbols."""
    if len(word) < n:
        return False
    seen = set()
    for i in range(len(word) - n + 1):
        perm = tuple(word[i:i + n])
        if len(set(perm)) == n:
            seen.add(perm)
    return len(seen) == math.factorial(n)

def _greedy_suffix_priority(n, time_limit=110.0):
    """
    Start with classic construction, then greedily remove redundant symbols.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    if n <= 5:
        return _classic_superperm(n)
    
    word = list(_classic_superperm(n))
    total_perms = math.factorial(n)
    
    # Precompute initial seen set
    def get_seen(word):
        seen = set()
        for i in range(len(word) - n + 1):
            perm = tuple(word[i:i + n])
            if len(set(perm)) == n:
                seen.add(perm)
        return seen
    
    seen = get_seen(word)
    
    # Try to remove each symbol one at a time, from left to right
    # If removal keeps validity, keep it removed
    i = 0
    while i < len(word) - n + 1:
        if time.time() > deadline - 5.0:
            break
        # Try removing word[i]
        new_word = word[:i] + word[i+1:]
        if len(new_word) < n:
            i += 1
            continue
        new_seen = get_seen(new_word)
        if len(new_seen) == total_perms:
            # Removal is valid
            word = new_word
            seen = new_seen
            # Don't increment i, check same position again
        else:
            i += 1
    
    return word

def construct(n):
    """
    Construct a superpermutation on n symbols.
    Returns a string of digits 1..n.
    """
    start_time = time.time()
    
    # For n <= 5, use the exact classic construction
    if n <= 5:
        word = _classic_superperm(n)
        return "".join(map(str, word))
    
    # For n >= 6, use classic construction + greedy removal
    word = _greedy_suffix_priority(n, time_limit=110.0)
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
