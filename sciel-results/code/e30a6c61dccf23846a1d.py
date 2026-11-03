"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import permutations
from math import factorial

def _classic_superperm(n):
    """Classic recursive construction of length 1! + 2! + ... + n!"""
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

def _perm_to_int(p, n):
    """Convert a permutation tuple to an integer using factorial number system."""
    result = 0
    for i, val in enumerate(p):
        result = result * (n - i) + (val - 1)
    return result

def _int_to_perm(x, n):
    """Convert an integer back to a permutation."""
    result = []
    remaining = list(range(1, n + 1))
    for i in range(n):
        fact = factorial(n - i - 1)
        idx = x // fact
        x = x % fact
        result.append(remaining[idx])
        remaining.pop(idx)
    return tuple(result)

def _build_perm_lookup(n):
    """Build a dictionary mapping permutation tuples to their integer indices."""
    lookup = {}
    for i, p in enumerate(permutations(range(1, n + 1))):
        lookup[p] = i
    return lookup

def _compute_covered(word, n, perm_lookup):
    """Compute the set of permutation indices covered by the word."""
    covered = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n:
            idx = perm_lookup.get(window)
            if idx is not None:
                covered.add(idx)
    return covered

def _try_remove_chars_fast(word, n, time_limit, perm_lookup):
    """Try to remove redundant characters using fast coverage checking."""
    start_time = time.time()
    word = list(word)
    total_perms = factorial(n)
    covered = _compute_covered(word, n, perm_lookup)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Try removing each character from the end first (less disruptive)
        for i in range(len(word) - 1, -1, -1):
            if (time.time() - start_time) >= time_limit:
                return word
            # Quick check: if removing this char would leave us with fewer than n chars, skip
            if len(word) - 1 < n:
                continue
            # Try removing character at position i
            test_word = word[:i] + word[i + 1:]
            new_covered = _compute_covered(test_word, n, perm_lookup)
            if len(new_covered) == len(covered):
                word = test_word
                covered = new_covered
                improved = True
                break  # Restart scan after a removal
    
    return word

def _try_swap_chars(word, n, time_limit, perm_lookup):
    """Try swapping adjacent characters to potentially enable further removals."""
    start_time = time.time()
    word = list(word)
    total_perms = factorial(n)
    covered = _compute_covered(word, n, perm_lookup)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(len(word) - 1):
            if (time.time() - start_time) >= time_limit:
                return word
            # Try swapping characters at positions i and i+1
            if word[i] != word[i + 1]:
                word[i], word[i + 1] = word[i + 1], word[i]
                new_covered = _compute_covered(word, n, perm_lookup)
                if len(new_covered) >= len(covered):
                    covered = new_covered
                    improved = True
                    break
                else:
                    # Swap back
                    word[i], word[i + 1] = word[i + 1], word[i]
    
    return word

def construct(n):
    start_time = time.time()
    time_limit = 130  # Leave 20s safety margin for 150s limit
    
    # Get classic superpermutation
    word = _classic_superperm(n)
    
    # For small n, the classic construction might already be optimal or close
    if n <= 4:
        return "".join(map(str, word))
    
    # Build permutation lookup for fast coverage checking
    perm_lookup = _build_perm_lookup(n)
    
    # Try to remove redundant characters
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 5:
        word = _try_remove_chars_fast(word, n, remaining_time * 0.7, perm_lookup)
    
    # Try swaps to enable further removals
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 5:
        word = _try_swap_chars(word, n, remaining_time * 0.5, perm_lookup)
    
    # Final pass: try removals again after swaps
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 3:
        word = _try_remove_chars_fast(word, n, remaining_time, perm_lookup)
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
