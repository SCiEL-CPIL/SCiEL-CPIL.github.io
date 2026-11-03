"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from itertools import permutations

# EVOLVE-BLOCK-START

def _classic_superperm(n):
    """Build the classic superpermutation of length sum(k! for k=1..n)."""
    word = [1]
    for m in range(2, n + 1):
        # Collect all (m-1)-permutations in order of first appearance
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap between end of new and start of block
            overlap = 0
            max_ov = min(len(new), len(block) - 1)
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def _is_valid_superperm(word, n):
    """Check if word contains all n! permutations of 1..n as consecutive windows."""
    expected = math.factorial(n)
    seen = set()
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i+n])
        if len(set(w)) == n:
            seen.add(w)
            if len(seen) == expected:
                return True
    return len(seen) == expected

def _remove_redundant(word, n, deadline):
    """
    Iteratively remove redundant characters from the superpermutation.
    A character at index i is redundant if removing it does not cause
    any permutation's count to drop to zero.
    """
    word = word[:]
    L = len(word)
    
    # Initialize counts
    counts = {}
    for i in range(L - n + 1):
        w = tuple(word[i:i+n])
        if len(set(w)) == n:
            counts[w] = counts.get(w, 0) + 1
            
    # Check if valid initially
    if len(counts) < math.factorial(n):
        return word # Should not happen if input is valid

    # Local search: try to remove characters
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Scan positions.
        for i in range(1, L - 1):
            if time.time() > deadline:
                return word
            
            # Determine which windows are affected by removing index i.
            # Windows j such that j <= i < j+n  =>  i-n+1 <= j <= i
            # j must be in [0, L-n]
            min_j = max(0, i - n + 1)
            max_j = min(i, L - n)
            
            removable = True
            affected_perms = []
            for j in range(min_j, max_j + 1):
                w = tuple(word[j:j+n])
                if len(set(w)) == n:
                    if counts.get(w, 0) <= 1:
                        removable = False
                        break
                    affected_perms.append(w)
            
            if removable:
                # Perform removal
                # Update counts for affected windows (they disappear)
                for w in affected_perms:
                    counts[w] -= 1
                    if counts[w] == 0:
                        del counts[w]
                
                # Remove from word
                word.pop(i)
                L -= 1
                improved = True
                # Restart scan
                break
        
    return word

def construct(n):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    # For n <= 4, classic construction is optimal
    if n <= 4:
        word = _classic_superperm(n)
        return "".join(map(str, word))
    
    # Build classic superpermutation as baseline
    best_word = _classic_superperm(n)
    best_len = len(best_word)
    
    # Verify baseline
    if not _is_valid_superperm(best_word, n):
        # Should never happen, but safety
        return "".join(map(str, best_word))
    
    # For n=5, optimal is 152 (classic gives 153)
    # For n=6, optimal is 872 (classic gives 873)
    # For n=7, optimal is 5905 (classic gives 5913)
    
    # Strategy: local search - try to remove redundant symbols
    
    if n >= 5:
        # Use the efficient incremental removal
        improved_word = _remove_redundant(best_word, n, deadline)
        
        if len(improved_word) < best_len:
            best_word = improved_word
            best_len = len(best_word)
            
        # Verify the final result
        if _is_valid_superperm(best_word, n):
            return "".join(map(str, best_word))
        else:
            # Fallback to original if somehow broken
            return "".join(map(str, _classic_superperm(n)))
    
    return "".join(map(str, best_word))

# EVOLVE-BLOCK-END
