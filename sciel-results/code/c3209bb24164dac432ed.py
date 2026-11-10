"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def _is_valid_superperm(word, n):
    """Check if word contains all n! permutations of 1..n as consecutive substrings."""
    if len(word) < n:
        return False
    target = set(itertools.permutations(range(1, n+1)))
    found = set()
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i+n])
        if len(set(w)) == n:  # valid permutation (all distinct)
            found.add(w)
        if len(found) == len(target):
            return True
    return len(found) == len(target)

def _count_missing(word, n):
    """Return set of permutations not found in word."""
    target = set(itertools.permutations(range(1, n+1)))
    found = set()
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i+n])
        if len(set(w)) == n:
            found.add(w)
    return target - found

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    if n <= 1:
        return "1"
    if n <= 3:
        # Small cases: brute force or known answers
        if n == 2:
            return "121"
        if n == 3:
            return "123121321"
    
    # Use the classic recursive construction which is fast and valid
    # For n=7, this gives length 1!+2!+...+7! = 5916
    
    word = [1]
    
    for m in range(2, n + 1):
        # Extract all (m-1)-permutations from current word in order of first appearance
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Build new word by inserting m into each permutation
        # Use a strategy that maximizes overlap
        new = []
        
        for p in perms:
            # Create the block: p + [m] + p
            block = list(p) + [m] + list(p)
            
            # Find maximum overlap with current end of new
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            
            new += block[overlap:]
        
        word = new
    
    best_word = list(word)
    best_len = len(best_word)
    
    # Local search: try to shorten the word
    # Strategy: try deleting positions where the character is redundant
    # i.e., removing it still leaves all permutations present
    
    def try_delete(word, pos, n):
        """Try deleting position pos, return new word if valid."""
        if pos < 0 or pos >= len(word):
            return None
        new_word = word[:pos] + word[pos+1:]
        if len(new_word) < n:
            return None
        # Quick check: does the deletion break any permutation?
        # We need to verify all permutations are still present
        # For efficiency, only check windows near the deletion point
        # But for correctness, we need full check for small n
        if n <= 6:
            if _is_valid_superperm(new_word, n):
                return new_word
        else:
            # For n=7, do a fast check
            target = set(itertools.permutations(range(1, n+1)))
            found = set()
            for i in range(len(new_word) - n + 1):
                w = tuple(new_word[i:i+n])
                if len(set(w)) == n:
                    found.add(w)
            if len(found) == len(target):
                return new_word
        return None
    
    # Iterative deletion search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for pos in range(len(best_word)):
            if time.time() > deadline:
                break
            new_word = try_delete(best_word, pos, n)
            if new_word is not None:
                best_word = new_word
                best_len = len(best_word)
                improved = True
                break  # restart from beginning after a successful deletion
    
    return "".join(map(str, best_word))
# EVOLVE-BLOCK-END
