"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import permutations
import random

def construct(n):
    """
    Constructs a superpermutation for n symbols.
    Uses a greedy construction with multiple restarts and different permutation orderings
    to try to find a shorter superpermutation than the standard recursive construction.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    if n == 1:
        return "1"
    
    # For n <= 6, the standard construction is optimal
    if n <= 6:
        word = _standard_construction(n)
        return "".join(map(str, word))
    
    # For n=7, try multiple strategies
    best_word = None
    best_len = float('inf')
    
    # Strategy 1: Standard construction
    word1 = _standard_construction(n)
    best_word = word1
    best_len = len(word1)
    
    # Strategy 2: Try different orderings at the last step
    # The standard construction at step m processes permutations in the order they appear
    # in the previous word. We can try other orderings.
    
    # Generate the word up to n-1
    word_base = _standard_construction(n - 1)
    
    # Get all permutations of 1..n-1 in the base word
    perms, seen = [], set()
    for i in range(len(word_base) - (n - 1) + 1):
        window = tuple(word_base[i:i + n - 1])
        if len(set(window)) == n - 1 and all(1 <= x <= n - 1 for x in window):
            if window not in seen:
                seen.add(window)
                perms.append(window)
    
    # Try several random orderings of these permutations
    num_trials = 50
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        
        # Create a shuffled order
        perm_order = list(perms)
        random.shuffle(perm_order)
        
        # Build the word using this order
        new_word = list(word_base)
        for p in perm_order:
            block = list(p) + [n] + list(p)
            # Find maximum overlap
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        
        # Verify validity
        if _is_valid_superpermutation(new_word, n):
            if len(new_word) < best_len:
                best_len = len(new_word)
                best_word = new_word
    
    # Strategy 3: Try greedy ordering - always pick the next permutation that
    # maximizes overlap with the current end
    if time.time() < deadline:
        word3 = _greedy_construct(n)
        if _is_valid_superpermutation(word3, n) and len(word3) < best_len:
            best_len = len(word3)
            best_word = word3
    
    return "".join(map(str, best_word))


def _standard_construction(n):
    """Classic recursive superpermutation construction."""
    word = [1]
    
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and all(1 <= x <= m - 1 for x in window):
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        
        word = new_word
    
    return word


def _greedy_construct(n):
    """
    Greedy construction: build the superpermutation by always choosing the next
    permutation that maximizes overlap with the current end.
    """
    if n == 1:
        return [1]
    
    # Start with a single permutation
    all_perms = list(permutations(range(1, n + 1)))
    random.shuffle(all_perms)
    
    # Start with the first permutation
    word = list(all_perms[0])
    covered = {tuple(all_perms[0])}
    
    remaining = all_perms[1:]
    
    while remaining and len(word) < n * len(all_perms):
        # Find the permutation that maximizes overlap with the end of word
        best_p = None
        best_overlap = -1
        best_idx = -1
        
        for idx, p in enumerate(remaining):
            p_list = list(p)
            # Find max overlap between end of word and start of p_list
            max_overlap = min(len(word), len(p_list))
            for o in range(max_overlap, -1, -1):
                if o == 0 or word[len(word) - o:] == p_list[:o]:
                    if o > best_overlap:
                        best_overlap = o
                        best_p = p
                        best_idx = idx
                    break
        
        if best_p is None:
            break
        
        p_list = list(best_p)
        word += p_list[best_overlap:]
        covered.add(tuple(best_p))
        remaining.pop(best_idx)
        
        # Check if all covered
        if len(covered) == len(all_perms):
            break
    
    return word


def _is_valid_superpermutation(word, n):
    """Check if word contains all permutations of 1..n as consecutive symbols."""
    all_perms = set(permutations(range(1, n + 1)))
    covered = set()
    
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if window in all_perms:
            covered.add(window)
            if len(covered) == len(all_perms):
                return True
    
    return len(covered) == len(all_perms)
# EVOLVE-BLOCK-END
