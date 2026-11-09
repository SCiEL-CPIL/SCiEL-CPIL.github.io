"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict
from itertools import permutations

def construct(n):
    """
    Construct a superpermutation of length 1!+2!+...+n! using the standard
    recursive construction, then prune it to find a shorter valid superpermutation.
    """
    # 1. Generate classic construction
    word = [1]
    for m in range(2, n + 1):
        # Find the order in which (m-1)-permutations appear in the current word
        perms, seen = [], set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Build the new word
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap between end of new and start of block
            overlap = 0
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # 2. Prune the word
    if n > 4:
        start_time = time.time()
        deadline = start_time + 110 # Safety margin
        
        # Pre-compute all permutations
        all_perms_set = set(permutations(range(1, n+1)))
        target_length = 5905 if n == 7 else len(word)
        
        def get_counts(w):
            counts = defaultdict(int)
            for i in range(len(w) - n + 1):
                window = tuple(w[i:i+n])
                if len(set(window)) == n:
                    counts[window] += 1
            return counts

        counts = get_counts(word)
        
        improved = True
        while improved and (time.time() - start_time) < deadline and len(word) > target_length:
            improved = False
            # Iterate from end to start
            for i in range(len(word) - 1, -1, -1):
                if (time.time() - start_time) > deadline:
                    break
                if len(word) <= target_length:
                    break
                
                # Try removing word[i]
                new_word = word[:i] + word[i+1:]
                
                # Check validity
                new_counts = get_counts(new_word)
                
                # Check if all perms are present
                valid = True
                if len(new_counts) < len(all_perms_set):
                    valid = False
                else:
                    for p in all_perms_set:
                        if new_counts[p] == 0:
                            valid = False
                            break
                
                if valid:
                    word = new_word
                    counts = new_counts
                    improved = True
        
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
