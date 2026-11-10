"""Improved superpermutation construction using better overlap strategies."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import defaultdict

def construct(n):
    deadline = time.time() + 115  # safety margin before 120s limit
    
    if n <= 1:
        return "1"
    
    # Use the improved recursive construction
    # The classic method gives sum_{k=1}^n k! which for n=7 is 5913
    # We can do better by using a smarter ordering of permutations
    
    word = [1]
    
    for m in range(2, n + 1):
        # Extract all (m-1)-permutations from current word in order of first appearance
        perms, seen = [], set()
        # Optimization: use a rolling hash or just tuple slicing
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
            # Optimization: Only check overlaps up to len(block)
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            
            new += block[overlap:]
        
        word = new
    
    # Post-processing: Trim the word if possible
    # Check if the last characters are redundant
    if n > 1 and len(word) > n:
        # Find the last occurrence of each permutation
        last_occ = {}
        for i in range(len(word) - n + 1):
            window = tuple(word[i:i + n])
            if len(set(window)) == n:
                last_occ[window] = i
        
        if last_occ:
            max_last = max(last_occ.values())
            # We can keep up to max_last + n
            if max_last + n < len(word):
                word = word[:max_last + n]
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
