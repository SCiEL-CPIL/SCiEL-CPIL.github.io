"""Shortest superpermutation on seven symbols."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import Counter

def construct(n):
    if n == 1:
        return "1"
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Classic recursive construction of superpermutation
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
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # For n <= 4, the classic construction is optimal, return immediately
    if n <= 4:
        return "".join(map(str, word))
    
    # For n >= 5, attempt greedy deletion to shorten the word
    # Precompute all permutations of 1..n
    all_perms = set(itertools.permutations(range(1, n + 1)))
    
    # Count how many times each permutation appears as a window
    perm_counts = Counter()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        perm_counts[window] += 1
    
    # Greedy deletion: try to remove each position
    # To be efficient, track which permutations would be affected
    changed = True
    while changed and (time.time() - start_time) < deadline - 5:
        changed = False
        for i in range(len(word)):
            if time.time() - start_time > deadline - 5:
                break
            
            # Determine which windows are affected by removing position i
            s_min = max(0, i - n + 1)
            s_max = min(i, len(word) - n)
            
            # Check if removing position i would uncover any permutation
            safe = True
            for s in range(s_min, s_max + 1):
                window = tuple(word[s:s + n])
                if window in all_perms and perm_counts[window] == 1:
                    safe = False
                    break
            
            if safe:
                # Remove position i
                word.pop(i)
                changed = True
                # Update perm_counts for affected windows
                if len(word) < 20000:
                    perm_counts = Counter()
                    for s in range(len(word) - n + 1):
                        window = tuple(word[s:s + n])
                        perm_counts[window] += 1
                break  # Restart scan as indices changed
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
