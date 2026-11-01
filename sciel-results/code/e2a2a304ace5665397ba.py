import time
import itertools
import sys
from collections import Counter

def _classic_superperm(n):
    """Classic recursive construction: length 1!+2!+...+n!."""
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
    return word

def _is_valid_superperm(word, n, all_perms):
    if len(word) < n:
        return False
    seen = set()
    for i in range(len(word) - n + 1):
        seen.add(tuple(word[i:i + n]))
    return all_perms.issubset(seen)

def _greedy_deletion(word, n, all_perms, deadline, start_time):
    """Greedy deletion of redundant symbols."""
    L = len(word)
    if L < n:
        return word
    
    # Count permutations
    perm_counts = Counter()
    for i in range(L - n + 1):
        perm_counts[tuple(word[i:i + n])] += 1
        
    # Find all initially deletable indices
    deletable = []
    for i in range(L):
        s_min = max(0, i - n + 1)
        s_max = min(i, L - n)
        safe = True
        for s in range(s_min, s_max + 1):
            w = tuple(word[s:s + n])
            if perm_counts[w] == 1:
                safe = False
                break
        if safe:
            deletable.append(i)
            
    if not deletable:
        return word
        
    # Try deleting all at once
    delete_set = set(deletable)
    new_word = [word[i] for i in range(L) if i not in delete_set]
    if _is_valid_superperm(new_word, n, all_perms):
        return new_word
        
    # If bulk fails, try one-by-one
    # We need to re-evaluate safety after each deletion
    # This is expensive if done naively, but we only have a few candidates
    current_word = word
    current_counts = perm_counts
    
    # Sort candidates by index to keep track of shifts? 
    # Or just re-scan for deletable indices each time?
    # Since we expect few deletions, re-scanning is fine.
    
    max_attempts = 20
    attempts = 0
    while attempts < max_attempts and time.time() < deadline:
        attempts += 1
        L = len(current_word)
        if L < n:
            break
            
        # Re-compute counts? 
        # If we only delete one, we can update counts locally.
        # But for simplicity and safety with small n, re-compute is fast enough for 1-2 deletions.
        # However, for n=7, L=5913, re-computing is O(40k) ops. 20 attempts = 800k ops. Fast.
        
        perm_counts = Counter()
        for i in range(L - n + 1):
            perm_counts[tuple(current_word[i:i + n])] += 1
            
        # Find first deletable
        found = -1
        for i in range(L):
            s_min = max(0, i - n + 1)
            s_max = min(i, L - n)
            safe = True
            for s in range(s_min, s_max + 1):
                w = tuple(current_word[s:s + n])
                if perm_counts[w] == 1:
                    safe = False
                    break
            if safe:
                found = i
                break
                
        if found == -1:
            break
            
        # Delete it
        current_word.pop(found)
        
    return current_word

def construct(n):
    if n == 1:
        return "1"
    if n <= 4:
        word = _classic_superperm(n)
        return "".join(map(str, word))

    start_time = time.time()
    deadline = start_time + 115

    all_perms = set(itertools.permutations(range(1, n + 1)))
    
    # Generate classic
    word = _classic_superperm(n)
    
    # Try greedy deletion
    polished = _greedy_deletion(word, n, all_perms, deadline, start_time)
    
    # Validate polished
    if _is_valid_superperm(polished, n, all_perms):
        return "".join(map(str, polished))
        
    return "".join(map(str, word))
