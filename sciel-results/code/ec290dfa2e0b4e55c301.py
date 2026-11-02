import time
import sys
from collections import deque

def construct(n):
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    start_time = time.time()
    TIME_LIMIT = 115.0  # Leave 5s margin for 120s limit
    
    # Step 1: Generate the standard recursive superpermutation
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
            # Find max overlap
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # Step 2: Compress using greedy deletion
    # Precompute permutation IDs
    import itertools
    all_perms = list(itertools.permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    counts = [0] * num_perms
    unique_count = 0
    
    # Initialize counts
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i + n])
        if w in perm_to_id:
            pid = perm_to_id[w]
            if counts[pid] == 0:
                unique_count += 1
            counts[pid] += 1
    
    # Greedy Deletion
    # We try to delete characters one by one, restarting the scan after each deletion
    # To speed up, we can try multiple passes with different strategies
    
    L = len(word)
    
    # Multiple passes of greedy deletion
    for pass_num in range(20):
        if time.time() - start_time > TIME_LIMIT:
            break
        
        deleted_in_pass = False
        i = 0
        while i < len(word):
            if time.time() - start_time > TIME_LIMIT:
                break
                
            L = len(word)
            if L <= n:
                break
                
            # Identify windows covering position i
            start = max(0, i - n + 1)
            end = min(i, L - n)
            
            old_ids = []
            # Remove old windows
            for j in range(start, end + 1):
                w = tuple(word[j:j + n])
                if w in perm_to_id:
                    pid = perm_to_id[w]
                    old_ids.append(pid)
                    counts[pid] -= 1
                    if counts[pid] == 0:
                        unique_count -= 1
                        
            # Create new word
            new_word = word[:i] + word[i + 1:]
            new_L = L - 1
            
            # Add new windows
            new_ids = []
            for j in range(start, min(end, new_L - n) + 1):
                if j < 0:
                    continue
                w = tuple(new_word[j:j + n])
                if w in perm_to_id:
                    pid = perm_to_id[w]
                    new_ids.append(pid)
                    if counts[pid] == 0:
                        unique_count += 1
                    counts[pid] += 1
                        
            # Check validity
            if unique_count == num_perms:
                # Accept deletion
                word = new_word
                deleted_in_pass = True
                # Don't increment i, since indices shifted
                i = max(0, i - 1)
            else:
                # Revert
                for pid in new_ids:
                    counts[pid] -= 1
                    if counts[pid] == 0:
                        unique_count -= 1
                for pid in old_ids:
                    counts[pid] += 1
                    if counts[pid] == 1:
                        unique_count += 1
                i += 1
        
        if not deleted_in_pass:
            break
    
    return "".join(map(str, word))
