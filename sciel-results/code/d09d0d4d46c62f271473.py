import math
import time
import sys
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (150s limit)
    
    # Recursive construction
    word = [1]
    for m in range(2, n + 1):
        # Extract unique permutations of length m-1 from word
        # in order of appearance
        perms = []
        seen = set()
        # word length is L. windows of size m-1
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            # Check if it's a permutation (distinct elements)
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            # Block: p[0]...p[m-2], m, p[0]...p[m-2]
            block = list(p) + [m] + list(p)
            
            # Find max overlap with new_word
            max_overlap = 0
            limit = min(len(new_word), len(block))
            for o in range(limit, -1, -1):
                if o == 0:
                    break
                if new_word[-o:] == block[:o]:
                    max_overlap = o
                    break
            
            new_word.extend(block[max_overlap:])
        
        word = new_word
        
        # Time check
        if time.time() > deadline:
            break
    
    # Local search polish
    # Try to remove characters
    if time.time() < deadline - 10.0:
        # Precompute permutation IDs for fast lookup
        all_perms = list(permutations(range(1, n + 1)))
        perm_to_id = {p: i for i, p in enumerate(all_perms)}
        num_perms = len(all_perms)
        
        # Initialize counts
        counts = [0] * num_perms
        for i in range(len(word) - n + 1):
            p = tuple(word[i:i+n])
            if p in perm_to_id:
                counts[perm_to_id[p]] += 1
        
        # Check if valid initially
        if any(c == 0 for c in counts):
            # Should not happen for standard construction
            return "".join(map(str, word))
        
        current_word = word[:]
        
        # Fast deletion check using incremental counts
        def try_delete(idx):
            """
            Attempts to delete word[idx]. Returns True if successful.
            Modifies word and counts if successful.
            """
            L = len(word)
            if L == n:  # Cannot delete if only one window
                return False
            
            # Store old values to revert if fail
            old_counts = counts.copy()
            
            # 1. Decrement old windows affected by deletion
            # Affected windows in old word: start k in [max(0, idx-n+1), min(idx, L-n)]
            old_affected = []
            for k in range(max(0, idx - n + 1), min(idx + 1, L - n + 1)):
                p = tuple(word[k:k+n])
                pid = perm_to_id.get(p)
                if pid is not None:
                    counts[pid] -= 1
                    old_affected.append((k, pid))
            
            # 2. Simulate deletion to find new windows
            # New word is word[:idx] + word[idx+1:]
            # New windows affected are those starting at k in [max(0, idx-n+1), min(idx, L-n-1)]
            new_affected = []
            for k in range(max(0, idx - n + 1), min(idx, L - n)):
                # The window is new_word[k:k+n]
                # If k < idx:
                #   new_word[k:k+idx-k] = word[k:idx]
                #   new_word[k+idx-k:k+idx-k+n-(idx-k)] = word[idx+1:k+n+1]
                if k < idx:
                    w1 = word[k:idx]
                    w2 = word[idx+1:k+n+1]
                    p = tuple(w1 + w2)
                else:  # k >= idx
                    # new_word[k] = word[k+1]
                    # window is word[k+1 : k+1+n]
                    p = tuple(word[k+1:k+1+n])
                
                pid = perm_to_id.get(p)
                if pid is not None:
                    counts[pid] += 1
                    new_affected.append((k, pid))
            
            # Check if all counts are >= 1
            valid = True
            for pid in range(num_perms):
                if counts[pid] < 1:
                    valid = False
                    break
            
            if valid:
                # Commit deletion
                del word[idx]
                return True
            else:
                # Revert
                counts[:] = old_counts
                return False
        
        # Try deleting each character from left to right, multiple passes
        random.seed(42)
        current = word[:]
        
        # Pass 1: Left-to-right deletion
        for i in range(len(current)):
            if time.time() > deadline - 5.0:
                break
            if len(current) <= n:
                break
            idx = i
            if try_delete(idx):
                i -= 1  # Stay at same index since list shrank
        
        # Pass 2: Random deletions
        for _ in range(2000):
            if time.time() > deadline - 5.0:
                break
            if len(current) <= n:
                break
            idx = random.randint(0, len(current) - 1)
            if try_delete(idx):
                pass  # Success
        
        word = current

    return "".join(map(str, word))
