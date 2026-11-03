import math
import time
import sys
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We stop at 145s.
    deadline = start_time + 145.0
    
    # 1. Classic Recursive Construction
    # Length is 1! + 2! + ... + n!
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
        
        # Time check during construction (usually very fast, but safe)
        if time.time() > deadline:
            break
    
    # If n is small, classic is likely optimal or near-optimal and fast to verify
    if n <= 5:
        return "".join(map(str, word))
        
    # 2. Local Search Polish
    if time.time() > deadline - 5.0:
        return "".join(map(str, word))

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
    
    # Fast deletion check using incremental counts with delta-revert
    def try_delete(idx):
        """
        Attempts to delete current_word[idx]. 
        Returns True if successful.
        Modifies current_word and counts if successful.
        """
        L = len(current_word)
        if L == n:  # Cannot delete if only one window
            return False
        
        # List of (perm_id, delta) to revert if deletion fails
        changes = []
        
        # 1. Decrement old windows affected by deletion
        # Affected windows in old word: start k in [max(0, idx-n+1), min(idx, L-n)]
        for k in range(max(0, idx - n + 1), min(idx + 1, L - n + 1)):
            p = tuple(current_word[k:k+n])
            pid = perm_to_id.get(p)
            if pid is not None:
                counts[pid] -= 1
                changes.append((pid, -1))
                if counts[pid] < 0:
                    # This shouldn't happen if counts were correct, but safe check
                    # Actually, if it goes negative, it's definitely invalid
                    pass
        
        # 2. Simulate deletion to find new windows
        # New word is current_word[:idx] + current_word[idx+1:]
        # New windows affected are those starting at k in [max(0, idx-n+1), min(idx, L-n-1)]
        for k in range(max(0, idx - n + 1), min(idx, L - n)):
            # The window is new_word[k:k+n]
            # If k < idx:
            #   new_word[k:k+idx-k] = current_word[k:idx]
            #   new_word[k+idx-k:k+idx-k+n-(idx-k)] = current_word[idx+1:k+n+1]
            if k < idx:
                w1 = current_word[k:idx]
                w2 = current_word[idx+1:k+n+1]
                p = tuple(w1 + w2)
            else:  # k >= idx
                # new_word[k] = current_word[k+1]
                # window is current_word[k+1 : k+1+n]
                p = tuple(current_word[k+1:k+1+n])
                
            pid = perm_to_id.get(p)
            if pid is not None:
                counts[pid] += 1
                changes.append((pid, 1))
        
        # Check if all counts are >= 1
        valid = True
        # We only need to check the pids that were changed
        # However, a pid could have been 1, become 0 (invalid), then become 1 again?
        # If a pid is not in 'changes', its count is unchanged.
        # If its count was >= 1 before, it is still >= 1.
        # So we only need to check the pids in 'changes'.
        # But wait, if a pid appears in 'changes' multiple times, we need its final state.
        # It's safer to just check the final value of any pid that was touched.
        
        touched_pids = set()
        for pid, _ in changes:
            touched_pids.add(pid)
            
        for pid in touched_pids:
            if counts[pid] < 1:
                valid = False
                break
        
        if valid:
            # Commit deletion
            del current_word[idx]
            return True
        else:
            # Revert
            for pid, delta in changes:
                counts[pid] -= delta
            return False
        
    # Search Loop
    random.seed(42)
    
    # Pass 1: Left-to-right deletion
    # This is often the most effective for the start of the word
    i = 0
    while i < len(current_word):
        if time.time() > deadline - 5.0:
            break
        if len(current_word) <= n:
            break
        if try_delete(i):
            # Stay at same index since list shrank
            pass
        else:
            i += 1
            
    # Pass 2: Right-to-left deletion
    i = len(current_word) - 1
    while i >= 0:
        if time.time() > deadline - 5.0:
            break
        if len(current_word) <= n:
            break
        if try_delete(i):
            # Stay at same index (which is now i-1 effectively if we decrement)
            # But since list shrank, index i in new list is the element that was at i+1.
            # We want to check the new element at position i.
            pass
        else:
            i -= 1
            
    # Pass 3: Random deletions
    # Use remaining time
    while time.time() < deadline - 2.0:
        if len(current_word) <= n:
            break
        idx = random.randint(0, len(current_word) - 1)
        try_delete(idx)
        
    return "".join(map(str, current_word))
