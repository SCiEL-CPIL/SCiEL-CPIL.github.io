import time
import itertools
import random
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    # 1. Generate classic superpermutation
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
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # 2. Prepare for Greedy Deletion
    target_perms = set(itertools.permutations(range(1, n + 1)))
    
    # Initialize counts of permutations in the current word
    perm_counts = {}
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i + n])
        if p in target_perms:
            perm_counts[p] = perm_counts.get(p, 0) + 1
    
    best_word = list(word)
    best_len = len(word)
    
    # Function to check if deleting index i is valid
    def try_delete(idx, word, perm_counts, n, target_perms):
        # Windows affected are those starting from max(0, idx - n + 1) to min(idx, len(word) - n)
        start_range = max(0, idx - n + 1)
        end_range = min(idx, len(word) - n)
        
        if start_range > end_range:
            return False  # No windows to check, but this shouldn't happen for valid idx
        
        # Check if any perm count would drop to 0
        affected = {}
        for i in range(start_range, end_range + 1):
            p = tuple(word[i:i + n])
            if p in target_perms:
                affected[p] = affected.get(p, 0) + 1
        
        for p, cnt in affected.items():
            if perm_counts.get(p, 0) <= cnt:
                return False
        
        # Apply deletion
        for p, cnt in affected.items():
            perm_counts[p] -= cnt
            if perm_counts[p] == 0:
                del perm_counts[p]
        
        word.pop(idx)
        return True
    
    # Forward greedy deletion pass
    def forward_pass(word, perm_counts, n, target_perms, deadline):
        i = 0
        progress = False
        while i < len(word) and time.time() < deadline:
            if try_delete(i, word, perm_counts, n, target_perms):
                progress = True
                # Don't increment i, check new element at position i
            else:
                i += 1
        return progress
    
    # Backward greedy deletion pass
    def backward_pass(word, perm_counts, n, target_perms, deadline):
        i = len(word) - 1
        progress = False
        while i >= 0 and time.time() < deadline:
            if try_delete(i, word, perm_counts, n, target_perms):
                progress = True
                # Don't decrement i, check new element at position i
            else:
                i -= 1
        return progress
    
    # Main optimization loop
    while time.time() < deadline - 2:
        # Try forward pass
        if forward_pass(word, perm_counts, n, target_perms, deadline - 2):
            if len(word) < best_len:
                best_word = list(word)
                best_len = len(word)
        
        # Try backward pass
        if backward_pass(word, perm_counts, n, target_perms, deadline - 2):
            if len(word) < best_len:
                best_word = list(word)
                best_len = len(word)
        
        # If no progress, try a random restart with slight perturbation
        # For now, just break if stuck
        # We could also try swapping adjacent elements if valid
    
    return "".join(map(str, best_word))
