import time
import random
import math

def _factorial(n):
    return math.factorial(n)

def _classic_superperm(n):
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
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def construct(n):
    if n <= 6:
        w = _classic_superperm(n)
        return "".join(map(str, w))
        
    if n > 7:
        w = _classic_superperm(n)
        return "".join(map(str, w))
        
    start_time = time.time()
    deadline = start_time + 140
    
    word = _classic_superperm(n)
    target_count = _factorial(n)
    
    # Precompute permutation to index mapping
    # Generate all permutations of 1..n
    from itertools import permutations
    perms = list(permutations(range(1, n+1)))
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Initialize counts
    counts = [0] * target_count
    active_count = 0
    
    # Initial windows
    for i in range(len(word) - n + 1):
        idx = perm_to_idx[tuple(word[i:i+n])]
        if counts[idx] == 0:
            active_count += 1
        counts[idx] += 1
        
    best_len = len(word)
    best_word = word[:]
    
    # Local Search
    # We will try to delete characters.
    # To escape local minima, we can also try swaps, but deletions are the primary goal.
    
    # We track the number of failed attempts to restart or change strategy.
    failed_attempts = 0
    last_improvement_time = start_time
    
    while time.time() < deadline:
        # Try a move
        # Random index
        i = random.randint(0, len(word) - 1)
        
        # Determine affected old windows
        # Start indices: max(0, i-n+1) to min(i, len(word)-n)
        start_old = max(0, i - n + 1)
        end_old = min(i, len(word) - n)
        
        old_indices = []
        old_counts_changes = [] # (idx, delta)
        
        for k in range(start_old, end_old + 1):
            idx = perm_to_idx[tuple(word[k:k+n])]
            old_indices.append(idx)
            old_counts_changes.append((idx, -1))
            
        # Tentative deletion
        # We need to know the new windows to update counts.
        # New windows start at: max(0, i-n+1) to min(i-1, len(word)-1-n)
        # Note: len(word) will be len(word)-1 after deletion.
        # So new end is min(i-1, len(word)-1-n)
        
        start_new = max(0, i - n + 1)
        end_new = min(i - 1, len(word) - 1 - n)
        
        new_indices = []
        new_counts_changes = []
        
        # We need to calculate the new window tuples.
        # This requires accessing word[k] to word[k+n-1].
        # Since we are deleting i, for k < i, the window is word[k..k+n-1].
        # For k >= i, the window is word[k..k+n-2] + word[k+n-1]? No.
        # The windows are formed from the list *after* deletion.
        
        # To avoid modifying the list yet, we can simulate the tuple.
        # Or, we can just apply the deletion, calculate new windows, check, and rollback.
        
        # Apply deletion
        removed_val = word[i]
        del word[i]
        
        # Update counts for new windows
        for k in range(start_new, end_new + 1):
            idx = perm_to_idx[tuple(word[k:k+n])]
            new_indices.append(idx)
            new_counts_changes.append((idx, 1))
            
        # Now check if active_count is still target_count
        # We need to apply the changes to counts and active_count.
        
        # First, remove old
        for idx, delta in old_counts_changes:
            counts[idx] += delta
            if counts[idx] == 0:
                active_count -= 1
                
        # Add new
        for idx, delta in new_counts_changes:
            counts[idx] += delta
            if counts[idx] == 1:
                active_count += 1
                
        if active_count == target_count:
            # Valid move
            best_len = len(word)
            best_word = word[:]
            last_improvement_time = time.time()
            failed_attempts = 0
        else:
            # Rollback
            # Remove new
            for idx, delta in new_counts_changes:
                counts[idx] -= delta
                if counts[idx] == 0:
                    active_count -= 1
                    
            # Restore old
            for idx, delta in old_counts_changes:
                counts[idx] -= delta # delta was -1, so -(-1)=+1
                if counts[idx] == 1:
                    active_count += 1
                    
            # Restore word
            word.insert(i, removed_val)
            
            failed_attempts += 1
            # If stuck for too long, we might want to try a swap or restart
            # But for now, just keep trying.
            
    return "".join(map(str, best_word))
