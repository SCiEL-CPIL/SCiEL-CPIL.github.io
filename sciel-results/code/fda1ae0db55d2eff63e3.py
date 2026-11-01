import time
import random
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    # 1. Generate initial superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract unique permutations of length m-1 from current word
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            # Construct block: p + [m] + p
            block = list(p) + [m] + list(p)
            # Find max overlap with current new
            overlap = 0
            # Max possible overlap is len(block) - 1 or len(new)
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0:
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    if n <= 4:
        return "".join(map(str, word))
    
    # 2. Local Search Deletion with improved efficiency
    current = list(word)
    L = len(current)
    
    # Precompute window hashes using a rolling hash approach
    # For n=7, we can use a simple tuple-based approach but optimize the checks
    
    perm_count = defaultdict(int)
    for i in range(L - n + 1):
        p = tuple(current[i:i+n])
        perm_count[p] += 1
    
    target_perms = n
    for i in range(1, n):
        target_perms *= i
        
    if len(perm_count) < target_perms:
        return "".join(map(str, current))
    
    # Deletion loop with bidirectional scan and randomization
    while time.time() < deadline:
        # Alternate between forward and backward scans to escape local minima
        scan_direction = random.choice([1, -1])
        indices = list(range(L))
        if scan_direction == -1:
            indices.reverse()
        random.shuffle(indices)
        
        deleted_any = False
        
        for i in indices:
            if time.time() > deadline:
                break
            
            # Quick check: if position is at boundary, skip
            if i < 0 or i >= L:
                continue
            
            # Check if i can be deleted
            start_w = max(0, i - n + 1)
            end_w = min(i, L - n)
            
            # Collect old windows that will be destroyed
            old_windows = []
            for w in range(start_w, end_w + 1):
                p = tuple(current[w:w+n])
                old_windows.append(p)
            
            # Compute new windows that will be created
            new_windows = []
            for j in range(max(0, i - n + 1), i):
                if j + n > L - 1:
                    continue
                # Window starting at j after deletion
                # It spans from j to i-1 (before deleted) and i to i+n-2 (after deleted)
                prefix = current[j:i]
                suffix = current[i+1:i+1+(n-(i-j))]
                if len(suffix) == n - (i - j):
                    p = tuple(prefix + suffix)
                    new_windows.append(p)
            
            new_set = set(new_windows)
            
            # Check if any old window with count 1 is not preserved
            can_delete = True
            for p in old_windows:
                if perm_count[p] == 1 and p not in new_set:
                    can_delete = False
                    break
            
            if can_delete:
                # Perform deletion and update counts
                for p in old_windows:
                    perm_count[p] -= 1
                    if perm_count[p] == 0:
                        del perm_count[p]
                        
                for p in new_windows:
                    perm_count[p] += 1
                    
                current = current[:i] + current[i+1:]
                L -= 1
                deleted_any = True
                # Break to restart scan with updated indices
                break
                
        if not deleted_any:
            break  # No more deletions possible
    
    return "".join(map(str, current))
