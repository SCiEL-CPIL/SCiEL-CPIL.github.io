import time
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # 1. Generate standard superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract unique (m-1)-permutations in order of appearance
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find max overlap
            overlap = 0
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # 2. Optimization via greedy deletion
    L = len(word)
    if L < n:
        return "".join(map(str, word))
    
    # Precompute base-n powers for fast window hashing
    powers = [1] * n
    for k in range(1, n):
        powers[k] = powers[k-1] * n
    top_power = powers[n-1]
    
    # Greedy deletion loop
    changed = True
    while changed and time.time() < deadline:
        changed = False
        L = len(word)
        if L <= n:
            break
        
        # Precompute all window hashes for current word using sliding window
        num_windows = L - n + 1
        window_hashes = [0] * num_windows
        if num_windows > 0:
            h = 0
            for k in range(n):
                h = h * n + (word[k] - 1)
            window_hashes[0] = h
            for i in range(1, num_windows):
                h = (h - (word[i-1] - 1) * top_power) * n + (word[i+n-1] - 1)
                window_hashes[i] = h
        
        # Build counts
        counts = {}
        for wh in window_hashes:
            counts[wh] = counts.get(wh, 0) + 1
        
        for i in range(L):
            if time.time() > deadline:
                break
            
            # Windows covering position i
            j_min = max(0, i - n + 1)
            j_max = min(L - n, i)
            
            # Old windows to remove
            old_hashes = [window_hashes[j] for j in range(j_min, j_max + 1)]
            
            # New bridging windows after deletion
            new_L = L - 1
            j_new_min = max(0, i - n + 1)
            j_new_max = min(new_L - n, i - 1)
            
            new_hashes = []
            for j_new in range(j_new_min, j_new_max + 1):
                h = 0
                for k in range(j_new, j_new + n):
                    if k < i:
                        h = h * n + (word[k] - 1)
                    else:
                        h = h * n + (word[k + 1] - 1)
                new_hashes.append(h)
            
            new_set = set(new_hashes)
            
            # Check if deletion is valid: no window should drop to count 0
            valid = True
            for wh in old_hashes:
                if counts[wh] == 1 and wh not in new_set:
                    valid = False
                    break
            
            if valid:
                # Perform deletion
                word.pop(i)
                changed = True
                break
    
    return "".join(map(str, word))
