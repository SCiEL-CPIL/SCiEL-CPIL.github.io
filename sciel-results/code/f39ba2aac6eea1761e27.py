import time
import itertools
from collections import defaultdict

def _classic_superperm(n):
    """Classic recursive superpermutation construction."""
    if n == 1:
        return [1]
    word = _classic_superperm(n - 1)
    m = n
    perms, seen = [], set()
    # Extract (n-1)-permutations in order of appearance
    for i in range(len(word) - m + 2):
        window = tuple(word[i:i + m - 1])
        if len(set(window)) == m - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
            
    new = []
    for p in perms:
        block = list(p) + [m] + list(p)
        # Find max overlap with end of new
        overlap = 0
        max_ov = min(len(new), len(block))
        for o in range(max_ov, 0, -1):
            if new[len(new) - o:] == block[:o]:
                overlap = o
                break
        new += block[overlap:]
    return new

def construct(n):
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "123412134132413241423142314321"
    if n <= 6:
        word = _classic_superperm(n)
        return "".join(map(str, word))
        
    if n == 7:
        # Generate classic superpermutation
        word = _classic_superperm(n)
        
        # Precompute all permutations
        all_perms = set(itertools.permutations(range(1, n + 1)))
        
        # Count occurrences of each permutation in the current word
        perm_counts = defaultdict(int)
        L = len(word)
        for i in range(L - n + 1):
            window = tuple(word[i:i + n])
            if window in all_perms:
                perm_counts[window] += 1
        
        # Local search: try to delete characters
        # We iterate multiple times until no improvement or time limit
        improved = True
        while improved and (time.time() - start_time) < deadline:
            improved = False
            L = len(word)
            # Iterate from end to start to avoid index shifting issues?
            # Or just iterate and shift indices.
            # Iterating from end is safer.
            for i in range(L - 1, -1, -1):
                if (time.time() - start_time) > deadline:
                    break
                
                # Try deleting word[i]
                # Affected windows: starting at i-n+1 to i
                # Window start indices: max(0, i-n+1) to min(i, L-n)
                start_idx = max(0, i - n + 1)
                end_idx = min(i, L - n)
                
                valid = True
                affected_perms = []
                
                for k in range(start_idx, end_idx + 1):
                    window = tuple(word[k:k + n])
                    if window in all_perms:
                        if perm_counts[window] == 1:
                            valid = False
                            break
                        affected_perms.append(window)
                        perm_counts[window] -= 1
                
                if valid:
                    # Delete word[i]
                    del word[i]
                    improved = True
                else:
                    # Restore counts
                    for p in affected_perms:
                        perm_counts[p] += 1
        
        return "".join(map(str, word))
        
    # For n > 7, just use classic
    word = _classic_superperm(n)
    return "".join(map(str, word))
