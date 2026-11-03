import time
from itertools import permutations
from collections import defaultdict

def construct(n):
    deadline = time.time() + 140  # 10s safety margin

    # Classic recursive construction of length 1! + 2! + ... + n!
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
            best_overlap = 0
            max_check = min(len(new), len(block))
            for o in range(max_check, 0, -1):
                if new[-o:] == block[:o]:
                    best_overlap = o
                    break
            new += block[best_overlap:]
        word = new

    if n <= 6:
        return "".join(map(str, word))

    # For n >= 7, apply local search to shorten the word
    all_perms = set(permutations(range(1, n + 1)))
    current = word[:]
    n_local = n
    L = len(current)
    
    # Build reference count map: window -> count of occurrences
    ref_count = defaultdict(int)
    for i in range(L - n_local + 1):
        ref_count[tuple(current[i:i + n_local])] += 1
    
    # Count how many required permutations are covered (ref_count >= 1)
    covered = sum(1 for p in all_perms if ref_count[p] > 0)
    total_perms = len(all_perms)
    
    # Verify classic construction is valid
    if covered < total_perms:
        return "".join(map(str, current))

    # Fast local search: try deleting each character using reference counting
    improved = True
    start_time = time.time()
    
    while improved and time.time() - start_time < 130:  # Safety margin
        improved = False
        L = len(current)
        num_windows = L - n_local + 1
        
        # Try deleting from left to right
        for i in range(L):
            if (i % 500 == 0) and time.time() - start_time > 130:
                break
            
            # Check if deleting position i would cause any permutation to become uncovered
            # Windows affected: those that include position i
            # Window starting at j covers positions j..j+n-1
            # Position i is in window j iff j <= i <= j+n-1, i.e., i-n+1 <= j <= i
            # Also j must be in [0, num_windows-1]
            
            j_start = max(0, i - n_local + 1)
            j_end = min(i, num_windows - 1)
            
            # Check if any affected window has ref_count == 1 and is a required permutation
            can_delete = True
            old_windows = []
            for j in range(j_start, j_end + 1):
                w = tuple(current[j:j + n_local])
                old_windows.append(w)
                # Optimization: Only check all_perms if count is 1
                if ref_count.get(w, 0) == 1:
                    if w in all_perms:
                        can_delete = False
                        break
            
            if can_delete:
                # Perform deletion
                current.pop(i)
                L -= 1
                num_windows -= 1
                improved = True
                
                # Update reference counts incrementally
                # 1. Decrement counts for old windows that included position i
                for w in old_windows:
                    ref_count[w] -= 1
                
                # 2. Increment counts for new windows in the affected region
                # New windows spanning the join point:
                # k in [max(0, i-n+1), min(i-1, L-1-n)]
                k_start = max(0, i - n_local + 1)
                k_end = min(i - 1, L - 1 - n_local)
                
                for k in range(k_start, k_end + 1):
                    new_w = tuple(current[k:k + n_local])
                    ref_count[new_w] += 1

            # If cannot delete, move on to next position

    return "".join(map(str, current))
