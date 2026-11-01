import time
import itertools
from math import factorial

# EVOLVE-BLOCK-START

def _standard_superperm(n):
    """Classic recursive construction of length 1! + 2! + ... + n!."""
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
                if o == 0:
                    overlap = 0
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word


def _validate(word, n):
    """Check if word contains all n! permutations of 1..n as n-consecutive symbols."""
    if n <= 1:
        return len(word) == n
    if len(word) < n:
        return False
    required = set(itertools.permutations(range(1, n + 1)))
    found = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n:
            found.add(window)
    return found == required


def _optimize_superperm(word, n, time_limit=100):
    """
    Local search to shorten the superpermutation.
    Uses bytearray for faster slicing and incremental count updates.
    """
    if n <= 1:
        return word
    
    # Convert to bytearray for faster operations
    # Symbols are 1..n, so they fit in a byte
    word_bytes = bytearray(word)
    L = len(word_bytes)
    
    # Map permutations to indices
    all_perms = list(itertools.permutations(range(1, n + 1)))
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    # Initialize counts
    counts = [0] * num_perms
    
    # Helper to get index of a window from bytearray
    def get_idx(w, start):
        window = tuple(w[start:start + n])
        # Check uniqueness
        # For n<=7, set check is fast
        if len(set(window)) == n:
            return perm_to_idx.get(window, -1)
        return -1

    # Initial counts
    for i in range(len(word_bytes) - n + 1):
        idx = get_idx(word_bytes, i)
        if idx != -1:
            counts[idx] += 1
    
    start_time = time.time()
    improved = True
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        L = len(word_bytes)
        
        # Try to remove each symbol
        for i in range(L):
            if (time.time() - start_time) > time_limit:
                return list(word_bytes)
            
            # Identify windows to remove: those touching i
            # j such that j <= i <= j+n-1  =>  i-n+1 <= j <= i
            start_remove = max(0, i - n + 1)
            end_remove = min(i, L - n)
            
            # Check if removal is valid
            # 1. Identify new windows created by the gap
            # New windows start at i-k in the NEW word.
            # In the OLD word, they are formed by:
            # left = word_bytes[i-k : i]
            # right = word_bytes[i+1 : i+1 + (n-k)]
            
            new_indices = set()
            for k in range(1, n):
                # Left part length k, Right part length n-k
                left = word_bytes[i-k:i]
                right = word_bytes[i+1:i+1+n-k]
                if len(left) + len(right) != n:
                    continue
                new_win = tuple(left + right)
                if len(set(new_win)) == n:
                    idx = perm_to_idx.get(new_win, -1)
                    if idx != -1:
                        new_indices.add(idx)
            
            # 2. Check if any removed window is unique and not covered by new windows
            valid = True
            for j in range(start_remove, end_remove + 1):
                idx = get_idx(word_bytes, j)
                if idx != -1:
                    if counts[idx] == 1 and idx not in new_indices:
                        valid = False
                        break
            
            if valid:
                # Perform removal
                # Update counts for removed windows
                for j in range(start_remove, end_remove + 1):
                    idx = get_idx(word_bytes, j)
                    if idx != -1:
                        counts[idx] -= 1
                
                # Update counts for new windows
                for idx in new_indices:
                    counts[idx] += 1
                
                # Remove symbol i
                word_bytes.pop(i)
                improved = True
                break # Restart loop with new length
    
    return list(word_bytes)


def construct(n):
    if n <= 1:
        return "".join(str(i) for i in range(1, n + 1))
    
    start_time = time.time()
    deadline = 115  # Safety margin before 120s limit
    
    # For small n, use the exact optimal construction
    if n <= 4:
        return "".join(map(str, _standard_superperm(n)))
    
    # For n=5,6,7: use standard + local search
    base_word = _standard_superperm(n)
    
    # Try to improve with local search
    remaining_time = deadline - (time.time() - start_time)
    if remaining_time < 1:
        return "".join(map(str, base_word))
        
    improved_word = _optimize_superperm(base_word, n, time_limit=remaining_time)
    
    # Validate
    if _validate(improved_word, n):
        return "".join(map(str, improved_word))
    
    return "".join(map(str, base_word))

# EVOLVE-BLOCK-END
