import time
from itertools import permutations

# Deadline guard: The evaluator kills at 150s. We aim to finish in < 10s.
DEADLINE = time.time() + 140.0

def _build_recursive(n):
    """Classic recursive superpermutation of length 1!+2!+...+n!."""
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
            # Find maximum overlap
            max_overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    max_overlap = o
                    break
            new += block[max_overlap:]
        word = new
    return word

def _compute_hash_counts(word, n):
    """
    Compute a dictionary mapping rolling hash of each n-window to its count.
    Uses base = n+1 to avoid collisions for n<=7 (max hash < (n+1)^n which fits in int).
    """
    if len(word) < n:
        return {}
    base = n + 1
    powb = base ** (n - 1)
    counts = {}
    
    # Initial hash
    h = 0
    for x in word[:n]:
        h = h * base + x
    counts[h] = counts.get(h, 0) + 1
    
    # Roll
    for i in range(n, len(word)):
        h = (h - word[i - n] * powb) * base + word[i]
        counts[h] = counts.get(h, 0) + 1
        
    return counts

def construct(n):
    if n <= 1:
        return "1"
    
    start_time = time.time()
    word = _build_recursive(n)
    
    # For small n, the recursive construction is optimal.
    if n <= 5:
        return "".join(map(str, word))
    
    base = n + 1
    powb = base ** (n - 1)
    
    # Helper to get hash of a specific slice quickly (for verification or local checks)
    def get_hash(slice_list):
        h = 0
        for x in slice_list:
            h = h * base + x
        return h

    # Main optimization loop
    # We repeatedly try to find a deletable character.
    # Since N is small (~6000), recomputing counts after each deletion is cheap.
    
    while time.time() < DEADLINE:
        counts = _compute_hash_counts(word, n)
        found_deletion = False
        
        # Scan left to right for a deletable index
        for i in range(len(word)):
            if time.time() > DEADLINE:
                break
            
            # Identify windows containing index i
            # Window j contains i if j <= i <= j + n - 1
            # => i - n + 1 <= j <= i
            # j must be valid window start: 0 <= j <= len(word) - n
            start_j = max(0, i - n + 1)
            end_j = min(i, len(word) - n)
            
            can_remove = True
            for j in range(start_j, end_j + 1):
                # Compute hash of word[j:j+n]
                # We can do this directly from the list
                h = 0
                for k in range(j, j + n):
                    h = h * base + word[k]
                
                if counts.get(h, 0) == 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Perform deletion
                word.pop(i)
                found_deletion = True
                break # Restart scan with new word
        
        if not found_deletion:
            break
            
    return "".join(map(str, word))
