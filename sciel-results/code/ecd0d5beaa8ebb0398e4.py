import time
import random
import math

def construct(n):
    start_time = time.time()
    # 150s limit. Use 145s for construction + search to be safe.
    deadline = start_time + 145.0
    
    # 1. Standard recursive construction (Seed)
    # This gives a valid superpermutation of length sum_{k=1}^n k!
    # For n=7, this is 1+2+6+24+120+720+5040 = 5913.
    # The known optimum is 5905.
    
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
            # Find max overlap
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # For small n, standard is optimal
    if n < 5:
        return "".join(map(str, word))

    # 2. Optimization Phase
    # Goal: Remove redundant symbols to shorten the word while keeping all permutations.
    
    # Pre-compute mapping from permutation tuple to index (0 to n!-1)
    # This allows us to use a bitset for validity checking.
    # We need to map every permutation of 1..n to a unique integer.
    # We can use the factorial number system (Lehmer code).
    
    target_count = 1
    for i in range(2, n+1):
        target_count *= i
        
    # Pre-compute factorials
    factorials = [1] * (n + 1)
    for i in range(1, n + 1):
        factorials[i] = factorials[i-1] * i
        
    def perm_to_index(p):
        # p is a tuple of length n containing 1..n
        # Calculate Lehmer code
        index = 0
        used = [False] * (n + 1)
        for i in range(n):
            count = 0
            for j in range(1, n + 1):
                if not used[j] and j < p[i]:
                    count += 1
            index += count * factorials[n - 1 - i]
            used[p[i]] = True
        return index

    # We need to check if all 5040 permutations are present.
    # We can use a Python integer as a bitset.
    # 5040 bits is about 630 bytes, which is fine.
    full_mask = (1 << target_count) - 1

    def is_valid(w):
        if len(w) < n:
            return False
        mask = 0
        # Iterate through windows
        # Optimization: Early exit if we have all bits set?
        # Checking (mask == full_mask) every step is expensive.
        # But we can check periodically or at the end.
        # Actually, just building the mask is fast enough.
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i+n])
            # Check if window is a permutation of 1..n
            # Since w is constructed from permutations, most windows are valid.
            # But after deletions, some might not be.
            # We must verify it's a permutation.
            # A fast check: sum is n(n+1)/2 and no duplicates?
            # Or just check length of set.
            # Given the structure, let's assume if it looks like a permutation, it is.
            # But to be safe and fast:
            if len(set(window)) == n:
                idx = perm_to_index(window)
                mask |= (1 << idx)
                # Optional: early exit if mask is full
                if mask == full_mask:
                    return True
        return mask == full_mask

    current_word = word
    current_len = len(current_word)
    
    # Optimization Loop
    # Strategy:
    # 1. Try to remove symbols from random positions.
    # 2. If valid, keep the removal.
    # 3. Repeat until no more removals are possible or time runs out.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Iterate through indices in a randomized order to avoid local minima
        indices = list(range(len(current_word)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            
            # Try removing current_word[i]
            # Optimization: Skip if removing this symbol breaks a "critical" window?
            # Hard to know without checking.
            
            new_word = current_word[:i] + current_word[i+1:]
            
            if is_valid(new_word):
                current_word = new_word
                improved = True
                # Restart the scan with a new shuffled list to handle index shifts
                break
                
    return "".join(map(str, current_word))
