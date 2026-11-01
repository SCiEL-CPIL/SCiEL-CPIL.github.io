import time
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    # The evaluator kills after 150s. We use a 140s deadline to be safe.
    deadline = start_time + 140.0

    # 1. Standard recursive construction
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
    
    current_word = word
    current_len = len(current_word)
    
    target_count = 1
    for i in range(2, n+1):
        target_count *= i
        
    # Pre-compute counts
    counts = {}
    for i in range(len(current_word) - n + 1):
        window = tuple(current_word[i:i+n])
        if len(set(window)) == n:
            counts[window] = counts.get(window, 0) + 1
            
    # Fast try_remove
    def try_remove(i):
        L = len(current_word)
        if i < 0 or i >= L:
            return False
            
        # Check windows containing i
        start_j = max(0, i - n + 1)
        end_j = min(i, L - n)
        
        # 1. Validate: No window containing i should be unique
        for j in range(start_j, end_j + 1):
            w = tuple(current_word[j:j+n])
            # In a valid superperm, all these are perms. 
            # But we should be safe.
            if len(set(w)) == n:
                if counts.get(w, 0) == 1:
                    return False
        
        # 2. Update counts
        # Decrement old windows
        for j in range(start_j, end_j + 1):
            w = tuple(current_word[j:j+n])
            if len(set(w)) == n:
                counts[w] -= 1
                if counts[w] == 0:
                    del counts[w]
                    
        # Perform removal
        current_word.pop(i)
        
        # Increment new windows
        # The new windows are formed by the shift.
        # The gap is now between i-1 and i (in new indexing).
        # Windows in new word that cover this junction:
        # Start j_new such that j_new <= i-1 and j_new+n-1 >= i
        # i.e., i-n+1 <= j_new <= i-1
        # Also j_new must be valid index in new word [0, L-1-n]
        new_L = L - 1
        new_start_j = max(0, i - n + 1)
        new_end_j = min(i - 1, new_L - n)
        
        for j in range(new_start_j, new_end_j + 1):
            w = tuple(current_word[j:j+n])
            if len(set(w)) == n:
                counts[w] = counts.get(w, 0) + 1
            else:
                # If a non-perm window is created, it doesn't affect the count of perms,
                # but it means the word is no longer a "clean" superperm?
                # The problem says "contains every one of the 5040 permutations... as seven consecutive symbols".
                # It does NOT require that *only* permutations appear as windows.
                # So creating non-perm windows is fine, as long as all 5040 perms are still present.
                pass
        
        return True

    improved = True
    while improved and time.time() < deadline:
        improved = False
        L = len(current_word)
        indices = list(range(L))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            
            if try_remove(i):
                improved = True
                break
                
    return "".join(map(str, current_word))
