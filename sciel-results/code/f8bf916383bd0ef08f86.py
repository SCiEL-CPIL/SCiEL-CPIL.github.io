"""Superpermutation construction with time-bounded improvement search."""
import math
import time
import random
from itertools import permutations

# EVOLVE-BLOCK-START

def construct(n):
    start = time.time()
    DEADLINE = 145.0  # 5s safety margin before 150s kill
    
    # Step 1: Build the classic recursive superpermutation
    def build_classic(n):
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
                overlap = next((o for o in range(min(len(new), len(block)), -1, -1)
                               if o == 0 or new[len(new) - o:] == block[:o]), 0)
                new += block[overlap:]
            word = new
        return word
    
    # Step 2: Efficient validity check using window count
    def is_valid(w, n):
        if len(w) < n:
            return False
        target_count = math.factorial(n)
        seen = set()
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i+n])
            # Check if it's a valid permutation (all distinct symbols from 1 to n)
            if len(set(window)) == n:
                seen.add(window)
                if len(seen) == target_count:
                    return True
        return len(seen) == target_count
    
    # Step 3: Greedy removal optimization
    def greedy_remove(word, n, deadline):
        """Try to remove symbols while maintaining validity."""
        current_word = list(word)
        current_len = len(current_word)
        
        # Pre-compute target count
        target_count = math.factorial(n)
        
        # Helper to quickly check validity
        def check_valid(w):
            if len(w) < n:
                return False
            seen = set()
            for i in range(len(w) - n + 1):
                window = tuple(w[i:i+n])
                if len(set(window)) == n:
                    seen.add(window)
                    if len(seen) == target_count:
                        return True
            return len(seen) == target_count
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try positions in random order to avoid bias
            indices = list(range(len(current_word)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline:
                    break
                
                # Try removing current_word[i]
                new_word = current_word[:i] + current_word[i+1:]
                
                if check_valid(new_word):
                    current_word = new_word
                    improved = True
                    # Update indices since length changed
                    indices = list(range(len(current_word)))
                    random.shuffle(indices)
        
        return current_word
    
    # Step 4: Main construction
    word = build_classic(n)
    
    # Verify initial word is valid
    if not is_valid(word, n):
        # Fallback for small n
        if n <= 4:
            target = set(permutations(range(1, n + 1)))
            result = list(range(1, n + 1))
            for p in permutations(range(1, n + 1)):
                found = False
                for i in range(len(result) - n + 1):
                    if tuple(result[i:i + n]) == p:
                        found = True
                        break
                if not found:
                    best_append = list(p)
                    for o in range(n - 1, -1, -1):
                        if len(result) >= o and result[len(result) - o:] == list(p[:o]):
                            best_append = list(p[o:])
                            break
                    result.extend(best_append)
            return "".join(map(str, result))
        return "".join(map(str, word))
    
    # Try to improve if we have time
    elapsed = time.time() - start
    if elapsed < DEADLINE - 2.0:
        improved_word = greedy_remove(word, n, DEADLINE - 2.0)
        if is_valid(improved_word, n):
            word = improved_word
    
    return "".join(map(str, word))

# EVOLVE-BLOCK-END
