import random
import time
import sys
from itertools import permutations

# EVOLVE-BLOCK-START

def construct(n):
    """
    Construct a superpermutation of 1..n.
    
    Uses the standard recursive construction as a starting point,
    then attempts local search to reduce length.
    """
    if n <= 1:
        return "1"
    
    # Standard recursive construction
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
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # Local search to try to reduce length
    word = _local_search(word, n, time_budget=100)
    
    return "".join(map(str, word))

def _local_search(word, n, time_budget=100):
    """Try to shorten the word while maintaining superpermutation property."""
    start_time = time.time()
    word = list(word)
    
    def is_valid(w):
        if len(w) < n:
            return False
        seen = set()
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i+n])
            if len(set(window)) == n:
                seen.add(window)
        return len(seen) == n * (n ** 0)  # n! permutations... wait, need factorial
        # Actually: len(seen) == len(list(permutations(range(1, n+1))))
        # But that's expensive. Let's just check count.
    
    # Precompute number of permutations
    import math
    target = math.factorial(n)
    
    def count_perms(w):
        seen = set()
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i+n])
            if len(set(window)) == n:
                seen.add(window)
        return len(seen)
    
    current_count = count_perms(word)
    if current_count < target:
        return word  # Not valid, return as-is
    
    # Try to remove symbols at various positions
    # This is a simple 1-move local search
    for attempt in range(10000):
        if time.time() - start_time > time_budget:
            break
        
        # Try removing a random symbol
        if len(word) <= n:
            break
        pos = random.randint(1, len(word) - 2)  # Don't remove first or last
        new_word = word[:pos] + word[pos+1:]
        
        if count_perms(new_word) >= target:
            word = new_word
    
    return word

# EVOLVE-BLOCK-END
