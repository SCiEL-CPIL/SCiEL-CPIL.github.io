import math
import itertools
import time

def construct(n):
    # Generate standard superpermutation
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
            overlap = 0
            # Optimization: max possible overlap is len(block)-1 or len(new)-1
            # But we can just check from min(len(new), len(block)) down to 0
            max_check = min(len(new), len(block))
            for o in range(max_check, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # Pruning phase
    # Target length for n=7 is 5905. Standard is 5913.
    # We try to remove characters one by one.
    
    def is_valid(w):
        if len(w) < n:
            return False
        seen = set()
        for i in range(len(w) - n + 1):
            seen.add(tuple(w[i:i+n]))
        return len(seen) == n * (n-1) * (n-2) * (n-3) * (n-4) * (n-5) * (n-6) if n==7 else len(seen) == int(math.factorial(n))
        
    # Precompute factorial for n
    fact_n = 1
    for i in range(1, n+1):
        fact_n *= i
        
    def check_validity(w):
        seen = set()
        for i in range(len(w) - n + 1):
            seen.add(tuple(w[i:i+n]))
        return len(seen) == fact_n

    # Greedy removal
    # We use a deadline
    start_time = time.time()
    deadline = start_time + 100 # Leave 50s margin
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(word)):
            if time.time() > deadline:
                break
            # Try removing word[i]
            # Optimization: if word[i] is part of a unique permutation that only appears here, 
            # removing it will definitely break validity.
            # But checking uniqueness is as hard as validity.
            
            # Just try removal
            new_word = word[:i] + word[i+1:]
            if check_validity(new_word):
                word = new_word
                changed = True
                break # Restart scan from beginning
    
    return "".join(map(str, word))
