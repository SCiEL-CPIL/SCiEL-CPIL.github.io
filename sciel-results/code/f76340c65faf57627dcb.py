import time
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0

    # 1. Generate seed superpermutation
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
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new

    if n <= 4:
        return "".join(map(str, word))

    current_word = word[:]
    
    all_perms = set(permutations(range(1, n+1)))
    
    def is_valid(w):
        if len(w) < n:
            return False
        seen = set()
        for i in range(len(w) - n + 1):
            seen.add(tuple(w[i:i+n]))
            if len(seen) == len(all_perms):
                return True
        return False

    # Greedy removal pass
    i = 0
    while i < len(current_word):
        if time.time() > deadline:
            break
        
        # Try removing current_word[i]
        new_word = current_word[:i] + current_word[i+1:]
        if is_valid(new_word):
            current_word = new_word
            # Stay at the same index i
        else:
            i += 1

    # If we have time, try to improve further with swaps
    # This is optional and might not help much for length reduction
    # unless it enables further removals.
    
    return "".join(map(str, current_word))
