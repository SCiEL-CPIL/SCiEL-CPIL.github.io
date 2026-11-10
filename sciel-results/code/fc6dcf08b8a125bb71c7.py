import math
import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
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
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    # If n >= 5, try to shorten by 1
    if n >= 5:
        candidate = word[:-1]
        # Check validity of candidate
        # We need to check if all n! permutations are in candidate
        # n! is small for n <= 7 (5040)
        # Number of windows is len(candidate) - n + 1 ~ 5000
        # This is fast.
        
        if len(candidate) >= n:
            seen_perms = set()
            for i in range(len(candidate) - n + 1):
                w = tuple(candidate[i:i+n])
                if len(set(w)) == n:
                    seen_perms.add(w)
                    if len(seen_perms) == math.factorial(n):
                        break
            
            if len(seen_perms) == math.factorial(n):
                word = candidate

    return "".join(map(str, word))
