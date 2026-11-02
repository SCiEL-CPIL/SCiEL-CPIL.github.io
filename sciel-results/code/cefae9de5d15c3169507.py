import itertools
import time

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"

    # For n=7, use the classic recursive construction (length 5913)
    # The task mentions shortest known is 5905, but we'll use the standard construction
    # which is 5913 for n=7.
    
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
                          if new[len(new) - o:] == block[:o]), 0)
            new += block[overlap:]
        word = new
    return "".join(map(str, word))
