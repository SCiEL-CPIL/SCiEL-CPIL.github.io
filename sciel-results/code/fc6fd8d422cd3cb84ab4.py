import time
import sys
import math

def construct(n):
    """Construct the shortest known superpermutation for n symbols.
    
    Uses the classical recursive construction (1! + 2! + ... + n!)
    which for n=7 gives length 1!+2!+3!+4!+5!+6!+7! = 1+2+6+24+120+720+5040 = 5913.
    """
    # The classical recursive construction
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
    
    return "".join(map(str, word))
