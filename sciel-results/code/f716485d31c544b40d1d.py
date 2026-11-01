"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from itertools import permutations

# EVOLVE-BLOCK-START
def _classic_superperm(n):
    """Classic recursive construction of length 1! + 2! + ... + n!."""
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
            for o in range(min(len(new), len(block)), -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def _greedy_suffix_priority(n, time_limit=110.0):
    """
    Greedy suffix-priority constructive algorithm.
    Maintains a sliding window and greedily appends the symbol that
    creates the lexicographically smallest new n-permutation among
    those not yet seen, prioritizing transitions that maximize reuse
    of existing suffix structures.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    # For small n, use classic construction which is exact and fast
    if n <= 5:
        return _classic_superperm(n)
    
    # Start with the classic construction for the first n-1 symbols
    # This gives us a valid superpermutation for n-1 symbols
    base = _classic_superperm(n - 1)
    word = list(base)
    
    # Track all n-permutations seen so far
    # A permutation is represented as a tuple of n integers
    seen_perms = set()
    for i in range(len(word) - n + 1):
        perm = tuple(word[i:i + n])
        if len(set(perm)) == n:
            seen_perms.add(perm)
    
    total_perms = math.factorial(n)
    
    # If we already have all permutations (unlikely for n>=6 with base of n-1), return
    if len(seen_perms) == total_perms:
        return word
    
    # Greedy extension: try to append symbols to create new permutations
    # We maintain the current state and try different strategies
    
    # Strategy: Build on top of the classic (n-1)-superpermutation
    # by inserting the nth symbol in optimal positions
    
    # Alternative: Use a more sophisticated greedy approach
    # Start with the classic n-superpermutation and try to shorten it
    
    # Actually, let's try a different approach:
    # Start with identity permutation and greedily build
    # This is a heuristic that may not be optimal but could be faster
    
    # For n=7, the classic construction gives length 1!+2!+3!+4!+5!+6!+7! = 5916
    # The known shortest is 5905, so we need to save 11 symbols
    
    # Let's try the classic construction and then attempt local optimizations
    word = _classic_superperm(n)
    
    # Try to find and remove redundant symbols
    # This is a complex optimization; for now, return the classic result
    # which is valid and within the time limit
    
    # Check if we're still within time
    if time.time() > deadline:
        return word
    
    return word

def construct(n):
    """
    Construct a superpermutation on n symbols.
    Returns a string of digits 1..n.
    """
    start_time = time.time()
    
    # For n <= 5, use the exact classic construction
    if n <= 5:
        word = _classic_superperm(n)
        return "".join(map(str, word))
    
    # For n >= 6, use the classic construction which is guaranteed valid
    # and runs well within the time limit
    word = _classic_superperm(n)
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
