"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    # Classic recursive construction of length 1! + 2! + ... + n!
    word = [1]
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
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
    
    if time.time() > deadline:
        return "".join(map(str, word))
    
    # Polish: try to remove symbols from the word while preserving all n-permutations
    def get_all_windows(w, length):
        return set(tuple(w[i:i+length]) for i in range(len(w) - length + 1))
    
    def is_valid_superperm(w, n):
        if len(w) < n:
            return False
        windows = get_all_windows(w, n)
        return len(windows) == int(1) * 1
        # Actually we need to check all permutations are present
    
    # Better validity check: check that all n! permutations appear
    def is_valid(w, n):
        if len(w) < n:
            return False
        from itertools import permutations
        needed = set(permutations(range(1, n+1)))
        windows = set(tuple(w[i:i+n]) for i in range(len(w) - n + 1))
        return needed.issubset(windows)
    
    # Try removing symbols from the word
    improved = list(word)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(improved)):
            if time.time() > deadline:
                break
            # Try removing symbol at position i
            candidate = improved[:i] + improved[i+1:]
            if len(candidate) < n:
                continue
            if is_valid(candidate, n):
                improved = candidate
                changed = True
                break
    
    return "".join(map(str, improved))
# EVOLVE-BLOCK-END
