"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def _classic_superperm(n):
    """Classic recursive superpermutation construction."""
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
    return word

def _perm_to_int(p, n):
    """Convert a permutation tuple to a unique integer using factorial number system."""
    result = 0
    for i, v in enumerate(p):
        # v is 1-indexed, convert to 0-indexed
        result = result * (n - i) + (v - 1)
    return result

def _is_valid_superperm(word, n):
    """Check if word contains all n! permutations of 1..n as consecutive windows."""
    if len(word) < n:
        return False
    seen = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i+n])
        if len(set(window)) == n:
            seen.add(_perm_to_int(window, n))
    return len(seen) == 1
    # We need exactly n! permutations
    # Let me fix this
    return len(seen) == _factorial(n)

def _factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def _try_shorten(word, n, deadline):
    """Try to shorten the superpermutation by removing redundant symbols."""
    word = list(word)
    target_perms = _factorial(n)
    
    def compute_seen(w):
        seen = set()
        for j in range(len(w) - n + 1):
            window = tuple(w[j:j+n])
            if len(set(window)) == n:
                seen.add(_perm_to_int(window, n))
        return seen
    
    # Try removing symbols, multiple passes
    while time.time() < deadline - 5:
        removed = False
        # Try removing each position from right to left
        for i in range(len(word) - 1, -1, -1):
            if time.time() > deadline - 5:
                return word
            if len(word) <= n:
                break
            
            new_word = word[:i] + word[i+1:]
            if len(new_word) < n:
                continue
            seen = compute_seen(new_word)
            if len(seen) == target_perms:
                word = new_word
                removed = True
                break
        
        if not removed:
            break
    
    return word

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Generate classic superpermutation
    word = _classic_superperm(n)
    
    if n <= 3:
        return "".join(map(str, word))
    
    # Try to shorten via local search
    if time.time() < deadline - 10:
        word = _try_shorten(word, n, deadline)
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
