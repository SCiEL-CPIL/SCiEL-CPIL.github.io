"""Superpermutation construction with deadline guard."""
import time
import sys

def _classical_superperm(n):
    """Classic recursive construction of length 1! + 2! + ... + n!."""
    if n <= 1:
        return [1]
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

def _verify(word, n):
    """Check that every permutation of 1..n appears as n consecutive symbols."""
    if len(word) < n:
        return False
    seen = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i+n])
        if len(set(window)) == n:
            seen.add(window)
    return len(seen) == _factorial(n)

def _factorial(n):
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r

def construct(n):
    """Construct a superpermutation on n symbols. Returns string of digits."""
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    if n <= 1:
        return "1"
    
    # For small n, use the classical construction which is optimal
    if n <= 4:
        word = _classical_superperm(n)
        return "".join(map(str, word))
    
    # For n=5,6,7: use classical construction as baseline
    # The classical construction gives length sum(k! for k=1..n)
    # For n=7: 1+2+6+24+120+720+5040 = 5913
    # The best known is 5905 (Teodorescu 2026)
    
    # Strategy: Use classical construction, then try to find overlaps/shortenings
    # via a greedy local search on the sequence of permutations
    
    word = _classical_superperm(n)
    
    if time.time() > deadline - 5:
        return "".join(map(str, word))
    
    # Try to improve by reordering the permutation sequence
    # The classical construction processes permutations in a specific order.
    # We can try to find a better ordering that allows more overlap.
    
    # For n=7, this is very hard. Let's try a simple improvement:
    # Check if we can merge some adjacent blocks more tightly.
    
    # Actually, for n>=5, the classical construction is already near-optimal.
    # The difference between 5913 and 5905 is only 8 characters.
    # Finding those improvements requires sophisticated search.
    
    # Let's try a simple hill-climbing on the last few permutations' order
    # This is unlikely to help much for n=7 within the time budget,
    # but it's worth attempting.
    
    if n >= 5 and time.time() < deadline - 10:
        # Try to find a better ordering for the last level
        # This is a simplified approach - just try swapping adjacent permutations
        # in the last construction step
        
        best_word = word
        best_len = len(word)
        
        # Reconstruct the permutation list for the last step
        prev_word = _classical_superperm(n - 1)
        perms, seen = [], set()
        for i in range(len(prev_word) - n + 2):
            window = tuple(prev_word[i:i + n - 1])
            if len(set(window)) == n - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Try local swaps
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(perms) - 1):
                if time.time() > deadline - 5:
                    break
                # Swap perms[i] and perms[i+1]
                perms[i], perms[i+1] = perms[i+1], perms[i]
                
                new = []
                for p in perms:
                    block = list(p) + [n] + list(p)
                    overlap = 0
                    for o in range(min(len(new), len(block)), -1, -1):
                        if o == 0 or new[len(new) - o:] == block[:o]:
                            overlap = o
                            break
                    new += block[overlap:]
                
                if len(new) < best_len:
                    best_len = len(new)
                    best_word = new
                    improved = True
                else:
                    # Swap back
                    perms[i], perms[i+1] = perms[i+1], perms[i]
        
        if len(best_word) < len(word):
            word = best_word
    
    return "".join(map(str, word))
