import time
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Build initial superpermutation using classic construction
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
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # Local search: try to remove symbols
    word = compress_superperm(word, n, deadline)
    
    return "".join(map(str, word))

def compress_superperm(word, n, deadline):
    """Try to shorten the superpermutation by removing redundant symbols."""
    word = list(word)
    improved = True
    pass_count = 0
    
    while improved and time.time() < deadline:
        improved = False
        pass_count += 1
        
        # Try removing each symbol one at a time
        for i in range(len(word)):
            if time.time() > deadline:
                break
            
            # Skip if removing would make word too short
            if len(word) <= n * n:
                break
            
            # Try removing position i
            candidate = word[:i] + word[i+1:]
            
            # Quick check: does candidate still contain all permutations?
            if is_valid_superperm(candidate, n):
                word = candidate
                improved = True
                # After removal, restart scan from beginning
                break
        
        # If we made progress, also try adjacent swaps to find better arrangements
        if improved and time.time() < deadline:
            for i in range(len(word) - 1):
                if time.time() > deadline:
                    break
                # Swap adjacent symbols
                word[i], word[i+1] = word[i+1], word[i]
                if is_valid_superperm(word, n):
                    word[i], word[i+1] = word[i+1], word[i]  # Keep swap
                    improved = True
                else:
                    word[i], word[i+1] = word[i+1], word[i]  # Revert
                    break
    
    return word

def is_valid_superperm(word, n):
    """Check if word contains all n! permutations of 1..n as consecutive substrings."""
    if len(word) < n:
        return False
    
    target_count = 1
    for i in range(2, n + 1):
        target_count *= i
    
    seen = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n:  # Valid permutation (all distinct)
            if window not in seen:
                seen.add(window)
                if len(seen) == target_count:
                    return True
    
    return len(seen) == target_count
