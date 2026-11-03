import time
from math import factorial

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer
    
    # 1. Generate Classic Superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract all (m-1)-permutations from current word
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            # Insert m into p
            # Classic construction: p + [m] + p
            # But we need to overlap.
            block = list(p) + [m] + list(p)
            # Find max overlap with current new
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new)-o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # 2. Greedy Deletion Local Search
    target_count = factorial(n)
    
    def is_valid(w):
        seen = set()
        # Optimization: if len(w) < n, return False
        if len(w) < n:
            return False
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i+n])
            # Check if permutation
            # Since w contains only 1..n, we can check distinctness
            if len(set(window)) == n:
                seen.add(window)
                if len(seen) == target_count:
                    return True
        return len(seen) == target_count

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try deleting each character
        # We iterate backwards or forwards?
        # If we delete, indices change.
        # So we must restart the loop or iterate carefully.
        # Restarting is safer.
        for i in range(len(word)):
            if time.time() > deadline:
                break
            # Try deleting word[i]
            candidate = word[:i] + word[i+1:]
            if is_valid(candidate):
                word = candidate
                improved = True
                break # Restart loop with new length
    
    return "".join(map(str, word))
