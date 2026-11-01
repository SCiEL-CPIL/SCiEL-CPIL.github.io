import time
import random
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 110s hard limit for internal logic to allow for return overhead

    # 1. Generate the naive superpermutation
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
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    best_word = word[:]
    n_perms = int(n)
    total_perms = 1
    for i in range(2, n + 1):
        total_perms *= i
        
    # Helper to check validity
    def is_valid(w):
        if len(w) < n:
            return False
        seen = set()
        for i in range(len(w) - n + 1):
            seen.add(tuple(w[i:i+n]))
        return len(seen) == total_perms

    current_len = len(best_word)
    
    # Initial check
    if not is_valid(best_word):
        return "".join(map(str, best_word))

    # Loop
    while time.time() < deadline:
        op = random.random()
        candidate = best_word[:]
        
        if op < 0.5:
            # Delete
            if len(candidate) <= n:
                continue
            idx = random.randint(0, len(candidate) - 1)
            del candidate[idx]
        elif op < 0.75:
            # Swap adjacent
            if len(candidate) < 2:
                continue
            idx = random.randint(0, len(candidate) - 2)
            candidate[idx], candidate[idx+1] = candidate[idx+1], candidate[idx]
        else:
            # Move
            if len(candidate) < 2:
                continue
            idx = random.randint(0, len(candidate) - 1)
            new_idx = random.randint(0, len(candidate) - 1)
            if idx == new_idx:
                continue
            val = candidate.pop(idx)
            candidate.insert(new_idx, val)
            
        if is_valid(candidate):
            best_word = candidate
            current_len = len(best_word)
            
    return "".join(map(str, best_word))
