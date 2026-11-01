import time
import random
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 110s hard limit for internal logic to allow for return overhead

    # 1. Generate the naive superpermutation using standard construction
    word = [1]
    for m in range(2, n + 1):
        # Get all permutations of 1..m-1 that appear as windows in current word
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
    total_perms = 1
    for i in range(2, n + 1):
        total_perms *= i
        
    # Helper to check validity - optimized with early exit
    def is_valid(w):
        if len(w) < n:
            return False
        seen = set()
        for i in range(len(w) - n + 1):
            seen.add(tuple(w[i:i+n]))
            if len(seen) == total_perms:
                return True
        return len(seen) == total_perms

    # Greedy deletion: try to delete each position, keep if still valid
    def greedy_delete(w, direction='left'):
        result = w[:]
        if direction == 'left':
            i = 0
            while i < len(result) and time.time() < deadline:
                if len(result) > n:
                    candidate = result[:i] + result[i+1:]
                    if is_valid(candidate):
                        result = candidate
                        # Don't increment, try same position again
                    else:
                        i += 1
                else:
                    break
        else:
            i = len(result) - 1
            while i >= 0 and time.time() < deadline:
                if len(result) > n:
                    candidate = result[:i] + result[i+1:]
                    if is_valid(candidate):
                        result = candidate
                        # Don't decrement, try same position again
                    else:
                        i -= 1
                else:
                    break
        return result

    # Apply multiple greedy passes in both directions
    for _ in range(3):
        if time.time() >= deadline:
            break
        best_word = greedy_delete(best_word, 'left')
        best_word = greedy_delete(best_word, 'right')
    
    # Local search phase for further improvements
    current_len = len(best_word)
    
    while time.time() < deadline:
        op = random.random()
        candidate = best_word[:]
        
        if op < 0.7:
            # Delete
            if len(candidate) <= n:
                continue
            idx = random.randint(0, len(candidate) - 1)
            del candidate[idx]
        elif op < 0.85:
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
