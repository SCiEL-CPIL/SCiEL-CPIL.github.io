import time
import random
import math

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s limit
    
    # 1. Generate standard recursive superpermutation
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
        
    # Convert to list of ints for easier manipulation
    S = list(word)
    orig_len = len(S)
    
    # 2. Initialize counts
    def build_counts(S, n):
        counts = {}
        for i in range(len(S) - n + 1):
            p = tuple(S[i:i+n])
            counts[p] = counts.get(p, 0) + 1
        return counts
    
    counts = build_counts(S, n)
    total_perms = math.factorial(n)
    
    best_S = S[:]
    best_len = len(S)
    
    # 3. Local Search with multiple strategies
    def is_valid(S, n):
        """Check if S contains all permutations of 1..n"""
        seen = set()
        for i in range(len(S) - n + 1):
            seen.add(tuple(S[i:i+n]))
        return len(seen) == total_perms
    
    def try_remove_pass(S, counts, n, deadline, direction='right_to_left'):
        """Try to remove characters, return new S and counts if any removal was made"""
        removed = False
        if direction == 'right_to_left':
            indices = range(len(S) - 1, -1, -1)
        else:
            indices = range(len(S))
        
        for i in indices:
            if time.time() > deadline:
                break
            if i >= len(S):
                continue
            
            # Calculate affected old windows
            start_old = max(0, i - n + 1)
            end_old = min(i, len(S) - n)
            
            # Check if removal is valid
            valid = True
            old_perms_to_dec = []
            
            for j in range(start_old, end_old + 1):
                p = tuple(S[j:j+n])
                if counts.get(p, 0) == 1:
                    valid = False
                    break
                old_perms_to_dec.append(p)
            
            if not valid:
                continue
            
            # Perform the removal
            for p in old_perms_to_dec:
                counts[p] -= 1
                if counts[p] == 0:
                    del counts[p]
            
            # Add new straddling windows
            new_len = len(S) - 1
            start_new = max(0, i - n + 1)
            end_new = min(i - 1, new_len - n)
            
            for k in range(start_new, end_new + 1):
                window_chars = []
                for idx in range(k, k+n):
                    s_idx = idx if idx < i else idx + 1
                    window_chars.append(S[s_idx])
                p_new = tuple(window_chars)
                counts[p_new] = counts.get(p_new, 0) + 1
            
            S.pop(i)
            removed = True
            # After pop, the next index in our iteration is now at the same position
            # For right_to_left, we just continue (next i is i-1, which is correct)
            # For left_to_right, we need to re-examine the same index
            if direction == 'left_to_right':
                # The element that was at i+1 is now at i, so we should check i again
                # But our for loop will move to i+1, so we need to adjust
                # Actually, since we popped at i, the next element to check should be at i again
                # But for loop increments, so we'll skip it. Let's handle this differently.
                pass
        
        return S, counts, removed
    
    def try_remove_pass_left(S, counts, n, deadline):
        """Left-to-right removal pass"""
        removed = False
        i = 0
        while i < len(S):
            if time.time() > deadline:
                break
            
            # Calculate affected old windows
            start_old = max(0, i - n + 1)
            end_old = min(i, len(S) - n)
            
            # Check if removal is valid
            valid = True
            old_perms_to_dec = []
            
            for j in range(start_old, end_old + 1):
                p = tuple(S[j:j+n])
                if counts.get(p, 0) == 1:
                    valid = False
                    break
                old_perms_to_dec.append(p)
            
            if not valid:
                i += 1
                continue
            
            # Perform the removal
            for p in old_perms_to_dec:
                counts[p] -= 1
                if counts[p] == 0:
                    del counts[p]
            
            # Add new straddling windows
            new_len = len(S) - 1
            start_new = max(0, i - n + 1)
            end_new = min(i - 1, new_len - n)
            
            for k in range(start_new, end_new + 1):
                window_chars = []
                for idx in range(k, k+n):
                    s_idx = idx if idx < i else idx + 1
                    window_chars.append(S[s_idx])
                p_new = tuple(window_chars)
                counts[p_new] = counts.get(p_new, 0) + 1
            
            S.pop(i)
            removed = True
            # Don't increment i, since the next position is now at i
        
        return S, counts, removed
    
    # Multiple passes of removal, alternating directions
    max_passes = 100
    for pass_num in range(max_passes):
        if time.time() > deadline:
            break
        
        if pass_num % 2 == 0:
            S, counts, removed = try_remove_pass(S, counts, n, deadline, 'right_to_left')
        else:
            S, counts, removed = try_remove_pass_left(S, counts, n, deadline)
        
        if removed:
            if len(S) < best_len:
                best_S = S[:]
                best_len = len(S)
        else:
            # Stuck - try a targeted perturbation: swap adjacent elements
            if len(S) > n and random.random() < 0.5:
                i = random.randint(1, len(S) - 1)
                S[i-1], S[i] = S[i], S[i-1]
                counts = build_counts(S, n)
            elif len(S) > n:
                # Try a small block reversal
                i = random.randint(0, len(S) - 3)
                S[i:i+3] = S[i:i+3][::-1]
                counts = build_counts(S, n)
            else:
                break
    
    # Final validity check
    if not is_valid(best_S, n):
        # Fall back to the original construction if invalid
        return "".join(map(str, word))
    
    return "".join(map(str, best_S))
