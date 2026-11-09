import time
import sys
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    
    # Generate classic superpermutation
    def classic_superperm(n):
        if n == 1:
            return [1]
        prev = classic_superperm(n - 1)
        perms = []
        seen = set()
        for i in range(len(prev) - (n - 2)):
            window = tuple(prev[i:i + n - 1])
            if len(set(window)) == n - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [n] + list(p)
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        return new
    
    word = classic_superperm(n)
    
    # Precompute powers for fast window encoding
    # code = sum(word[i+j] * n^j) for j in 0..n-1
    # When sliding: new_code = (old_code - word[i] * n^(n-1)) * n + word[i+n]
    n_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        n_powers[i] = n_powers[i-1] * n
    n_pow_n_minus_1 = n_powers[n-1]
    
    def encode_window_fast(w):
        code = 0
        for x in w:
            code = code * n + (x - 1)
        return code
    
    # Precompute all valid permutation encodings
    all_perms = set()
    for p in permutations(range(1, n + 1)):
        code = 0
        for x in p:
            code = code * n + (x - 1)
        all_perms.add(code)
    
    def build_window_codes(word, n):
        L = len(word)
        num_windows = L - n + 1
        window_codes = [0] * num_windows
        if num_windows <= 0:
            return window_codes
        # First window
        code = 0
        for j in range(n):
            code = code * n + (word[j] - 1)
        window_codes[0] = code
        # Slide
        for i in range(1, num_windows):
            code = (code - (word[i-1] - 1) * n_pow_n_minus_1) * n + (word[i+n-1] - 1)
            window_codes[i] = code
        return window_codes
    
    # Build initial window codes
    window_codes = build_window_codes(word, n)
    
    # Count occurrences of each permutation
    counts = {}
    for code in window_codes:
        if code in all_perms:
            counts[code] = counts.get(code, 0) + 1
    
    # Multi-pass greedy pruning with direction support
    def try_prune(word, window_codes, counts, n, deadline, direction=1):
        L = len(word)
        removed_any = False
        positions = list(range(L))
        if direction == -1:
            positions.reverse()
        
        for i in positions:
            if time.time() >= deadline:
                break
            L = len(word)
            if L < n:
                break
            if i >= L:
                continue
            
            # Windows affected by removing word[i]: starts s where s <= i <= s+n-1
            start_min = max(0, i - n + 1)
            start_max = min(i, L - n)
            
            # Check if any affected window has count == 1
            safe = True
            for s in range(start_min, start_max + 1):
                code = window_codes[s]
                if code in all_perms and counts.get(code, 0) == 1:
                    safe = False
                    break
            
            if not safe:
                continue
            
            # Compute new window codes BEFORE removal using old word
            new_codes = {}
            for s in range(start_min, start_max + 1):
                if s < i:
                    w_new = word[s:i] + word[i+1:s+n+1]
                else:
                    w_new = word[i+1:i+n+1]
                code = encode_window_fast(w_new)
                new_codes[s] = code
            
            # Decrement old window counts
            for s in range(start_min, start_max + 1):
                code = window_codes[s]
                if code in all_perms:
                    counts[code] -= 1
                    if counts[code] <= 0:
                        del counts[code]
            
            # Increment new window counts
            for s, code in new_codes.items():
                if code in all_perms:
                    counts[code] = counts.get(code, 0) + 1
            
            # Perform removal
            word.pop(i)
            
            # Rebuild window_codes array efficiently
            new_len = len(word)
            new_wc = build_window_codes(word, n)
            
            window_codes = new_wc
            removed_any = True
        
        return word, window_codes, counts, removed_any
    
    # Multiple passes in both directions
    for _ in range(200):
        if time.time() >= deadline:
            break
        word, window_codes, counts, removed_any = try_prune(word, window_codes, counts, n, deadline, direction=1)
        if not removed_any:
            break
        if time.time() >= deadline:
            break
        word, window_codes, counts, removed_any = try_prune(word, window_codes, counts, n, deadline, direction=-1)
        if not removed_any:
            break
    
    # Second phase: try removals in order of "redundancy" (positions with windows having higher counts)
    # This can find removals that sequential passes miss
    L = len(word)
    if L >= n and time.time() < deadline:
        # Build a priority list: for each position, compute max count among affected windows
        # Higher count = more redundant = safer to try first
        pos_priority = []
        for i in range(L):
            start_min = max(0, i - n + 1)
            start_max = min(i, L - n)
            max_count = 0
            for s in range(start_min, start_max + 1):
                code = window_codes[s]
                if code in all_perms:
                    c = counts.get(code, 0)
                    if c > max_count:
                        max_count = c
            pos_priority.append((max_count, i))
        
        # Sort by priority (highest first)
        pos_priority.sort(key=lambda x: -x[0])
        
        # Try removals in priority order
        for max_count, i in pos_priority:
            if time.time() >= deadline:
                break
            L = len(word)
            if L < n:
                break
            if i >= L:
                continue
            if max_count == 0:
                continue
            
            start_min = max(0, i - n + 1)
            start_max = min(i, L - n)
            
            safe = True
            for s in range(start_min, start_max + 1):
                code = window_codes[s]
                if code in all_perms and counts.get(code, 0) == 1:
                    safe = False
                    break
            
            if not safe:
                continue
            
            new_codes = {}
            for s in range(start_min, start_max + 1):
                if s < i:
                    w_new = word[s:i] + word[i+1:s+n+1]
                else:
                    w_new = word[i+1:i+n+1]
                code = encode_window_fast(w_new)
                new_codes[s] = code
            
            for s in range(start_min, start_max + 1):
                code = window_codes[s]
                if code in all_perms:
                    counts[code] -= 1
                    if counts[code] <= 0:
                        del counts[code]
            
            for s, code in new_codes.items():
                if code in all_perms:
                    counts[code] = counts.get(code, 0) + 1
            
            word.pop(i)
            window_codes = build_window_codes(word, n)
        
        # Another round of standard passes after priority removal
        for _ in range(100):
            if time.time() >= deadline:
                break
            word, window_codes, counts, removed_any = try_prune(word, window_codes, counts, n, deadline, direction=1)
            if not removed_any:
                break
            if time.time() >= deadline:
                break
            word, window_codes, counts, removed_any = try_prune(word, window_codes, counts, n, deadline, direction=-1)
            if not removed_any:
                break
    
    return "".join(map(str, word))

if __name__ == "__main__":
    res = construct(7)
    print(len(res))
