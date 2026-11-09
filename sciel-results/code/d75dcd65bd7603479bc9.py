import time
import sys
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin before 150s kill
    
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
    
    # Encode each window as an integer for fast hashing
    def encode_window(w):
        code = 0
        for x in w:
            code = code * n + (x - 1)
        return code
    
    # Precompute all valid permutation encodings
    all_perms = set()
    for p in permutations(range(1, n + 1)):
        all_perms.add(encode_window(p))
    
    # Build initial window codes
    L = len(word)
    window_codes = [0] * (L - n + 1)
    for i in range(L - n + 1):
        window_codes[i] = encode_window(word[i:i+n])
    
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
                code = encode_window(w_new)
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
            
            # Rebuild window_codes array
            new_len = len(word)
            new_wc = [0] * (new_len - n + 1)
            
            for s in range(new_len - n + 1):
                if s in new_codes:
                    new_wc[s] = new_codes[s]
                elif s < start_min:
                    new_wc[s] = window_codes[s]
                else:
                    old_s = s + 1
                    if old_s < len(window_codes):
                        new_wc[s] = window_codes[old_s]
                    else:
                        w = word[s:s+n]
                        new_wc[s] = encode_window(w)
            
            window_codes = new_wc
            removed_any = True
        
        return word, window_codes, counts, removed_any
    
    # Multiple passes in both directions
    for _ in range(500):
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
    
    # Final validation: ensure all permutations are present
    L = len(word)
    seen_perms = set()
    for i in range(L - n + 1):
        code = encode_window(word[i:i+n])
        if code in all_perms:
            seen_perms.add(code)
    
    if len(seen_perms) == len(all_perms):
        return "".join(map(str, word))
    else:
        # Fallback: return the original classic superpermutation
        return "".join(map(str, classic_superperm(n)))

if __name__ == "__main__":
    res = construct(7)
    print(len(res))
