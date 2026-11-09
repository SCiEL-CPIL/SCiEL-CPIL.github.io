import time
import sys
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
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
    # Use base-n encoding
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
    
    # Multi-pass greedy pruning
    # We remove a character at position i if no permutation with count==1
    # is in the affected windows
    
    def try_prune(word, window_codes, counts, n, deadline):
        i = 0
        L = len(word)
        removed_any = False
        
        while i < L and time.time() < deadline:
            L = len(word)
            if L < n:
                break
            
            # Windows affected by removing word[i]: starts s where s <= i <= s+n-1
            # s in [max(0, i-n+1), min(i, L-n)]
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
                i += 1
                continue
            
            # Removal is safe. Update counts.
            # Decrement old windows
            for s in range(start_min, start_max + 1):
                code = window_codes[s]
                if code in all_perms:
                    counts[code] -= 1
                    if counts[code] == 0:
                        del counts[code]
            
            # Increment new windows
            # New window at start s (s in [start_min, i]):
            # W'[s:s+n] = W[s:i] + W[i+1:s+n+1]
            # But we need to be careful with the window_codes array
            # After removal, the array shifts
            
            # Actually, let's just rebuild affected window codes
            # New windows start at s in [start_min, i] in the new word
            # In the old word, these correspond to:
            # For s < i: new_window = word[s:i] + word[i+1:s+n+1]
            # For s == i: new_window = word[i+1:i+n+1]
            
            new_start_min = start_min
            new_start_max = min(i, L - 1 - n)
            
            new_codes = {}
            for s in range(new_start_min, new_start_max + 1):
                if s < i:
                    w_new = word[s:i] + word[i+1:s+n+1]
                else:
                    w_new = word[i+1:i+n+1]
                code = encode_window(w_new)
                new_codes[s] = code
                if code in all_perms:
                    counts[code] = counts.get(code, 0) + 1
            
            # Perform removal
            word.pop(i)
            
            # Update window_codes array
            # The array length decreases by 1
            # Windows at positions >= i shift left by 1
            # Windows at positions < i are unchanged
            # But windows in [start_min, i] are replaced by new_codes
            
            new_len = len(word)
            new_wc = [0] * (new_len - n + 1)
            
            for s in range(new_len - n + 1):
                if s in new_codes:
                    new_wc[s] = new_codes[s]
                elif s < start_min:
                    new_wc[s] = window_codes[s]
                else:
                    # This window was at position s+1 in the old array
                    # (since we removed position i, and s >= i)
                    # Actually, for s >= i, the new window at s corresponds to
                    # old window at s+1 (if s+1 < old_len - n + 1)
                    old_s = s + 1
                    if old_s < len(window_codes):
                        new_wc[s] = window_codes[old_s]
                    else:
                        # Recompute
                        w = word[s:s+n]
                        new_wc[s] = encode_window(w)
            
            window_codes = new_wc
            removed_any = True
            # Don't increment i, next element shifted into position i
        
        return word, window_codes, counts, removed_any
    
    # Multiple passes
    for _ in range(50):
        if time.time() >= deadline:
            break
        word, window_codes, counts, removed_any = try_prune(word, window_codes, counts, n, deadline)
        if not removed_any:
            break
    
    return "".join(map(str, word))

if __name__ == "__main__":
    res = construct(7)
    print(len(res))
