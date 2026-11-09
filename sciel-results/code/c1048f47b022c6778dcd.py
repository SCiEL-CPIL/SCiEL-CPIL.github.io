import time
import sys
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    
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
    L = len(word)
    
    # Encode permutations to integers for fast hashing
    # Map symbol 1..n to 0..n-1
    perm_map = {}
    all_perms = set()
    for p in permutations(range(n)):
        code = 0
        for x in p:
            code = code * n + x
        all_perms.add(code)
        
    def encode_window(w):
        code = 0
        for x in w:
            code = code * n + (x - 1)
        return code

    # Precompute window codes
    window_codes = [0] * (L - n + 1)
    code = 0
    for j in range(n):
        code = code * n + (word[j] - 1)
    window_codes[0] = code
    
    n_pow_n_minus_1 = n ** (n - 1)
    
    for i in range(1, L - n + 1):
        code = (code - (word[i-1] - 1) * n_pow_n_minus_1) * n + (word[i+n-1] - 1)
        window_codes[i] = code
        
    # Count occurrences
    counts = {}
    for c in window_codes:
        if c in all_perms:
            counts[c] = counts.get(c, 0) + 1
            
    # Map position to set of perm codes that cover it
    # pos_to_perms[i] = set of codes
    # This is memory heavy but L ~ 6000, 5040 perms.
    # Each perm covers n positions. Total entries ~ 6000 * 7 = 42000.
    pos_to_perms = [set() for _ in range(L)]
    
    for i, c in enumerate(window_codes):
        if c in all_perms:
            for j in range(n):
                pos_to_perms[i + j].add(c)
                
    # Critical positions: positions that are in any perm with count 1
    critical = [False] * L
    for i, c in enumerate(window_codes):
        if c in all_perms and counts.get(c, 0) == 1:
            for j in range(n):
                critical[i + j] = True
                
    removed = [False] * L
    current_len = L
    
    # Greedy removal
    # We maintain a queue of candidate positions
    # A position is a candidate if it is not critical and not removed
    
    # To be efficient, we scan from left to right.
    # If we remove a position, indices shift.
    # It is easier to work with a list of active positions or just modify the word and re-index?
    # Re-indexing is O(L). L is small (6000).
    
    # Let's use a list of indices to try removing.
    # We'll iterate until no changes or time limit.
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Rebuild critical set? No, update incrementally is hard with shifting indices.
        # Given L is small, let's just re-evaluate candidates.
        
        # Find first non-critical, non-removed position
        # We need to check safety again because counts changed.
        
        # Optimization: Keep a list of "safe" positions.
        # When we remove a position, only positions in the affected window range [i-n+1, i+n-1] might change status.
        
        # Let's just scan.
        # i from 0 to current_len - 1
        # If not removed[i] and not critical[i]:
        #    Check if still safe.
        #    If safe, remove.
        
        # To avoid O(N^2) scanning, we can use a pointer.
        
        i = 0
        while i < current_len:
            if time.time() >= deadline:
                break
            if not removed[i]:
                # Check if i is critical
                is_critical = False
                for c in pos_to_perms[i]:
                    if counts.get(c, 0) == 1:
                        is_critical = True
                        break
                        
                if not is_critical:
                    # Try to remove i
                    # Affected windows: starts s such that s <= i <= s+n-1
                    # s in [i-n+1, i]
                    start_min = max(0, i - n + 1)
                    start_max = min(i, current_len - n)
                    
                    # Check safety for all affected perms
                    # If any affected perm has count 1, we can't remove i (should be caught by is_critical, but counts might have changed)
                    safe = True
                    for s in range(start_min, start_max + 1):
                        c = window_codes[s]
                        if c in all_perms and counts.get(c, 0) == 1:
                            safe = False
                            break
                            
                    if safe:
                        # Remove i
                        removed[i] = True
                        current_len -= 1
                        changed = True
                        
                        # Update counts and pos_to_perms?
                        # This is the expensive part.
                        # We need to remove the windows that are destroyed or shifted.
                        
                        # Windows destroyed:
                        # Any window that included i is now invalid or shifted.
                        # Actually, removing i makes all windows containing i disappear.
                        # New windows are formed by concatenation of left and right parts.
                        
                        # This incremental update is complex.
                        # Given the small size, let's just rebuild the whole structure if we remove?
                        # No, that's O(L^2) total. 6000^2 = 36M. Might be too slow in Python.
                        
                        # Alternative: Just break and restart the loop with updated word?
                        # That's what the seed did, but slowly.
                        
                        # Let's try a different approach:
                        # Don't update incrementally. Just keep a list of "active" positions.
                        # But the window codes depend on the actual word content.
                        
                        # Let's stick to the seed's approach but optimize the inner loop.
                        # The seed rebuilds window codes from scratch.
                        # We can do better:
                        # When we remove i, we can update the window_codes array in O(n).
                        
                        # But the indices shift.
                        # Let's just use the "rebuild from scratch" but only if we removed something.
                        # And do it in a tight loop.
                        
                        # To make this faster, we will break out of the inner while and restart the outer while.
                        # This ensures correctness.
                        
                        # Rebuild word, window_codes, counts, pos_to_perms, critical
                        new_word = [word[k] for k in range(current_len) if not removed[k]]
                        word = new_word
                        L = len(word)
                        
                        # Rebuild window codes
                        window_codes = [0] * (L - n + 1)
                        code = 0
                        for j in range(n):
                            code = code * n + (word[j] - 1)
                        window_codes[0] = code
                        for k in range(1, L - n + 1):
                            code = (code - (word[k-1] - 1) * n_pow_n_minus_1) * n + (word[k+n-1] - 1)
                            window_codes[k] = code
                            
                        # Rebuild counts
                        counts = {}
                        for c in window_codes:
                            if c in all_perms:
                                counts[c] = counts.get(c, 0) + 1
                                
                        # Rebuild pos_to_perms
                        pos_to_perms = [set() for _ in range(L)]
                        for k, c in enumerate(window_codes):
                            if c in all_perms:
                                for j in range(n):
                                    pos_to_perms[k + j].add(c)
                                    
                        # Rebuild critical
                        critical = [False] * L
                        for k, c in enumerate(window_codes):
                            if c in all_perms and counts.get(c, 0) == 1:
                                for j in range(n):
                                    critical[k + j] = True
                                    
                        # Reset removed
                        removed = [False] * L
                        current_len = L
                        i = 0 # Restart scan from beginning
                        break # Break inner while to restart outer while logic cleanly?
                        # Actually, we can just continue the outer while loop.
                        
            i += 1
            
    # Final word
    return "".join(map(str, [w for w, r in zip(word, removed) if not r]))

if __name__ == "__main__":
    res = construct(7)
    print(len(res))
