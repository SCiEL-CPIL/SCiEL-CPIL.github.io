import time
import random
import itertools
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s limit
    
    # Handle small cases
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    # 1. Generate the standard recursive superpermutation
    # This gives length sum(k! for k=1..n)
    # For n=7: 1+2+6+24+120+720+5040 = 5913
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
            # Find maximum overlap between end of new and start of block
            overlap = 0
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    S = list(word)
    best_S = S[:]
    best_len = len(S)
    
    # 2. Initialize window counts
    counts = defaultdict(int)
    for i in range(len(S) - n + 1):
        p = tuple(S[i:i+n])
        counts[p] += 1
    
    # 3. Greedy removal pass (fast, deterministic)
    # Try to remove each character from right to left
    # A removal is valid if no window that is destroyed had count 1
    
    def try_remove(S, counts, i, n):
        """Check if removing S[i] is valid. Returns True if valid."""
        # Windows affected: those containing position i
        # Window start j where j <= i <= j+n-1, i.e., i-n+1 <= j <= i
        # Also j must be in [0, len(S)-n]
        start_old = max(0, i - n + 1)
        end_old = min(i, len(S) - n)
        
        for j in range(start_old, end_old + 1):
            p = tuple(S[j:j+n])
            if counts[p] == 1:
                return False
        return True
    
    def perform_remove(S, counts, i, n):
        """Remove S[i] and update counts."""
        # Decrement old windows
        start_old = max(0, i - n + 1)
        end_old = min(i, len(S) - n)
        old_windows = []
        for j in range(start_old, end_old + 1):
            p = tuple(S[j:j+n])
            old_windows.append(p)
        
        # Increment new windows (straddling the gap)
        # New string length is len(S)-1
        # New windows that differ: those that cross position i
        # In new string, positions < i map to S positions < i
        # In new string, positions >= i map to S positions >= i+1
        # A window at new index k straddles if it uses positions from both sides
        # k <= i-1 < k+n, i.e., i-n < k <= i-1, i.e., i-n+1 <= k <= i-1
        # Also k must be in [0, len(S)-1-n] = [0, len(S)-n-1]
        start_new = max(0, i - n + 1)
        end_new = min(i - 1, len(S) - n - 1)
        
        new_windows = []
        for k in range(start_new, end_new + 1):
            window_chars = []
            for idx in range(k, k+n):
                s_idx = idx if idx < i else idx + 1
                window_chars.append(S[s_idx])
            new_windows.append(tuple(window_chars))
        
        # Apply
        S.pop(i)
        for p in old_windows:
            counts[p] -= 1
            if counts[p] == 0:
                del counts[p]
        for p in new_windows:
            counts[p] += 1
    
    # Multiple greedy removal passes
    while time.time() < deadline:
        progress = False
        # Try removal from right to left
        for i in range(len(S) - 1, -1, -1):
            if time.time() > deadline:
                break
            if try_remove(S, counts, i, n):
                perform_remove(S, counts, i, n)
                progress = True
                if len(S) < best_len:
                    best_S = S[:]
                    best_len = len(S)
        
        if not progress:
            # No more removals possible. Try a perturbation: swap two random positions
            # and see if that enables removals
            if len(S) > 2 and random.random() < 0.3:
                # Try a random swap
                i1 = random.randint(0, len(S) - 2)
                i2 = random.randint(0, len(S) - 2)
                if i1 != i2:
                    S[i1], S[i2] = S[i2], S[i1]
                    # Rebuild counts
                    counts = defaultdict(int)
                    for j in range(len(S) - n + 1):
                        p = tuple(S[j:j+n])
                        counts[p] += 1
                    # Check if still valid (all 5040 perms present)
                    # For n=7, 5040 perms. Check count
                    if len(counts) < 5040:
                        # Invalid, revert
                        S[i1], S[i2] = S[i2], S[i1]
                        counts = defaultdict(int)
                        for j in range(len(S) - n + 1):
                            p = tuple(S[j:j+n])
                            counts[p] += 1
                    else:
                        # Try removals after swap
                        for i in range(len(S) - 1, -1, -1):
                            if time.time() > deadline:
                                break
                            if try_remove(S, counts, i, n):
                                perform_remove(S, counts, i, n)
                                if len(S) < best_len:
                                    best_S = S[:]
                                    best_len = len(S)
                        # If no improvement from swap, revert
                        if len(S) >= best_len:
                            S = best_S[:]
                            counts = defaultdict(int)
                            for j in range(len(S) - n + 1):
                                p = tuple(S[j:j+n])
                                counts[p] += 1
            else:
                break
    
    return "".join(map(str, best_S))
