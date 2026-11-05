import itertools
import time
import heapq
import random
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "123412314231243121342132413214321"
    if n == 5:
        # Standard construction is optimal for n=5
        pass

    def build_standard(n):
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            # Extract all (m-1)-permutations from the current word
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                # Find maximum overlap with the end of 'new'
                overlap = 0
                max_ov = min(len(new), len(block))
                for o in range(max_ov, -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return word

    word = build_standard(n)
    L = len(word)
    
    if n <= 6:
        return "".join(map(str, word))

    # Optimization for n=7
    # We want to minimize length by deleting characters.
    
    def try_optimize(word, time_limit):
        # Initial state
        w = list(word)
        L = len(w)
        
        # Frequency of each n-permutation
        freq = defaultdict(int)
        for i in range(L - n + 1):
            p = tuple(w[i:i+n])
            freq[p] += 1
            
        # Check if all perms are present
        # For n=7, there are 5040 perms.
        # We assume the standard construction has all of them.
        
        # Heap of candidate positions to delete.
        # A position i is a candidate if it is "safe".
        # Safe means: for all windows containing i, freq > 1.
        
        # To avoid O(N*N) initialization, we can just pick random positions.
        # But a heap of "currently safe" positions is better.
        
        # Let's use a simple random restart strategy with greedy deletion.
        best_len = L
        best_word = w[:]
        
        for seed in range(1000): # Limit restarts by time
            if time.time() > time_limit:
                break
            
            random.seed(seed)
            w = list(word)
            L = len(w)
            freq = defaultdict(int)
            for i in range(L - n + 1):
                p = tuple(w[i:i+n])
                freq[p] += 1
            
            # List of positions that are currently safe
            # We can't maintain this perfectly easily, so we use a random sampling approach
            # or a heap with lazy validation.
            
            # Let's use a heap of all positions 1..L-2, and validate on pop.
            # This is O(L log L) per restart, which is ~6000 * 13 = 78k ops.
            # 1000 restarts = 78M ops. Might be slow in Python.
            
            # Alternative: Just pick random positions from 1 to L-2.
            # If it's safe, delete.
            # If not, try another.
            # This is fast but might get stuck.
            
            # Let's combine:
            # 1. Try to delete from a set of "promising" positions.
            # 2. If stuck, restart.
            
            # To speed up, we'll use a local variable for freq and w.
            
            # We'll try to delete as many as possible in one pass.
            deletions = 0
            max_failures = 1000
            failures = 0
            
            while failures < max_failures:
                if time.time() > time_limit:
                    break
                
                if L <= n:
                    break
                    
                # Pick a random position
                pos = random.randint(1, L - 2)
                
                # Check if pos is safe
                # Affected windows: [pos-n+1, pos]
                lo = max(0, pos - n + 1)
                hi = min(pos, L - n)
                
                is_safe = True
                for w_idx in range(lo, hi + 1):
                    p = tuple(w[w_idx:w_idx+n])
                    if freq[p] <= 1:
                        is_safe = False
                        break
                
                if is_safe:
                    # Delete
                    # Update freq
                    for w_idx in range(lo, hi + 1):
                        p = tuple(w[w_idx:w_idx+n])
                        freq[p] -= 1
                        if freq[p] == 0:
                            del freq[p]
                    
                    # Perform deletion
                    w = w[:pos] + w[pos+1:]
                    L -= 1
                    deletions += 1
                    
                    # Update new windows
                    new_lo = max(0, pos - n)
                    new_hi = min(pos, L - n)
                    for w_idx in range(new_lo, new_hi + 1):
                        p = tuple(w[w_idx:w_idx+n])
                        freq[p] += 1
                        
                    failures = 0 # Reset
                else:
                    failures += 1
            
            if L < best_len:
                best_len = L
                best_word = w[:]
                
        return best_word

    # Run optimization
    optimized_word = try_optimize(word, deadline)
    
    # Final check
    # Verify that optimized_word contains all permutations
    # This is expensive (5040 perms * 5905 windows), but necessary for correctness.
    # Actually, we can just check the length. If it's 5905, it's likely valid.
    # But to be safe, we should verify.
    
    # For n=7, verification is:
    # Create a set of all windows in optimized_word.
    # Check if it equals the set of all permutations of 1..7.
    
    # This takes ~0.1s.
    
    required = set(itertools.permutations(range(1, n + 1)))
    present = set()
    for i in range(len(optimized_word) - n + 1):
        present.add(tuple(optimized_word[i:i+n]))
        
    if present != required:
        # Fallback to standard
        return "".join(map(str, word))
        
    return "".join(map(str, optimized_word))
