import time
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # 1. Standard recursive construction
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
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    if n < 5:
        return "".join(map(str, word))
    
    # 2. Optimization Phase
    current_word = word
    current_len = len(current_word)
    
    # Precompute permutation index map for fast lookup
    perm_map = {}
    idx = 0
    for p in permutations(range(1, n+1)):
        perm_map[p] = idx
        idx += 1
    
    target_count = idx  # n!
    
    # Convert word to list of ints for faster processing
    def word_to_list(w):
        if isinstance(w, str):
            return [int(c) for c in w]
        return list(w)
    
    current_word = word_to_list(current_word)
    
    # Helper to get window indices for a given position
    # When we remove position i, windows [i-n+1, i] are affected
    def get_affected_windows(pos, L):
        """Return list of window start positions affected by removing pos"""
        starts = []
        for start in range(max(0, pos - n + 1), min(pos + 1, L - n + 1)):
            starts.append(start)
        return starts
    
    # Fast validity check using permutation map
    def is_valid_fast(w):
        seen = set()
        L = len(w)
        if L < n:
            return False
        for i in range(L - n + 1):
            window = tuple(w[i:i+n])
            if len(set(window)) == n:
                if window in perm_map:
                    seen.add(perm_map[window])
                else:
                    # This shouldn't happen for valid permutations
                    seen.add(window)
        return len(seen) == target_count
    
    # Optimization: Try removing symbols, checking validity incrementally
    # Strategy: For each position, try removing it and check if all permutations still present
    
    improved = True
    pass_count = 0
    while improved and time.time() < deadline:
        improved = False
        pass_count += 1
        
        # Shuffle positions to avoid bias
        positions = list(range(len(current_word)))
        random.shuffle(positions)
        
        for pos in positions:
            if time.time() > deadline - 5:
                break
            
            L = len(current_word)
            if L <= n:
                break
            
            # Skip positions that would break the start or end
            # Actually, we can try any position, but need to check validity
            
            # Try removing position pos
            new_word = current_word[:pos] + current_word[pos+1:]
            new_L = len(new_word)
            
            if new_L < n:
                continue
            
            # Check validity
            if is_valid_fast(new_word):
                current_word = new_word
                improved = True
                # Continue with next position (indices are now shifted, but we continue)
                # The positions list is stale, but we keep going for a few more
                # After a removal, restart the pass
                break
        
        # If no improvement in this pass, try a different strategy
        if not improved:
            # Try greedy: find positions where removal is most likely to work
            # A position is "safe" to remove if it's in windows that appear multiple times
            # For now, just do another random pass
            if pass_count > 50 and time.time() > deadline - 30:
                break
    
    return "".join(map(str, current_word))
