"""Superpermutation construction with time-bounded improvement search."""
import math
import time
import random
from itertools import permutations
import numpy as np

# EVOLVE-BLOCK-START

def construct(n):
    start = time.time()
    DEADLINE = 130.0  # safety margin before 150s kill

    def build_classic(n):
        """Classic recursive superpermutation construction."""
        if n == 1:
            return [1]
        if n == 2:
            return [1, 2, 1]
        if n == 3:
            return [1, 2, 3, 1, 2, 3, 1]
        
        # For n >= 4, use the standard recursive construction
        # S(n) is built from S(n-1) by inserting n
        prev = build_classic(n - 1)
        
        # Find all occurrences of (n-1) permutations in prev
        # Standard construction: for each permutation p of 1..(n-1) that appears 
        # as a window in prev, insert n after it
        
        # Actually, let's use the simpler iterative approach
        word = [1]
        for m in range(2, n + 1):
            # Collect all distinct (m-1)-length windows that are permutations
            perms = []
            seen = set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            
            # Build new word by expanding each permutation
            new_word = []
            for p in perms:
                # For each permutation p, we create: p, m, p (but with overlap)
                block = list(p) + [m]
                # Check overlap with current end of new_word
                max_overlap = min(len(new_word), len(block))
                overlap = 0
                for o in range(max_overlap, 0, -1):
                    if new_word[len(new_word) - o:] == block[:o]:
                        overlap = o
                        break
                new_word.extend(block[overlap:])
            
            word = new_word
        
        return word

    def is_valid(w, n):
        """Check if word contains all permutations of 1..n."""
        if len(w) < n:
            return False
        target_count = math.factorial(n)
        seen = set()
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i + n])
            if len(set(window)) == n:
                seen.add(window)
                if len(seen) == target_count:
                    return True
        return len(seen) == target_count

    def count_missing_perms(w, n):
        """Count how many permutations are missing."""
        if len(w) < n:
            return math.factorial(n)
        seen = set()
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i + n])
            if len(set(window)) == n:
                seen.add(window)
        return math.factorial(n) - len(seen)

    def greedy_remove_optimized(word, n, deadline):
        """Greedy removal with efficient validity checking using reference counting."""
        current_word = list(word)
        current_len = len(current_word)
        target_count = math.factorial(n)
        
        if target_count == 0:
            return current_word
        
        # Precompute which windows are valid permutations
        # Use a dictionary to count occurrences of each permutation
        perm_count = {}
        for i in range(len(current_word) - n + 1):
            window = tuple(current_word[i:i + n])
            if len(set(window)) == n:
                perm_count[window] = perm_count.get(window, 0) + 1
        
        distinct_perms = len(perm_count)
        
        # Helper to check if a single position removal preserves all permutations
        def can_remove(pos):
            if pos < 0 or pos >= len(current_word):
                return False
            
            # Windows that contain position pos: from max(0, pos-n+1) to min(pos, len(current_word)-n)
            affected_start = max(0, pos - n + 1)
            affected_end = min(pos, len(current_word) - n)
            
            # Check if any permutation becomes undercounted
            for w_start in range(affected_start, affected_end + 1):
                window = tuple(current_word[w_start:w_start + n])
                if len(set(window)) == n:
                    cnt = perm_count.get(window, 0)
                    if cnt == 1:
                        # This permutation would be lost
                        return False
            
            return True
        
        # Helper to actually remove a position and update perm_count
        def remove_position(pos):
            nonlocal current_word, current_len, perm_count, distinct_perms
            
            affected_start = max(0, pos - n + 1)
            affected_end = min(pos, len(current_word) - n)
            
            # Decrement counts for windows containing pos
            for w_start in range(affected_start, affected_end + 1):
                window = tuple(current_word[w_start:w_start + n])
                if len(set(window)) == n:
                    cnt = perm_count[window]
                    if cnt == 1:
                        del perm_count[window]
                        distinct_perms -= 1
                    else:
                        perm_count[window] = cnt - 1
            
            # Perform the removal
            current_word = current_word[:pos] + current_word[pos + 1:]
            current_len -= 1
            
            # Increment counts for new windows that form
            # After removing position pos, new windows can form
            # Windows that now include positions that weren't in a window before
            # New window starts: from max(0, pos-n+1) to min(pos-1, len(current_word)-n)
            # Actually, after removal, the word is shorter. New windows are those 
            # that span the gap. Let me think...
            
            # When we remove position pos, the character at pos+1 shifts left.
            # New windows that form are those ending at positions that now have a 
            # character from the right side.
            
            # Windows in the new word:
            # For each start s in the new word, the window is new_word[s:s+n]
            # A window is "new" if it wasn't a valid window before (or didn't exist)
            # Windows that could be new: those that include the shifted character
            # The character that was at pos+1 is now at pos.
            # So windows that include position pos in the new word are "potentially new"
            # In the new word, position pos has the old character from pos+1.
            
            new_len = len(current_word)
            if new_len >= n:
                for w_start in range(max(0, pos - n + 1), min(pos + 1, new_len - n + 1)):
                    window = tuple(current_word[w_start:w_start + n])
                    if len(set(window)) == n:
                        if window not in perm_count:
                            perm_count[window] = 1
                            distinct_perms += 1
                        else:
                            perm_count[window] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try positions in random order
            indices = list(range(len(current_word)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline:
                    break
                if distinct_perms != target_count:
                    break
                
                if can_remove(i):
                    remove_position(i)
                    improved = True
                    # Don't restart - continue with updated indices
                    # But indices may be stale, so let's just break and restart
                    indices = list(range(len(current_word)))
                    random.shuffle(indices)
        
        return current_word

    # Build initial superpermutation
    word = build_classic(n)
    
    # Verify initial word is valid
    if not is_valid(word, n):
        # Fallback
        return "".join(map(str, word))
    
    initial_len = len(word)
    
    # Try greedy removal
    if time.time() < DEADLINE - 5.0:
        improved_word = greedy_remove_optimized(word, n, DEADLINE - 5.0)
        if is_valid(improved_word, n):
            word = improved_word
    
    # Second pass: try more aggressive removal with different strategies
    if time.time() < DEADLINE - 3.0:
        # Try removal from left to right, then right to left
        current = list(word)
        target_count = math.factorial(n)
        
        def try_remove_pass(arr, direction):
            arr = list(arr)
            changed = True
            while changed and time.time() < DEADLINE - 3.0:
                changed = False
                if direction == 'l2r':
                    idxs = range(len(arr))
                else:
                    idxs = range(len(arr) - 1, -1, -1)
                
                for i in idxs:
                    if time.time() > DEADLINE - 3.0:
                        break
                    
                    # Quick check: remove and verify
                    new_arr = arr[:i] + arr[i + 1:]
                    if len(new_arr) >= n:
                        seen = set()
                        for j in range(len(new_arr) - n + 1):
                            w = tuple(new_arr[j:j + n])
                            if len(set(w)) == n:
                                seen.add(w)
                                if len(seen) == target_count:
                                    break
                        if len(seen) == target_count:
                            arr = new_arr
                            changed = True
            
            return arr
        
        word2 = try_remove_pass(current, 'l2r')
        if is_valid(word2, n) and len(word2) < len(word):
            word = word2
        
        word3 = try_remove_pass(current, 'r2l')
        if is_valid(word3, n) and len(word3) < len(word):
            word = word3
    
    return "".join(map(str, word))

# EVOLVE-BLOCK-END
