import time
import itertools
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # 1. Generate initial superpermutation
    word = [1]
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    if time.time() > deadline:
        return "".join(map(str, word))
    
    # 2. Precompute required permutations as bytes
    # n=7, symbols 1..7. Bytes are 0..255, so 1..7 fits.
    required_perms = set(bytes(p) for p in itertools.permutations(range(1, n + 1)))
    num_perms = len(required_perms)
    
    # 3. Local Search
    improved = list(word)
    # Convert to bytes for faster slicing
    # But we need to modify it.
    # We can keep it as a list, and convert to bytes for validity check.
    # Or keep it as bytes and use bytearray?
    # bytearray supports slicing and modification.
    
    curr = bytearray(improved)
    
    def is_valid_fast(b):
        if len(b) < n:
            return False
        if len(b) - n + 1 < num_perms:
            return False
        
        found = set()
        # b is a bytearray or bytes
        for i in range(len(b) - n + 1):
            window = b[i:i+n]
            if window in required_perms:
                found.add(window)
                if len(found) == num_perms:
                    return True
        return len(found) == num_perms
    
    random.seed(42)
    
    # Optimization: If we remove a char, the length decreases by 1.
    # We can try to remove from the end first?
    # The recursive construction appends blocks. The end might be less redundant.
    # Actually, the beginning is "1", then "121", etc.
    # Let's just use random.
    
    while time.time() < deadline:
        success = False
        # Try 50 random removals
        for _ in range(50):
            if time.time() > deadline:
                break
            if len(curr) <= n:
                break
                
            i = random.randint(0, len(curr) - 1)
            
            # Check if removal is valid
            # Create a copy? No, we can check without modifying if we slice.
            # But slicing a bytearray creates a new bytearray.
            # candidate = curr[:i] + curr[i+1:]
            
            # To avoid creating the candidate if it's likely invalid,
            # we can check if the character at i is "essential".
            # But that's hard.
            
            candidate = curr[:i] + curr[i+1:]
            
            if is_valid_fast(candidate):
                curr = candidate
                success = True
                break
        
        # If no success, we might try a different strategy?
        # For now, just loop.

    return "".join(map(str, curr))
