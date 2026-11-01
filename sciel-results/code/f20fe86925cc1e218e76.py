import time
import random
import itertools

def construct(q, n, radius):
    # This function is specific to the task: q=3, n=7, radius=1
    # But we should handle the general case if possible, or just assume the task parameters.
    # The task says "evaluator passes this task's alphabet size q, length n and covering radius".
    # So q=3, n=7, radius=1 is fixed for this task instance.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback to the original direct sum for other parameters
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Precompute spheres for all 2187 words
    # Word is an integer 0..2186
    # Convert int to base-3 digits of length 7
    # Sphere of w: w itself, and w with one digit changed to the other 2 values
    
    # Helper: int to digits
    def int_to_digits(x, n):
        digits = [0] * n
        for i in range(n):
            digits[i] = x % 3
            x //= 3
        return digits
        
    # Helper: digits to int
    def digits_to_int(digits):
        x = 0
        for i in range(n-1, -1, -1):
            x = x * 3 + digits[i]
        return x
        
    # Precompute spheres
    spheres = [[] for _ in range(2187)]
    for w in range(2187):
        digits = int_to_digits(w, n)
        spheres[w].append(w)
        for i in range(n):
            original = digits[i]
            for val in range(3):
                if val != original:
                    new_digits = digits[:]
                    new_digits[i] = val
                    spheres[w].append(digits_to_int(new_digits))
                    
    # 2. Initial set: Tetracode product (243 words)
    # Tetracode is 9 words of length 4.
    # Product with 3^3 = 27 words of length 3.
    # Total 243 words.
    
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=3):
            word_digits = h + list(tail)
            initial_codewords.append(digits_to_int(word_digits))
            
    C = initial_codewords
    in_C = [False] * 2187
    for c in C:
        in_C[c] = True
        
    # Coverage count
    coverage = [0] * 2187
    for c in C:
        for x in spheres[c]:
            coverage[x] += 1
            
    # Local Search
    # We want to minimize len(C)
    
    # To speed up, we can keep a list of indices or just the list C
    # Since C is a list, removal is O(N). N is ~200, so it's fine.
    
    # Random seed for reproducibility in evaluation? Not required, but good for testing.
    # random.seed(42) # Un-comment for debugging
    
    while time.time() < deadline:
        # Try to remove a random codeword
        if len(C) > 0:
            i = random.randint(0, len(C) - 1)
            c = C[i]
            
            # Try removal
            valid = True
            for x in spheres[c]:
                coverage[x] -= 1
                if coverage[x] == 0:
                    valid = False
                    break
            
            if not valid:
                # Restore
                for x in spheres[c]:
                    coverage[x] += 1
                # If we broke early, we need to restore the ones we decremented
                # The loop above broke, so some are decremented, some are not.
                # Better to just re-calculate or track.
                # Let's refactor the check to be safer.
                
            # Refactored Check:
            # We will write a helper to check validity of a set of changes.
            
    # Let's rewrite the loop with a cleaner check mechanism.
    
    # Reset state to initial
    C = initial_codewords[:]
    in_C = [False] * 2187
    for c in C:
        in_C[c] = True
    coverage = [0] * 2187
    for c in C:
        for x in spheres[c]:
            coverage[x] += 1
            
    while time.time() < deadline:
        i = random.randint(0, len(C) - 1)
        c = C[i]
        
        # 1. Attempt Removal
        # Decrement
        for x in spheres[c]:
            coverage[x] -= 1
            
        # Check
        valid_removal = True
        for x in spheres[c]:
            if coverage[x] == 0:
                valid_removal = False
                break
                
        if valid_removal:
            # Accept removal
            C.pop(i)
            in_C[c] = False
        else:
            # Restore
            for x in spheres[c]:
                coverage[x] += 1
                
            # 2. Attempt Swap
            # Pick a random word w not in C
            # To avoid infinite loops picking same w, we can just pick random 0..2186
            # and check in_C.
            # Probability of collision is ~200/2187 ~ 10%. Acceptable.
            
            w = random.randint(0, 2186)
            if in_C[w]:
                continue # Skip if w is already in C, effectively a removal attempt which we know might fail or succeed
                         # But we already tried removal of c. If w in C, swap is just removal of c.
                         # So if removal failed, swap with existing w will also fail.
                         # So we should pick w NOT in C.
            
            # Decrement c, Increment w
            for x in spheres[c]:
                coverage[x] -= 1
            for x in spheres[w]:
                coverage[x] += 1
                
            # Check
            valid_swap = True
            # We only need to check spheres that might have gone to 0.
            # Those are in sphere(c). sphere(w) only increases.
            # However, if x is in both, it stays same or changes by -1+1=0?
            # coverage[x] = old - 1 (from c) + 1 (from w) = old.
            # So only x in sphere(c) \ sphere(w) are at risk.
            # But checking all sphere(c) is fast (15 items).
            
            for x in spheres[c]:
                if coverage[x] == 0:
                    valid_swap = False
                    break
                    
            if valid_swap:
                # Accept swap
                C[i] = w
                in_C[c] = False
                in_C[w] = True
            else:
                # Restore
                for x in spheres[w]:
                    coverage[x] -= 1
                for x in spheres[c]:
                    coverage[x] += 1

    # Convert back to lists of digits
    result = []
    for c in C:
        result.append(int_to_digits(c, n))
        
    return result
