import numpy as np
import itertools
import random

def construct(q, n, radius):
    # The task is fixed to q=3, n=8, radius=1
    if q != 3 or n != 8 or radius != 1:
        # Fallback to direct sum for other parameters if needed, though task is specific
        # But for safety, we can just return the direct sum for general case
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Specific optimization for q=3, n=8, r=1
    M = 3**8
    # Precompute balls for all words
    # Word is an integer 0..M-1. Convert to base 3 digits.
    # Ball of a word c: c itself + 8 neighbors (change one digit to +1 or -1 mod 3)
    
    # Helper to get digits
    def to_digits(x):
        digits = [0]*n
        for i in range(n):
            digits[i] = x % 3
            x //= 3
        return digits
    
    def to_int(digits):
        x = 0
        for i in range(n-1, -1, -1):
            x = x * 3 + digits[i]
        return x

    balls = [None] * M
    for i in range(M):
        digits = to_digits(i)
        ball = [i]
        for pos in range(n):
            # +1
            d_plus = digits.copy()
            d_plus[pos] = (d_plus[pos] + 1) % 3
            ball.append(to_int(d_plus))
            # -1
            d_minus = digits.copy()
            d_minus[pos] = (d_minus[pos] - 1) % 3
            ball.append(to_int(d_minus))
        balls[i] = ball

    # Initial code: Direct Sum
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            word = h + list(tail)
            initial_code.append(to_int(word))
    
    code = set(initial_code)
    in_code = np.zeros(M, dtype=bool)
    for c in code:
        in_code[c] = True
        
    count = np.zeros(M, dtype=np.int32)
    for c in code:
        for p in balls[c]:
            count[p] += 1
            
    # Local Search
    # We will try to remove redundant codewords and swap to enable more removals.
    # To avoid getting stuck in a specific local minimum, we can run multiple iterations or randomize.
    # Given the time limit, we'll run a tight loop.
    
    improved = True
    while improved:
        improved = False
        
        # Phase 1: Remove redundant codewords
        to_remove = []
        for c in list(code):
            # Check if all points in ball[c] are covered by >1 codeword
            redundant = True
            for p in balls[c]:
                if count[p] == 1:
                    redundant = False
                    break
            if redundant:
                to_remove.append(c)
                
        if to_remove:
            for c in to_remove:
                code.remove(c)
                in_code[c] = False
                for p in balls[c]:
                    count[p] -= 1
            improved = True
            continue # Restart loop as counts changed
        
        # Phase 2: Try to swap a codeword to enable removal of others
        # For each c in code, find its unique points U_c
        # If there is a c_new not in code such that U_c is subset of ball[c_new], swap.
        swapped = False
        for c in list(code):
            # Find unique points
            unique_points = []
            for p in balls[c]:
                if count[p] == 1:
                    unique_points.append(p)
            
            if not unique_points:
                # Should have been removed in Phase 1, but maybe logic gap?
                # If no unique points, it's redundant.
                code.remove(c)
                in_code[c] = False
                for p in balls[c]:
                    count[p] -= 1
                improved = True
                swapped = True
                break
                
            # Find candidates for c_new
            # Candidates must cover all points in unique_points
            # Start with ball of first unique point
            candidates = set(balls[unique_points[0]])
            for p in unique_points[1:]:
                candidates.intersection_update(balls[p])
                
            # Filter out existing code
            candidates -= code
            
            if candidates:
                # Pick a candidate
                # Heuristic: pick one that covers more points with count > 1? 
                # Or just random.
                c_new = random.choice(list(candidates))
                
                # Perform swap
                code.remove(c)
                in_code[c] = False
                for p in balls[c]:
                    count[p] -= 1
                    
                code.add(c_new)
                in_code[c_new] = True
                for p in balls[c_new]:
                    count[p] += 1
                    
                improved = True
                swapped = True
                break # Restart loop
                
        if swapped:
            continue
            
        # If no improvement, break
        break

    return [list(map(int, to_digits(c))) for c in code]
