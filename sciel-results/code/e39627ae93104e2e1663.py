import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    M = 3**n
    # Precompute balls
    # Each word i has ball: i, and i +/- 3^pos for pos in 0..n-1
    # Note: modulo 3 arithmetic.
    # i + 3^pos might wrap?
    # Digits are 0,1,2.
    # +1: if digit is 2, becomes 0, subtract 3*3^pos.
    # -1: if digit is 0, becomes 2, add 3*3^pos.
    
    balls = []
    for i in range(M):
        ball = [i]
        for pos in range(n):
            p = 3**pos
            digit = (i // p) % 3
            if digit < 2:
                ball.append(i + p)
            else:
                ball.append(i - 2*p)
            
            if digit > 0:
                ball.append(i - p)
            else:
                ball.append(i + 2*p)
        balls.append(ball)
        
    # Convert to numpy array for fast indexing
    # balls_arr shape (M, 17)
    balls_arr = np.array(balls, dtype=np.int32)
    
    # Greedy Set Cover
    uncovered = np.ones(M, dtype=bool)
    code = []
    
    # Optimization: We can stop if uncovered is empty
    while np.any(uncovered):
        # Calculate coverage for all candidates
        # uncovered[balls_arr] gives shape (M, 17)
        # Sum along axis 1
        coverage = uncovered[balls_arr].sum(axis=1)
        
        # Find best candidate
        best_c = np.argmax(coverage)
        
        # If best coverage is 0, something is wrong (should not happen if uncovered is not empty)
        if coverage[best_c] == 0:
            break
            
        code.append(best_c)
        # Mark as covered
        uncovered[balls_arr[best_c]] = False
        
    # Local Search to improve
    # Initialize counts
    in_code = np.zeros(M, dtype=bool)
    for c in code:
        in_code[c] = True
        
    count = np.zeros(M, dtype=np.int32)
    for c in code:
        count[balls_arr[c]] += 1
        
    improved = True
    while improved:
        improved = False
        
        # Phase 1: Remove redundant
        to_remove = []
        for c in code:
            # Check if any point in ball[c] has count 1
            # If min(count[ball[c]]) > 1, redundant
            if np.min(count[balls_arr[c]]) > 1:
                to_remove.append(c)
                
        if to_remove:
            for c in to_remove:
                code.remove(c)
                in_code[c] = False
                count[balls_arr[c]] -= 1
            improved = True
            continue
            
        # Phase 2: Swap
        swapped = False
        for c in code:
            unique_points = balls_arr[c][count[balls_arr[c]] == 1]
            if len(unique_points) == 0:
                # Redundant, should have been caught
                code.remove(c)
                in_code[c] = False
                count[balls_arr[c]] -= 1
                improved = True
                swapped = True
                break
                
            # Find candidates covering all unique_points
            # Intersection of balls of unique points
            # Start with ball of first unique point
            candidates = set(balls[unique_points[0]])
            for p in unique_points[1:]:
                candidates.intersection_update(balls[p])
            candidates -= set(code)
            
            if candidates:
                c_new = random.choice(list(candidates))
                code.remove(c)
                in_code[c] = False
                count[balls_arr[c]] -= 1
                
                code.append(c_new)
                in_code[c_new] = True
                count[balls_arr[c_new]] += 1
                
                improved = True
                swapped = True
                break
                
        if swapped:
            continue

    # Return as list of lists of ints
    def to_digits(x):
        digits = [0]*n
        for i in range(n):
            digits[i] = x % 3
            x //= 3
        return digits

    return [list(map(int, to_digits(c))) for c in code]
