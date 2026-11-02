import random
import time

def construct(q, n, radius):
    # Specific handling for the known task instance K_3(6, 1)
    if q == 3 and n == 6 and radius == 1:
        return _solve_k3_6_1()
    else:
        # Fallback to a simple greedy set cover for other parameters
        # This is not optimized but ensures validity for other inputs if any
        return _generic_greedy_cover(q, n, radius)

def _solve_k3_6_1():
    N = 729
    # Precompute balls (neighborhoods of radius 1)
    # Index i corresponds to the number i in base 3 (6 digits)
    balls = []
    for i in range(N):
        digits = []
        x = i
        for _ in range(6):
            digits.append(x % 3)
            x //= 3
        # digits is least-significant first
        # To get neighbors, we change one digit
        
        ball = {i}
        for pos in range(6):
            orig_val = digits[pos]
            for val in range(3):
                if val == orig_val:
                    continue
                # Calculate new index
                # The contribution of digit at pos is 3^pos
                # diff = (val - orig_val) * (3 ** pos)
                new_idx = i + (val - orig_val) * (3 ** pos)
                ball.add(new_idx)
        balls.append(ball)
        
    # Precompute powers of 3 for speed
    pow3 = [1, 3, 9, 27, 81, 243]
    
    # Re-compute balls using pow3 for clarity and speed in index calc
    # The previous loop was correct but let's ensure it's robust.
    # i + (val - orig) * 3^pos is correct.
    
    best_code = None
    best_size = float('inf')
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Random seed for reproducibility in debugging, but random is fine for search
    # random.seed(42) # Uncomment for debug
    
    while time.time() < deadline:
        # 1. Randomized Greedy Set Cover
        uncovered = set(range(N))
        code = []
        
        while uncovered:
            best_cand = -1
            max_cov = -1
            candidates = []
            
            # Iterate all candidates
            for c in range(N):
                # Count coverage in uncovered
                cov = 0
                for w in balls[c]:
                    if w in uncovered:
                        cov += 1
                
                if cov > max_cov:
                    max_cov = cov
                    candidates = [c]
                elif cov == max_cov:
                    candidates.append(c)
            
            if not candidates:
                break # Should not happen
                
            # Random tie-breaking to explore different solutions
            c = random.choice(candidates)
            code.append(c)
            # Remove covered words
            for w in balls[c]:
                if w in uncovered:
                    uncovered.remove(w)
            
        # 2. Reduction (Local Search: Remove redundant codewords)
        # Initialize coverage counts
        counts = [0] * N
        for c in code:
            for w in balls[c]:
                counts[w] += 1
        
        # Repeat removal passes until no change
        changed = True
        while changed:
            changed = False
            # Iterate in reverse to handle indices safely if we pop
            # We check if removing c leaves any word with count 1 (which becomes 0)
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                can_remove = True
                for w in balls[c]:
                    if counts[w] == 1:
                        can_remove = False
                        break
                if can_remove:
                    code.pop(i)
                    for w in balls[c]:
                        counts[w] -= 1
                    changed = True
            
        if len(code) < best_size:
            best_size = len(code)
            best_code = code.copy()
            
            # If we find the known optimum, we can stop
            if best_size <= 73:
                break

    # Convert indices back to lists of digits
    result = []
    for idx in best_code:
        digits = []
        x = idx
        for _ in range(6):
            digits.append(x % 3)
            x //= 3
        digits.reverse()
        result.append(digits)
        
    return result

def _generic_greedy_cover(q, n, radius):
    """
    Simple greedy set cover for general q, n, radius.
    Not optimized, but functional.
    """
    N = q ** n
    # Precompute balls
    balls = []
    for i in range(N):
        digits = []
        x = i
        for _ in range(n):
            digits.append(x % q)
            x //= q
        digits.reverse() # MSB first for consistency with standard representation, though index calc uses LSB
        
        ball = {i}
        # Generate all neighbors within radius
        # This is expensive for large n, but for small n it's fine.
        # For radius 1:
        if radius == 1:
            for pos in range(n):
                for val in range(q):
                    if digits[pos] != val:
                        new_digits = list(digits)
                        new_digits[pos] = val
                        new_idx = 0
                        for d in new_digits:
                            new_idx = new_idx * q + d
                        ball.add(new_idx)
        balls.append(ball)
        
    uncovered = set(range(N))
    code = []
    while uncovered:
        best_cand = -1
        max_cov = -1
        for c in range(N):
            cov = len(balls[c] & uncovered)
            if cov > max_cov:
                max_cov = cov
                best_cand = c
        if best_cand == -1:
            break
        code.append(best_cand)
        uncovered -= balls[best_cand]
        
    result = []
    for idx in code:
        digits = []
        x = idx
        for _ in range(n):
            digits.append(x % q)
            x //= q
        digits.reverse()
        result.append(digits)
    return result
