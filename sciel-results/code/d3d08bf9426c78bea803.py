import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = 150.0
    
    total_words = q ** n
    if total_words > 100000:
        # Fallback for larger problems
        # For now, assume small n, q
        pass

    # Precompute balls
    balls = []
    for c in range(total_words):
        digits = []
        temp = c
        for _ in range(n):
            digits.append(temp % q)
            temp //= q
        digits.reverse()
        
        ball = [c]
        for pos in range(n):
            orig = digits[pos]
            for nv in range(q):
                if nv == orig: continue
                nd = digits[:]
                nd[pos] = nv
                nw = 0
                for d in reversed(nd):
                    nw = nw * q + d
                ball.append(nw)
        balls.append(ball)
        
    balls_np = [np.array(b, dtype=np.int32) for b in balls]
    
    best_solution = None
    best_len = float('inf')
    
    # Run multiple random restarts
    max_restarts = 100 # We will check time inside
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        uncovered = np.ones(total_words, dtype=bool)
        solution = []
        
        while np.any(uncovered):
            if time.time() > deadline - 5:
                break
                
            best_c = -1
            max_covered = 0
            candidates = []
            
            # Find all candidates with max coverage
            # To speed up, we can just find the max and collect ties
            # But scanning all is O(N)
            for c in range(total_words):
                cnt = np.sum(uncovered[balls_np[c]])
                if cnt > max_covered:
                    max_covered = cnt
                    candidates = [c]
                elif cnt == max_covered and cnt > 0:
                    candidates.append(c)
            
            if max_covered == 0:
                break
                
            # Randomly pick from candidates to diversify
            best_c = random.choice(candidates)
            
            solution.append(best_c)
            uncovered[balls_np[best_c]] = False
            
        if len(solution) < best_len:
            best_len = len(solution)
            best_solution = solution[:]
            
            # If we hit the known lower bound or a very good number, we can stop early?
            # Lower bound is 403. If we get close, keep searching for a bit.
            if best_len <= 410:
                break

    if best_solution is None:
        # Fallback to seed logic if something went wrong
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    final_solution = []
    for c in best_solution:
        digits = []
        temp = c
        for _ in range(n):
            digits.append(temp % q)
            temp //= q
        digits.reverse()
        final_solution.append(digits)
        
    return final_solution
