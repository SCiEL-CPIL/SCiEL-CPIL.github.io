import math
import random

def construct():
    # 1. Precompute LCM and scaled values
    # LCM(1..30)
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = 1
    for i in range(1, 31):
        L = lcm(L, i)
        
    A = [0] * 30
    for i in range(30):
        A[i] = L // (i + 1)
        
    # 2. Evaluation function using integers
    def eval_int(c):
        s = 0
        for i in range(30):
            if c[i] == 1:
                s += A[i]
            elif c[i] == -1:
                s -= A[i]
        return s

    # 3. Initial solution: Alternating signs
    # c = [1, -1, 1, -1, ...]
    c = [1 if i % 2 == 0 else -1 for i in range(30)]
    
    best_s = eval_int(c)
    if best_s == 0:
        # Perturb if zero
        c[0] = -c[0]
        best_s = eval_int(c)
        
    best_c = c[:]
    
    # 4. Local Search with Integer Arithmetic
    # We want to minimize abs(s) where s != 0
    
    # To speed up, we can precompute the effect of flipping each bit.
    # delta[i] = change in sum if we flip c[i]
    # If c[i] is 1, flipping to -1 changes sum by -2*A[i]
    # If c[i] is -1, flipping to 1 changes sum by 2*A[i]
    # If c[i] is 0, flipping to 1 changes sum by A[i], to -1 by -A[i]
    
    # We'll do a simple hill climbing with random restarts
    # But we need to be careful about the 60s limit.
    
    # Let's use a more structured search: 
    # Try all single flips, then all double flips?
    # Single flips: 30 * 2 = 60 checks per step.
    # Double flips: 30*29/2 * 4 = ~1700 checks.
    
    # Let's implement a function to try improving the current sum
    def try_improve(c, s):
        improved = False
        # Single flips
        for i in range(30):
            # Try changing c[i] to -1, 0, 1
            old_val = c[i]
            for new_val in [-1, 0, 1]:
                if new_val == old_val:
                    continue
                # Calculate new sum
                if new_val == 0:
                    new_s = s - old_val * A[i]
                else:
                    new_s = s + (new_val - old_val) * A[i]
                
                if new_s != 0 and abs(new_s) < abs(s):
                    c[i] = new_val
                    s = new_s
                    improved = True
                    # Don't break immediately, continue to find more improvements in this pass?
                    # Or break to restart the scan?
                    # Restarting scan is safer to ensure convergence.
                    break
            if improved:
                break
        return improved, s

    # Random restarts
    best_abs = abs(best_s)
    
    # We need to be efficient.
    # Let's run for a fixed number of iterations or time.
    # Since we don't have time measurement, we estimate iterations.
    # 1000 restarts, each with 1000 steps?
    
    import time
    start_time = time.time()
    time_limit = 50 # Leave 10s for safety
    
    for restart in range(10000):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize current state
        if restart == 0:
            c = best_c[:]
            s = best_s
        else:
            # Random initialization
            c = [random.choice([-1, 0, 1]) for _ in range(30)]
            s = eval_int(c)
            if s == 0:
                # Ensure non-zero
                c[0] = 1 if c[0] != 1 else -1
                s = eval_int(c)
        
        # Local search
        steps = 0
        while steps < 500: # Limit steps per restart
            if time.time() - start_time > time_limit:
                break
            improved, s = try_improve(c, s)
            if not improved:
                # Try a random kick to escape local minimum
                idx = random.randint(0, 29)
                old_val = c[idx]
                new_val = random.choice([-1, 0, 1])
                while new_val == old_val:
                    new_val = random.choice([-1, 0, 1])
                c[idx] = new_val
                s += (new_val - old_val) * A[idx]
                if s == 0:
                    # If we hit 0, kick again
                    idx2 = random.randint(0, 29)
                    old_val2 = c[idx2]
                    new_val2 = random.choice([-1, 0, 1])
                    while new_val2 == old_val2:
                        new_val2 = random.choice([-1, 0, 1])
                    c[idx2] = new_val2
                    s += (new_val2 - old_val2) * A[idx2]
                steps += 1
            else:
                steps += 1
                
        if abs(s) < best_abs and s != 0:
            best_abs = abs(s)
            best_s = s
            best_c = c[:]
            
    return best_c
