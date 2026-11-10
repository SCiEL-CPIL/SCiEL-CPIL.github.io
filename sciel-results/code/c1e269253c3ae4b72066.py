import time
import random
from math import gcd, exp

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    # 90s limit, keep 5s margin for safety and final return
    deadline = start_time + 85.0
    
    n = 30
    # Calculate LCM of 1..30
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
        
    # Weights w[i] = L / (i+1) for i in 0..29
    # c[i] corresponds to 1/(i+1)
    w = [L // (i + 1) for i in range(n)]
    
    best_val = float('inf')
    best_coeffs = None
    
    # Initial seed: [0]*28 + [1, -1]
    # Sum = 1/29 - 1/30 = 1/870
    seed_coeffs = [0] * 28 + [1, -1]
    S_seed = sum(c * w[i] for i, c in enumerate(seed_coeffs))
    if S_seed != 0:
        best_val = abs(S_seed)
        best_coeffs = seed_coeffs[:]
    else:
        best_coeffs = [0] * n

    def calc_sum(coeffs):
        return sum(c * w[i] for i, c in enumerate(coeffs))

    def get_energy(coeffs, S):
        if S == 0:
            return float('inf')
        return abs(S)

    random.seed(42)
    
    # Phase 1: Simulated Annealing with multiple restarts
    # We allocate ~60s for SA
    sa_deadline = start_time + 60.0
    
    # SA Parameters
    T_start = 10000.0
    T_end = 1.0
    alpha = 0.9999 # Cooling factor
    
    # Number of restarts depends on time
    # Each restart runs SA until time slice is up or convergence
    
    max_restarts = 10
    for restart in range(max_restarts):
        if time.time() > sa_deadline:
            break
            
        # Initialize current state for this SA run
        if restart == 0:
            curr = best_coeffs[:]
        elif restart % 2 == 0:
            curr = [random.choice([-1, 0, 1]) for _ in range(n)]
        else:
            # Mutate best
            if best_coeffs is not None:
                curr = best_coeffs[:]
                for _ in range(random.randint(1, 5)):
                    idx = random.randint(0, n - 1)
                    curr[idx] = random.choice([-1, 0, 1])
            else:
                curr = [random.choice([-1, 0, 1]) for _ in range(n)]
                
        curr_sum = calc_sum(curr)
        curr_energy = get_energy(curr, curr_sum)
        
        if curr_energy < best_val:
            best_val = curr_energy
            best_coeffs = curr[:]
            
        T = T_start
        # Run SA loop until time slice or energy stabilizes
        # We don't know how many steps fit, so we loop with time check
        while time.time() < sa_deadline:
            if T < T_end:
                break
                
            # Pick random index
            idx = random.randint(0, n - 1)
            old_c = curr[idx]
            
            # Pick random new value different from old
            new_c = random.choice([-1, 0, 1])
            if new_c == old_c:
                new_c = (-1 if new_c != -1 else 1) # Fallback, though choice is random from 3
            
            # Ensure new_c != old_c
            if new_c == old_c:
                # This is rare if we pick from 3, but let's be safe
                options = [v for v in [-1, 0, 1] if v != old_c]
                new_c = random.choice(options)
            
            delta_c = new_c - old_c
            delta_sum = delta_c * w[idx]
            new_sum = curr_sum + delta_sum
            new_energy = get_energy(curr, new_sum)
            
            # Check if better
            if new_energy < curr_energy:
                # Accept
                curr[idx] = new_c
                curr_sum = new_sum
                curr_energy = new_energy
                
                if curr_energy < best_val:
                    best_val = curr_energy
                    best_coeffs = curr[:]
            else:
                # Metropolis criterion
                if new_energy == float('inf'):
                    # Invalid move (sum became 0), reject
                    pass
                else:
                    delta_e = new_energy - curr_energy
                    # delta_e > 0
                    # p = exp(-delta_e / T)
                    # Use log to avoid underflow? exp is fine for reasonable T
                    p = exp(-delta_e / T)
                    if random.random() < p:
                        curr[idx] = new_c
                        curr_sum = new_sum
                        curr_energy = new_energy
                        
            T *= alpha

    # Phase 2: Local Search Polish on the best solution found by SA
    # Use the logic from the original program but optimized
    if time.time() < deadline - 2.0 and best_coeffs is not None:
        curr = best_coeffs[:]
        curr_sum = calc_sum(curr)
        curr_abs = abs(curr_sum) if curr_sum != 0 else float('inf')
        
        # 2a: Single-index steepest descent
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for idx in range(n):
                old_c = curr[idx]
                best_new_c = None
                best_new_abs = curr_abs
                for new_c in [-1, 0, 1]:
                    if new_c == old_c:
                        continue
                    delta = (new_c - old_c) * w[idx]
                    new_sum = curr_sum + delta
                    new_abs = abs(new_sum) if new_sum != 0 else float('inf')
                    if new_abs < best_new_abs:
                        best_new_abs = new_abs
                        best_new_c = new_c
                
                if best_new_c is not None:
                    curr[idx] = best_new_c
                    curr_sum += (best_new_c - old_c) * w[idx]
                    curr_abs = best_new_abs
                    improved = True
                    if curr_abs < best_val:
                        best_val = curr_abs
                        best_coeffs = curr[:]

        # 2b: Pair-move local search
        if time.time() < deadline - 1.0:
            # Precompute all possible deltas for pairs? 
            # Or just random sampling as in original, but more iterations
            # Since N=30, exhaustive pair scan is 30*29/2 * 3*3 = ~3800 * 9 = 34200 ops per pass.
            # This is fast. We can do exhaustive passes.
            
            improved = True
            while improved and time.time() < deadline - 1.0:
                improved = False
                for i1 in range(n):
                    if time.time() > deadline - 1.0:
                        break
                    for i2 in range(i1 + 1, n):
                        c1 = curr[i1]
                        c2 = curr[i2]
                        
                        best_move = None
                        best_new_abs = curr_abs
                        best_sum = curr_sum
                        
                        # Try all combinations of new values for c1, c2
                        for nc1 in [-1, 0, 1]:
                            if nc1 == c1: continue
                            for nc2 in [-1, 0, 1]:
                                if nc2 == c2: continue
                                
                                delta = (nc1 - c1) * w[i1] + (nc2 - c2) * w[i2]
                                new_sum = curr_sum + delta
                                new_abs = abs(new_sum) if new_sum != 0 else float('inf')
                                
                                if new_abs < best_new_abs:
                                    best_new_abs = new_abs
                                    best_move = (nc1, nc2)
                                    best_sum = new_sum
                                    
                        if best_move is not None:
                            curr[i1] = best_move[0]
                            curr[i2] = best_move[1]
                            curr_sum = best_sum
                            curr_abs = best_new_abs
                            improved = True
                            if curr_abs < best_val:
                                best_val = curr_abs
                                best_coeffs = curr[:]

    # Fallback
    if best_coeffs is None:
        return [0] * 28 + [1, -1]
        
    return best_coeffs
# EVOLVE-BLOCK-END
