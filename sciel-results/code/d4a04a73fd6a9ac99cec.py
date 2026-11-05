import time
import random
import math
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # Calculate LCM of 1..30
    L = 1
    for i in range(1, 31):
        L = L * i // gcd(L, i)
        
    weights = [L // i for i in range(1, 31)]
    
    best_abs_N = float('inf')
    best_coeffs = None
    
    def get_N(coeffs):
        return sum(c * w for c, w in zip(coeffs, weights))
    
    def polish(coeffs, current_N):
        """Greedy 1-flip local search to find local minimum."""
        nonlocal best_abs_N, best_coeffs
        current_abs_N = abs(current_N)
        if current_N != 0 and current_abs_N < best_abs_N:
            best_abs_N = current_abs_N
            best_coeffs = coeffs[:]
            if best_abs_N == 1:
                return True
        
        improved = True
        while improved:
            improved = False
            best_move = None
            best_new_abs_N = current_abs_N
            
            for i in range(30):
                old_val = coeffs[i]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    delta = (new_val - old_val) * weights[i]
                    new_N = current_N + delta
                    if new_N == 0:
                        continue
                    new_abs_N = abs(new_N)
                    if new_abs_N < best_new_abs_N:
                        best_new_abs_N = new_abs_N
                        best_move = (i, new_val)
            
            if best_move:
                i, new_val = best_move
                old_val = coeffs[i]
                coeffs[i] = new_val
                current_N += (new_val - old_val) * weights[i]
                current_abs_N = abs(current_N)
                
                if current_N != 0 and current_abs_N < best_abs_N:
                    best_abs_N = current_abs_N
                    best_coeffs = coeffs[:]
                    if best_abs_N == 1:
                        return True
                improved = True
                
        return False

    def simulated_annealing():
        nonlocal best_abs_N, best_coeffs
        
        # Initialize
        coeffs = [random.choice([-1, 0, 1]) for _ in range(30)]
        current_N = get_N(coeffs)
        current_E = abs(current_N) if current_N != 0 else float('inf')
        
        # SA Parameters
        T_start = 1000.0
        T_end = 1.0
        cooling_rate = 0.999
        
        # We run for a fixed number of steps or until time limit
        steps = 0
        max_steps = 50000
        
        while steps < max_steps and time.time() < deadline:
            # Pick a random index and a new value
            i = random.randint(0, 29)
            old_val = coeffs[i]
            new_val = random.choice([-1, 0, 1])
            
            if new_val == old_val:
                continue
                
            delta_N = (new_val - old_val) * weights[i]
            new_N = current_N + delta_N
            
            if new_N == 0:
                # Reject zero sum
                new_E = float('inf')
            else:
                new_E = abs(new_N)
                
            delta_E = new_E - current_E
            
            # Acceptance criterion
            if delta_E < 0:
                # Better
                coeffs[i] = new_val
                current_N = new_N
                current_E = new_E
            else:
                # Worse, accept with probability exp(-delta_E / T)
                if T_start > 0:
                    # Avoid overflow in exp
                    if delta_E / T_start < 700: 
                        prob = math.exp(-delta_E / T_start)
                    else:
                        prob = 0.0
                    
                    if random.random() < prob:
                        coeffs[i] = new_val
                        current_N = new_N
                        current_E = new_E
            
            # Update best
            if current_N != 0 and current_E < best_abs_N:
                best_abs_N = current_E
                best_coeffs = coeffs[:]
                if best_abs_N == 1:
                    return True
            
            # Cool down
            T_start *= cooling_rate
            if T_start < T_end:
                T_start = T_end
                
            steps += 1
            
        # Final polish
        if time.time() < deadline:
            polish(coeffs, current_N)
            
        return False

    # Run SA for a few restarts
    while time.time() < deadline:
        if simulated_annealing():
            break
            
    if best_coeffs is None:
        # Fallback: trivial non-zero sum
        # e.g. coeff[0]=1, rest 0 -> sum = L/1 = L
        best_coeffs = [1] + [0] * 29
        best_abs_N = L
        
    return best_coeffs
