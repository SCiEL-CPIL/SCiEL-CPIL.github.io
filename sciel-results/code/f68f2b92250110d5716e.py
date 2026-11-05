import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin
    
    n = 30
    
    # Calculate L = lcm(1..30)
    def lcm(a, b):
        return a * b // gcd(a, b)
        
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
        
    # A_i = L / i
    A = [L // i for i in range(1, n + 1)]
    
    # Initial state: 
    # Try to start with a small sum.
    # c[28]=1, c[29]=-1 => sum = L/29 - L/30 = L/870
    best_coeffs = [0] * n
    best_coeffs[28] = 1
    best_coeffs[29] = -1
    
    current_coeffs = best_coeffs[:]
    
    # Calculate current integer sum
    current_sum = 0
    for i in range(n):
        if current_coeffs[i] != 0:
            current_sum += current_coeffs[i] * A[i]
            
    # Ensure initial sum is non-zero
    if current_sum == 0:
        # Force a move
        current_coeffs[0] = 1
        current_sum += A[0]
        
    best_sum = current_sum
    best_abs_val = abs(best_sum)
    
    # Simulated Annealing Parameters
    # Max possible change in sum is 2 * max(A) = 2 * L
    T = 2 * L
    T_min = 1
    # Number of iterations per temperature step or just continuous cooling
    # Let's use a cooling factor
    cooling_factor = 0.99999
    
    # We will iterate until time limit
    while time.time() - start_time < deadline:
        # Pick random index
        i = random.randint(0, n - 1)
        old_val = current_coeffs[i]
        
        # Pick new value
        # Possible values: -1, 0, 1 excluding old_val
        if old_val == 0:
            new_val = random.choice([-1, 1])
        else:
            # old_val is 1 or -1
            # Options: 0, -old_val
            new_val = random.choice([0, -old_val])
            
        delta = (new_val - old_val) * A[i]
        new_sum = current_sum + delta
        
        # Constraint: new_sum must be non-zero
        if new_sum == 0:
            # Reject move to zero unless current is zero (which it shouldn't be)
            # Actually, if we are at 0, we MUST move. But we track best non-zero.
            # If current_sum != 0, we reject moves to 0 to stay in valid space.
            # If current_sum == 0, we accept any move.
            if current_sum != 0:
                continue
            else:
                # Accept escape from 0
                current_coeffs[i] = new_val
                current_sum = new_sum
                if abs(new_sum) < best_abs_val:
                    best_abs_val = abs(new_sum)
                    best_sum = new_sum
                    best_coeffs = current_coeffs[:]
                continue
        
        # Both current and new are non-zero
        new_abs = abs(new_sum)
        
        if new_abs < best_abs_val:
            best_abs_val = new_abs
            best_sum = new_sum
            best_coeffs = current_coeffs[:]
            
        # Acceptance criterion
        if new_abs < abs(current_sum):
            # Better, accept
            current_coeffs[i] = new_val
            current_sum = new_sum
        else:
            # Worse, accept with probability exp(-delta_E / T)
            delta_E = new_abs - abs(current_sum)
            # If T is very small, this will be close to 0
            if delta_E == 0:
                accept = True
            else:
                # Use random.random() < exp(-delta_E / T)
                # To avoid overflow/underflow in exp, we can compare logs or just use exp
                # delta_E is integer. T is float.
                # If delta_E / T > 700, exp is 0.
                ratio = delta_E / T
                if ratio > 700:
                    accept = False
                else:
                    accept = random.random() < pow(2.718281828, -ratio)
                    
            if accept:
                current_coeffs[i] = new_val
                current_sum = new_sum
        
        # Cool down
        T *= cooling_factor
        if T < T_min:
            T = T_min
            
    # Final check
    # best_coeffs should correspond to best_sum
    # Verify best_sum is non-zero
    if best_sum == 0:
        # Fallback
        return [0]*29 + [1, -1]
        
    return best_coeffs
