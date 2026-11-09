import time
import random
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Precompute LCM of 1..30 to work with integers
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
    
    L = reduce(lcm, range(1, 31))
    
    # Coefficients for 1/1 to 1/30 are mapped to indices 0..29
    # sum = c[0]/1 + c[1]/2 + ... + c[29]/30
    # Integer sum S = c[0]*L + c[1]*(L/2) + ... + c[29]*(L/30)
    weights = [L // i for i in range(1, 31)]
    
    # Initialize with a random valid configuration
    # Must be nonzero sum. 
    # Random init: mostly zeros, some -1, 0, 1
    current_c = [random.choice([-1, 0, 1]) for _ in range(30)]
    
    # Ensure nonzero sum
    current_sum = sum(c * w for c, w in zip(current_c, weights))
    while current_sum == 0:
        # Flip a random element
        idx = random.randint(0, 29)
        old_val = current_c[idx]
        new_val = random.choice([-1, 0, 1])
        if new_val != old_val:
            current_c[idx] = new_val
            current_sum += (new_val - old_val) * weights[idx]
            
    best_c = current_c[:]
    best_sum = current_sum
    best_abs = abs(current_sum)
    
    # Simulated Annealing
    # Start temperature high enough to accept bad moves, decay exponentially
    T = 1000.0
    T_min = 1e-6
    cooling_rate = 0.999
    
    while time.time() < deadline:
        # Generate neighbor
        idx = random.randint(0, 29)
        old_val = current_c[idx]
        # Choose a different value from {-1, 0, 1}
        new_val = random.choice([-1, 0, 1])
        if new_val == old_val:
            continue
            
        delta = (new_val - old_val) * weights[idx]
        new_sum = current_sum + delta
        
        # If new sum is zero, it's invalid. We must reject it.
        if new_sum == 0:
            # Reject zero sum states
            # We can still update temperature or just skip
            # Skipping is fine, but we might get stuck if many neighbors are 0.
            # Given sparsity, this is rare.
            continue
            
        new_abs = abs(new_sum)
        current_abs = abs(current_sum)
        
        if new_abs < current_abs:
            # Accept improvement
            current_c[idx] = new_val
            current_sum = new_sum
            
            if new_abs < best_abs:
                best_abs = new_abs
                best_sum = new_sum
                best_c = current_c[:]
        else:
            # Accept worse move with probability exp(-delta_E / T)
            # delta_E is the increase in cost
            delta_E = new_abs - current_abs
            if delta_E == 0:
                # Same absolute value, different sign? Or same value.
                # If abs is same, it's a neutral move. Accept with high prob to explore.
                if random.random() < 0.5:
                    current_c[idx] = new_val
                    current_sum = new_sum
            else:
                prob = 2.718281828459045 ** (-delta_E / T)
                if random.random() < prob:
                    current_c[idx] = new_val
                    current_sum = new_sum
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    return best_c
