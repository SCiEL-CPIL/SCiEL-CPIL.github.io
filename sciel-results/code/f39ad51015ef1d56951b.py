import time
import math
import random

def construct():
    n = 256
    t0 = time.time()
    deadline = t0 + 50.0
    
    # Thue-Morse initialization
    def popcount(x):
        return bin(x).count('1')
    
    x = [(-1) ** popcount(i) for i in range(n)]
    
    def full_cost(x):
        max_val = 0
        for d in range(1, n + 1):
            for r in range(d):
                s = 0
                for i in range(r, n, d):
                    s += x[i]
                    if s > max_val:
                        max_val = s
                    elif -s > max_val:
                        max_val = -s
        return max_val

    def partial_cost(x, d_max):
        max_val = 0
        for d in range(1, d_max + 1):
            for r in range(d):
                s = 0
                for i in range(r, n, d):
                    s += x[i]
                    if s > max_val:
                        max_val = s
                    elif -s > max_val:
                        max_val = -s
        return max_val

    current_cost = full_cost(x)
    best_x = x.copy()
    best_cost = current_cost
    
    # Simulated Annealing
    T = 5.0
    cooling = 0.995
    rng = random.Random(123)
    
    while time.time() < deadline:
        i = rng.randrange(n)
        x[i] *= -1
        
        # Determine d_max based on current best cost to limit search space
        # We use best_cost to be conservative? Or current_cost?
        # If we use current_cost, and current_cost is high, d_max is small.
        # But if the flip increased cost, we might miss it?
        # No, if we use d_max derived from current_cost, and the new cost is higher,
        # it must be in the checked range?
        # If the new cost comes from an unchecked progression, its length is < current_cost.
        # But if current_cost was 10, and we flip, and an unchecked progression (len 9) becomes 9,
        # and checked ones are 5, new cost is 9.
        # But 9 < 10. So it's an improvement.
        # What if an unchecked progression becomes 10? Impossible, len < 10.
        # So using current_cost is safe for detecting improvements.
        # But if we want to accept a worse move (SA), we need accurate cost.
        # If we accept a worse move, current_cost increases.
        # Then d_max decreases.
        # This is fine.
        
        d_max = n // current_cost
        if d_max < 1:
            d_max = 1
            
        new_partial = partial_cost(x, d_max)
        L_bound = (n - 1) // (d_max + 1) + 1
        new_cost = max(new_partial, L_bound)
        
        delta = new_cost - current_cost
        
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_x = x.copy()
        else:
            x[i] *= -1
            
        T *= cooling
        if T < 0.1:
            T = 0.1

    return best_x
