import time
from math import factorial

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    target = factorial(12)
    
    # Generate all divisors of target
    divisors = set()
    # 479001600 = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    for a in range(11):
        for b in range(6):
            for c in range(3):
                for d in range(2):
                    for e in range(2):
                        val = (2**a) * (3**b) * (5**c) * (7**d) * (11**e)
                        divisors.add(val)
    
    # Filter divisors > 12
    valid_divs = sorted([d for d in divisors if d > 12])
    
    best_max = [target]
    best_factors = [target]
    
    # DFS to find decomposition
    def dfs(rem, last, current_factors):
        if time.time() > deadline:
            return
        
        if rem == 1:
            if current_factors:
                max_f = current_factors[-1]
                if max_f < best_max[0]:
                    best_max[0] = max_f
                    best_factors[:] = current_factors[:]
            return
        
        if len(current_factors) >= 12:
            return
            
        for d in valid_divs:
            if d <= last:
                continue
            if d > rem:
                break
            if rem % d != 0:
                continue
                
            # Pruning: if d is already >= best_max, we can't improve
            if d >= best_max[0]:
                break
                
            current_factors.append(d)
            dfs(rem // d, d, current_factors)
            current_factors.pop()
            
    dfs(target, 12, [])
    
    return best_factors
