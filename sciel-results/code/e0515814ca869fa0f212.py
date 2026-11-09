import time
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    MAX_V = 1000000
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Calculate LCM of divisors for periodicity
    lcm_val = 1
    for d in divisors:
        lcm_val = _lcm(lcm_val, d)
        
    # Search for minimum width W
    for W in range(1, 200):
        if time.time() > deadline:
            break
            
        # Iterate L from 1 to lcm_val
        for L in range(1, lcm_val + 1):
            if time.time() > deadline:
                break
                
            # Check if L+W <= MAX_V
            if L + W > MAX_V:
                continue
                
            # Build candidate lists for each divisor
            candidates = []
            valid = True
            for d in divisors:
                # Find multiples of d in [L, L+W]
                first_m = ((L + d - 1) // d) * d
                cands = []
                m = first_m
                while m <= L + W:
                    cands.append(m)
                    m += d
                if not cands:
                    valid = False
                    break
                candidates.append(cands)
            
            if not valid:
                continue
                
            # Solve CSP with backtracking and MRV
            assign = [0] * n
            used_vals = set()
            
            def rec(assigned_count):
                if time.time() > deadline:
                    return False
                if assigned_count == n:
                    return True
                    
                # MRV
                best_i = -1
                best_avail = []
                for i in range(n):
                    if assign[i] == 0:
                        avail = [c for c in candidates[i] if c not in used_vals]
                        if len(avail) == 0:
                            return False
                        if best_i == -1 or len(avail) < len(best_avail):
                            best_i = i
                            best_avail = avail
                            if len(avail) == 1:
                                break
                                  
                if best_i == -1:
                    return True
                    
                for val in best_avail:
                    assign[best_i] = val
                    used_vals.add(val)
                    if rec(assigned_count + 1):
                        return True
                    used_vals.remove(val)
                    assign[best_i] = 0
                    
                return False

            if rec(0):
                # Verify
                if len(set(assign)) == n and all(1 <= x <= MAX_V for x in assign) and all(assign[i] % divisors[i] == 0 for i in range(n)):
                    return assign

    # Fallback
    return divisors[:]
