import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    target_frac = Fraction(5, 31)
    
    # MITM Search
    # We will try M from 1 to 60.
    # For each M, compute L = lcm(1..M).
    # Scale = L * 31.
    # Target_Scaled = Scale * 5 / 31 = L * 5.
    # Value_i = Scale / i = 31 * L / i.
    
    # Precompute 1/k as floats for quick pruning?
    # Or just use the integer values.
    
    def get_lcm(n):
        return math.lcm(*range(1, n+1)) if n > 1 else 1

    best_solution = None
    
    for M in range(1, 61):
        if time.time() > deadline:
            break
            
        L = get_lcm(M)
        # L can be very large. 
        # For M=40, L has ~20 digits.
        # Scale = 31 * L.
        # Target = 5 * L.
        
        target_val = 5 * L
        # Values: v[i] = 31 * L // i
        
        # Check if 31*L is divisible by i?
        # L is multiple of i. So yes.
        
        vals = [31 * L // i for i in range(1, M + 1)]
        
        mid = M // 2
        L_indices = range(1, mid + 1)
        R_indices = range(mid + 1, M + 1)
        
        # Generate subset sums for L
        sums_L = {}
        # Pruning: sum <= target_val
        
        def gen_L(start_idx, current_sum, current_indices):
            if current_sum > target_val:
                return
            if current_indices:
                if current_sum not in sums_L:
                    sums_L[current_sum] = list(current_indices)
            else:
                sums_L[0] = []
                
            for i in range(start_idx, len(L_indices)):
                idx = L_indices[i]
                v = vals[idx - 1]
                if current_sum + v > target_val:
                    continue
                gen_L(i + 1, current_sum + v, current_indices + [idx])
                
        # Optimization: L_indices is small.
        gen_L(0, 0, [])
        
        # Check R
        found = False
        result_holder = [None]
        
        def check_R(start_idx, current_sum, current_indices):
            nonlocal found
            if found:
                return
            if current_sum > target_val:
                return
            
            need = target_val - current_sum
            if need in sums_L:
                sol = sums_L[need] + current_indices
                # Verify
                # s = sum(1/x for x in sol)
                # if s == target_frac:
                #     result_holder[0] = sol
                #     found = True
                #     return
                # Integer check: sum(vals[x-1] for x in sol) == target_val
                # This is guaranteed by construction if no overflow.
                result_holder[0] = sol
                found = True
                return
                
            for i in range(start_idx, len(R_indices)):
                idx = R_indices[i]
                v = vals[idx - 1]
                if current_sum + v > target_val:
                    continue
                check_R(i + 1, current_sum + v, current_indices + [idx])
                
        check_R(0, 0, [])
        
        if result_holder[0]:
            return result_holder[0]
            
    # Fallback DFS
    # ... (Use the DFS from the current program) ...
    
    def greedy_solution():
        r = target_frac
        values = []
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
            if time.time() > deadline:
                break
        return values
    return greedy_solution()
