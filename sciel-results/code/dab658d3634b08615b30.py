import time
from math import gcd
from functools import reduce

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # Calculate LCM of divisors to define the period
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    period = reduce(lcm, divisors)
    
    start_time = time.time()
    deadline = start_time + 80  # Safety margin for 90s limit
    
    # We want to minimize W = max - min
    # The minimum possible W is n - 1 = 9 (if we could pack them tightly)
    # But due to divisibility constraints, it might be larger.
    # We iterate W from 9 upwards.
    
    # For each W, we check all possible starting positions L in [1, period]
    # because the pattern of multiples repeats every 'period' integers.
    
    for W in range(9, 200):
        if time.time() > deadline:
            break
            
        # To speed up, we can check L in a specific order or just sequentially.
        # Sequential is fine.
        for L in range(1, period + 1):
            if time.time() > deadline:
                break
            
            # Determine available multiples for each divisor in [L, L+W]
            available = []
            feasible = True
            for d in divisors:
                # First multiple of d >= L
                # k = ceil(L/d)
                k = (L + d - 1) // d
                first_m = k * d
                if first_m > L + W:
                    feasible = False
                    break
                
                # Collect all multiples in [L, L+W]
                mults = []
                m = first_m
                while m <= L + W:
                    mults.append(m)
                    m += d
                available.append(mults)
            
            if not feasible:
                continue
            
            # Solve the assignment problem: assign each divisor i to a distinct value from available[i]
            # Use backtracking with MRV (Minimum Remaining Values)
            
            # Pre-sort indices by the number of available options to improve pruning
            # This order is static for this window
            order = sorted(range(n), key=lambda i: len(available[i]))
            
            used_values = set()
            result = [0] * n
            
            def backtrack(idx):
                if idx == n:
                    return True
                
                i = order[idx]
                for val in available[i]:
                    if val not in used_values:
                        used_values.add(val)
                        result[i] = val
                        if backtrack(idx + 1):
                            return True
                        used_values.remove(val)
                        result[i] = 0
                return False
            
            if backtrack(0):
                # Verify result
                # 1. Distinct
                if len(set(result)) != n:
                    continue
                # 2. Positive and <= 1000000
                if any(x < 1 or x > 1000000 for x in result):
                    continue
                # 3. Divisibility
                valid = True
                for i in range(n):
                    if result[i] % divisors[i] != 0:
                        valid = False
                        break
                if valid:
                    # Convert to standard Python ints just in case, though they are already ints
                    return [int(x) for x in result]
            
    # Fallback if no solution found within time or W limit
    # This should not happen for small W, but ensures we return a valid list
    # The seed program returned divisors, which is valid (range = 39-6=33, score 1.0)
    # But we should try to return something better if we found a partial result? 
    # No, the task requires a complete valid list.
    # If we timed out, we must return a valid list.
    # The divisors themselves are a valid list.
    return divisors[:]
