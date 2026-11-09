import time
from math import factorial
from itertools import combinations

_START = time.time()
_DEADLINE = _START + 80

def construct():
    target = factorial(12)  # 479001600
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    
    # Generate all divisors of target > 12
    divisors = []
    for a in range(11):
        for b in range(6):
            for c in range(3):
                for d in range(2):
                    for e in range(2):
                        val = (2**a) * (3**b) * (5**c) * (7**d) * (11**e)
                        if val > 12:
                            divisors.append(val)
    
    divisors = sorted(set(divisors))
    div_set = set(divisors)
    n_divs = len(divisors)
    
    # Map each divisor to its index
    div_index = {v: i for i, v in enumerate(divisors)}
    
    # For each divisor, compute its prime exponent vector (a, b, c, d, e)
    # where val = 2^a * 3^b * 5^c * 7^d * 11^e
    div_exponents = {}
    for v in divisors:
        a = 0
        while v % 2 == 0:
            a += 1
            v //= 2
        b = 0
        while v % 3 == 0:
            b += 1
            v //= 3
        c = 0
        while v % 5 == 0:
            c += 1
            v //= 5
        d = 0
        while v % 7 == 0:
            d += 1
            v //= 7
        e = 0
        while v % 11 == 0:
            e += 1
            v //= 11
        div_exponents[div_index[v]] = (a, b, c, d, e)
    
    best_max = [target]
    best_factors = [[target]]
    
    def time_up():
        return time.time() > _DEADLINE
    
    # Try to find a good initial solution using a greedy approach
    # Pick the smallest divisors that divide the remaining product
    greedy_factors = []
    remaining = target
    used = set()
    for d in divisors:
        if d in used:
            continue
        if remaining % d == 0:
            greedy_factors.append(d)
            remaining //= d
            used.add(d)
            if remaining == 1:
                break
    
    if remaining == 1 and len(greedy_factors) >= 1:
        greedy_max = max(greedy_factors)
        if greedy_max < best_max[0]:
            best_max[0] = greedy_max
            best_factors[0] = list(greedy_factors)
    
    # DFS with bitmask-style pruning using exponent vectors
    # State: (remaining_exponents, min_allowed_index, current_max, count, current_factors)
    # remaining_exponents is a tuple (a, b, c, d, e) representing remaining prime exponents
    
    target_exp = (10, 5, 2, 1, 1)
    
    def dfs(rem_exp, min_idx, current_max, count, current_factors):
        if time_up():
            return
        
        # Check if rem_exp is (0,0,0,0,0)
        if rem_exp == (0, 0, 0, 0, 0):
            if count >= 1:
                if current_max < best_max[0]:
                    best_max[0] = current_max
                    best_factors[0] = list(current_factors)
            return
        
        if count >= 12:
            return
        
        if current_max >= best_max[0]:
            return
        
        # Try each divisor index >= min_idx
        for i in range(min_idx, n_divs):
            if time_up():
                return
            
            d = divisors[i]
            
            # Pruning: if d > remaining product, we can't use it
            # Actually, we need to check if d divides the remaining product
            # The remaining product's prime exponents must be >= d's exponents
            
            exp = div_exponents[i]
            if (rem_exp[0] < exp[0] or rem_exp[1] < exp[1] or 
                rem_exp[2] < exp[2] or rem_exp[3] < exp[3] or 
                rem_exp[4] < exp[4]):
                # This divisor's exponents exceed what's available
                # Since divisors are sorted by value, larger divisors may have smaller exponents
                # So we can't break, but we can skip this one
                continue
            
            new_max = max(current_max, d)
            if new_max >= best_max[0]:
                continue
            
            new_rem_exp = (rem_exp[0] - exp[0], rem_exp[1] - exp[1],
                          rem_exp[2] - exp[2], rem_exp[3] - exp[3],
                          rem_exp[4] - exp[4])
            
            current_factors.append(d)
            dfs(new_rem_exp, i + 1, new_max, count + 1, current_factors)
            current_factors.pop()
    
    dfs(target_exp, 0, 0, 0, [])
    
    return best_factors[0]
