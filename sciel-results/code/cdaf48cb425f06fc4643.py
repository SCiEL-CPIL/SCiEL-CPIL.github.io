import numpy as np

def construct():
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses
    # inv[x] is the modular inverse of x mod 1009
    inv = [0] * (MOD)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
        
    best_len = 0
    best_start = 2
    
    # The interval must be in [1, 1008].
    # 1 and 1008 are self-inverses, so they cannot be in the interval.
    # Thus L >= 2 and R <= 1007.
    
    for L in range(2, 1008):
        # R_max is the maximum valid right boundary for this L
        # Initially, we can go up to 1007
        R_max = 1007
        
        # We scan x from L upwards
        # If x > R_max, we stop because x is not in the interval [L, R_max]
        # Note: R_max can decrease as we find constraints
        for x in range(L, 1008):
            if x > R_max:
                break
                
            p = inv[x]
            
            if p == x:
                # Self-inverse, cannot be in interval
                R_max = x - 1
            elif p < x:
                # Partner is smaller than x
                # If partner is in [L, x-1], conflict
                if p >= L:
                    # Conflict found, interval must end before x
                    R_max = x - 1
            else:
                # p > x
                # Partner is larger than x
                # If interval extends to p, conflict
                # So R must be < p
                if R_max >= p:
                    R_max = p - 1
                    
        if R_max >= L:
            length = R_max - L + 1
            if length > best_len:
                best_len = length
                best_start = L
                
    return [best_start, best_len]
