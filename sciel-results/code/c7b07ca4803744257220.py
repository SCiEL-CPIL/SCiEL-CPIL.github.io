import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    # Known solution
    best_solution = [8, 31, 248]
    best_M = 248
    
    # Search for 3-term solutions with M < 248
    # 1/a + 1/b + 1/M = 5/31
    # 1/a + 1/b = (5M - 31) / (31M)
    # Let R = 5M - 31.
    # (Ra - 31M)(Rb - 31M) = (31M)^2
    
    for M in range(1, 248):
        if time.time() > deadline:
            break
            
        R = 5 * M - 31
        if R <= 0:
            continue
            
        target_num = 31 * M
        target_sq = target_num * target_num
        
        # Iterate divisors of target_sq
        # We need X = Ra - 31M.
        # X must be a divisor of target_sq.
        # Also X + 31M must be divisible by R.
        
        # To speed up, we can iterate X from 1 to sqrt(target_sq)
        # But target_sq can be up to (31*247)^2 ~ 5.8e6. sqrt ~ 2400.
        # This is very fast.
        
        limit = int(math.isqrt(target_sq))
        
        for X in range(1, limit + 1):
            if target_sq % X != 0:
                continue
                
            Y = target_sq // X
            
            # Check divisibility for a
            if (X + target_num) % R != 0:
                continue
            a = (X + target_num) // R
            
            # Check divisibility for b
            if (Y + target_num) % R != 0:
                continue
            b = (Y + target_num) // R
            
            # Check bounds and ordering
            if a < 1 or b < 1:
                continue
            if a > M or b > M:
                continue
            if a > b:
                continue
                
            # Check distinctness
            if a == b or a == M or b == M:
                continue
                
            # Found valid solution
            return [a, b, M]
            
    return best_solution
