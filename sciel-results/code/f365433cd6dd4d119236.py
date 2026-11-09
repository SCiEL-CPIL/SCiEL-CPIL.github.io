import time
from fractions import Fraction

TARGET = Fraction(5, 31)
MAX_DENOM = 10_000_000
MAX_TERMS = 256
DEADLINE = 85.0


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    def check_time():
        return time.time() > deadline

    # Known solution: [8, 31, 248], max = 248
    best_solution = [8, 31, 248]
    best_max = 248

    # Search for solutions with maximum denominator M < 248
    # Use DFS with strong pruning

    def dfs(remaining_num, remaining_den, min_d, max_d, current, depth):
        """remaining = remaining_num / remaining_den in lowest terms"""
        if check_time():
            return None
        if remaining_num == 0:
            return list(current)
        if depth >= MAX_TERMS:
            return None
        if remaining_num < 0:
            return None
        
        # d must satisfy: 1/d <= remaining => d >= remaining_den / remaining_num
        min_d_eff = max(min_d, (remaining_den + remaining_num - 1) // remaining_num)
        if min_d_eff > max_d:
            return None
        
        # Pruning: sum of 1/d for d in [min_d_eff, max_d] >= remaining
        # Upper bound: (max_d - min_d_eff + 1) / min_d_eff >= remaining_num/remaining_den
        # => (max_d - min_d_eff + 1) * remaining_den >= remaining_num * min_d_eff
        if (max_d - min_d_eff + 1) * remaining_den < remaining_num * min_d_eff:
            return None
        
        for d in range(min_d_eff, max_d + 1):
            if check_time():
                return None
            
            # rem_new = remaining - 1/d = (remaining_num * d - remaining_den) / (remaining_den * d)
            new_num = remaining_num * d - remaining_den
            new_den = remaining_den * d
            if new_num < 0:
                break
            
            if new_num == 0:
                return list(current) + [d]
            
            # Reduce fraction
            from math import gcd
            g = gcd(new_num, new_den)
            new_num_r = new_num // g
            new_den_r = new_den // g
            
            res = dfs(new_num_r, new_den_r, d + 1, max_d, current + [d], depth + 1)
            if res is not None:
                return res
        
        return None

    from math import gcd
    tn, td = TARGET.numerator, TARGET.denominator  # 5, 31

    for M in range(7, best_max):
        if check_time():
            break
        
        result = dfs(tn, td, 7, M, [], 0)
        if result is not None:
            # Verify
            total = sum(Fraction(1, d) for d in result)
            if total == TARGET and len(set(result)) == len(result):
                best_solution = result
                best_max = M
                break
    
    return best_solution
