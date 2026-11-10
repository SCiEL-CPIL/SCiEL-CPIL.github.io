import time
from fractions import Fraction

def construct():
    start = time.time()
    deadline = start + 85  # Safety margin
    
    target_num = 5
    target_den = 31
    
    best_max = float('inf')
    best_result = None
    
    def update_best(denoms):
        nonlocal best_max, best_result
        mx = max(denoms)
        if mx < best_max:
            best_max = mx
            best_result = list(denoms)

    def get_divisors(n):
        """Return list of divisors of n."""
        divs = []
        i = 1
        while i * i <= n:
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
            i += 1
        return divs

    def solve_two_terms(rem_num, rem_den, min_x):
        """
        Solve 1/x + 1/y = rem_num/rem_den with x < y and x >= min_x.
        Returns list of (x, y) pairs.
        """
        if rem_num <= 0 or rem_den <= 0:
            return []
        
        # (rem_num * x - rem_den) * (rem_num * y - rem_den) = rem_den^2
        const = rem_den * rem_den
        
        solutions = []
        
        # Iterate through divisors of const
        # Optimization: We can bound the search.
        # x > rem_den / rem_num
        # min_x is given.
        
        # We iterate A such that A is a divisor of const.
        # x = (A + rem_den) / rem_num
        # y = (const//A + rem_den) / rem_num
        
        # To ensure x < y, we need A < const//A, i.e., A^2 < const.
        # So A < sqrt(const).
        
        import math
        sqrt_const = int(math.isqrt(const))
        
        for A in range(1, sqrt_const + 1):
            if const % A != 0:
                continue
            
            B = const // A
            
            # Check x
            num_x = A + rem_den
            if num_x % rem_num != 0:
                continue
            x = num_x // rem_num
            
            if x < min_x:
                continue
                
            # Check y
            num_y = B + rem_den
            if num_y % rem_num != 0:
                continue
            y = num_y // rem_num
            
            if y <= x:
                continue
                
            # Optional: Check if y is within reasonable bounds (e.g., 10^7)
            if y <= 10000000:
                solutions.append((x, y))
                
        return solutions

    # --- Search 3 terms: 1/a + 1/b + 1/c = 5/31 ---
    # a >= 7
    # 3/a >= 5/31 => a <= 93/5 = 18.6 => a <= 18
    for a in range(7, 19):
        if time.time() > deadline:
            break
        
        rem_num = target_num * a - target_den
        rem_den = target_den * a
        
        if rem_num <= 0:
            continue
            
        sols = solve_two_terms(rem_num, rem_den, a + 1)
        for b, c in sols:
            update_best([a, b, c])

    # --- Search 4 terms: 1/a + 1/b + 1/c + 1/d = 5/31 ---
    # a >= 7
    # 4/a >= 5/31 => a <= 124/5 = 24.8 => a <= 24
    # However, we can prune a if a >= best_max
    for a in range(7, 25):
        if time.time() > deadline:
            break
        if a >= best_max:
            break
            
        rem1_num = target_num * a - target_den
        rem1_den = target_den * a
        
        if rem1_num <= 0:
            continue
            
        # Now solve 1/b + 1/c + 1/d = rem1
        # b > a
        # 3/b >= rem1 => b <= 3/rem1 = 3*rem1_den/rem1_num
        # Also 1/b < rem1 => b > 1/rem1 = rem1_den/rem1_num
        
        min_b = max(a + 1, int(rem1_den / rem1_num) + 1)
        max_b = int(3 * rem1_den / rem1_num)
        
        # Prune max_b if it exceeds best_max - 2 (since c > b, d > c)
        if max_b >= best_max - 2:
            max_b = best_max - 3
            
        for b in range(min_b, max_b + 1):
            if time.time() > deadline:
                break
                
            rem2_num = rem1_num * b - rem1_den
            rem2_den = rem1_den * b
            
            if rem2_num <= 0:
                continue
                
            # Solve 1/c + 1/d = rem2 with c > b
            sols = solve_two_terms(rem2_num, rem2_den, b + 1)
            for c, d in sols:
                if d < best_max:
                    update_best([a, b, c, d])

    # --- Search 5 terms: 1/a + 1/b + 1/c + 1/d + 1/e = 5/31 ---
    # This might be overkill if 3 or 4 term solutions are good enough.
    # But let's add it if time permits, with tight pruning.
    # a <= 5/ (5/31) = 31.
    
    if time.time() < deadline - 5: # Reserve 5s
        for a in range(7, 32):
            if time.time() > deadline:
                break
            if a >= best_max:
                break
                
            rem1_num = target_num * a - target_den
            rem1_den = target_den * a
            if rem1_num <= 0: continue
            
            min_b = max(a + 1, int(rem1_den / rem1_num) + 1)
            max_b = min(int(4 * rem1_den / rem1_num), best_max - 3)
            
            for b in range(min_b, max_b + 1):
                if time.time() > deadline:
                    break
                    
                rem2_num = rem1_num * b - rem1_den
                rem2_den = rem1_den * b
                if rem2_num <= 0: continue
                
                min_c = max(b + 1, int(rem2_den / rem2_num) + 1)
                max_c = min(int(3 * rem2_den / rem2_num), best_max - 2)
                
                for c in range(min_c, max_c + 1):
                    if time.time() > deadline:
                        break
                        
                    rem3_num = rem2_num * c - rem2_den
                    rem3_den = rem2_den * c
                    if rem3_num <= 0: continue
                    
                    # Solve 1/d + 1/e = rem3 with d > c
                    sols = solve_two_terms(rem3_num, rem3_den, c + 1)
                    for d, e in sols:
                        if e < best_max:
                            update_best([a, b, c, d, e])

    if best_result is not None:
        return best_result

    # Fallback
    return [7, 8, 9, 10000000]
