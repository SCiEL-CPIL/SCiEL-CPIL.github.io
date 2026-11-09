import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (evaluator kills at 90s)
    
    # First, try known small solutions
    # Check if there's a solution with 8 terms
    known_solutions = [
        # Try to find 8-term solutions
    ]
    
    # Try DFS for k=8,9,10 with optimized pruning
    for k in range(8, 15):
        if time.time() > deadline:
            break
        solution = find_solution_fast(k, start_time, deadline)
        if solution:
            return solution
    
    # Fallback: greedy algorithm
    return greedy_solution(start_time, deadline)

def find_solution_fast(k, start_time, deadline):
    """DFS with integer arithmetic for speed."""
    
    def dfs(idx, last_d, rem_num, rem_den, current_list):
        if time.time() > deadline:
            return None
            
        if idx == k:
            if rem_num == 0:
                return current_list
            return None
        
        terms_left = k - idx
        
        # If 1 term left, check if rem is a unit fraction
        if terms_left == 1:
            # rem_num/rem_den = 1/d => d = rem_den/rem_num, must be integer
            if rem_num == 1:
                d = rem_den
                if d > last_d and d <= 10000000:
                    return current_list + [d]
            # Or rem_num divides rem_den
            if rem_den % rem_num == 0:
                d = rem_den // rem_num
                if d > last_d and d >= 5 and d <= 10000000:
                    return current_list + [d]
            return None
        
        # Bounds for next denominator d
        # We need: 1/d < rem (i.e., d > rem_den/rem_num)
        # And: sum of terms_left terms starting from d must be >= rem
        #   => terms_left/d >= rem approximately => d <= terms_left * rem_den / rem_num
        
        # Lower bound: d > last_d and d > rem_den/rem_num
        # 1/d < rem_num/rem_den => d > rem_den/rem_num
        min_d = max(last_d + 1, 5)
        
        # d must satisfy: rem_den/rem_num < d, so d >= floor(rem_den/rem_num) + 1
        d_lower = rem_den // rem_num + 1
        if rem_den % rem_num == 0:
            # d must be strictly greater than rem_den/rem_num
            d_lower = rem_den // rem_num + 1
        min_d = max(min_d, d_lower)
        
        # Upper bound: we need sum_{i=0}^{terms_left-1} 1/(d+i) >= rem
        # Approximation: terms_left/d >= rem_num/rem_den => d <= terms_left * rem_den / rem_num
        d_upper = (terms_left * rem_den) // rem_num + 20
        if d_upper > 10000000:
            d_upper = 10000000
            
        if min_d > d_upper:
            return None
            
        # Also, the minimum possible sum with terms_left terms starting from min_d
        # must be >= rem. The max sum is 1/min_d + 1/(min_d+1) + ...
        # For pruning: if even the max possible sum < rem, no solution
        # Max sum ≈ terms_left / min_d (rough upper bound)
        # More precise: sum_{i=0}^{terms_left-1} 1/(min_d+i)
        # If this < rem, then no d >= min_d will work... wait, smaller d gives larger sum
        # So if sum from min_d is already < rem, then no solution
        # But we're iterating d from min_d upward, and as d increases, sum decreases
        # So if at d=min_d the sum is < rem, then for all larger d it's even smaller
        # Let's check: is 1/min_d + 1/(min_d+1) + ... + 1/(min_d+terms_left-1) >= rem?
        # This is a quick check
        
        # For efficiency, use a rough bound: terms_left/min_d >= rem
        # If not, prune
        if terms_left * rem_den < rem_num * min_d:
            # terms_left/min_d < rem_num/rem_den => sum too small
            return None
        
        for d in range(min_d, d_upper + 1):
            # new_rem = rem - 1/d = (rem_num*d - rem_den) / (rem_den*d)
            new_num = rem_num * d - rem_den
            if new_num < 0:
                break  # 1/d > rem, too large a fraction
            new_den = rem_den * d
            
            # Reduce fraction
            g = _gcd(new_num, new_den)
            if g > 1:
                new_num //= g
                new_den //= g
            
            res = dfs(idx + 1, d, new_num, new_den, current_list + [d])
            if res:
                return res
                
        return None
    
    return dfs(0, 4, 1, 1, [])

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def greedy_solution(start_time, deadline):
    """Greedy algorithm as fallback."""
    rem = Fraction(1, 1)
    last_d = 4
    result = []
    
    while rem > 0 and time.time() < deadline:
        # Next denominator: largest unit fraction <= rem
        # 1/d <= rem => d >= 1/rem
        d = rem.denominator // rem.numerator
        if rem.denominator % rem.numerator != 0:
            d += 1
        if d <= last_d:
            d = last_d + 1
        if d < 5:
            d = 5
        if d > 10000000:
            break
            
        result.append(d)
        rem -= Fraction(1, d)
        last_d = d
        
        if len(result) > 256:
            break
            
    if rem == 0:
        return result
    
    # If greedy didn't finish, try to complete with a known identity
    # This shouldn't happen for 1, but just in case
    # Use the fact that we can always add more terms
    # For safety, return what we have if it sums to 1, otherwise extend
    s = sum(Fraction(1, d) for d in result)
    if s == 1:
        return result
    
    # Extend using a simple construction
    # 1 = sum of what we have + remainder
    # Add terms to make up the remainder
    while rem > 0 and len(result) < 256 and time.time() < deadline:
        d = rem.denominator // rem.numerator
        if rem.denominator % rem.numerator != 0:
            d += 1
        if d <= last_d:
            d = last_d + 1
        if d < 5:
            d = 5
        if d > 10000000:
            # Use a different approach
            # Split rem into two parts
            break
        result.append(d)
        rem -= Fraction(1, d)
        last_d = d
        
    return result
