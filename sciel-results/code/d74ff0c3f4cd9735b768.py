import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s margin
    
    target_num = 5
    target_den = 31
    
    # Initial known solution
    best_solution = [8, 31, 248]
    best_max_den = 248
    
    def verify_solution(denoms):
        """Verify that the denominators sum to 5/31"""
        total = sum(Fraction(1, d) for d in denoms)
        return total == Fraction(5, 31)
    
    def search_3terms():
        """Specialized fast search for 3-term solutions"""
        nonlocal best_solution, best_max_den
        
        if time.time() > deadline:
            return
            
        # 1/a + 1/b + 1/c = 5/31 with 7 <= a < b < c < best_max_den
        for a in range(7, best_max_den):
            if time.time() > deadline:
                return
            if a >= best_max_den:
                break
                
            # Remainder: 5/31 - 1/a = (5a - 31)/(31a)
            rem_num = 5 * a - 31
            rem_den = 31 * a
            
            if rem_num <= 0:
                continue
                
            g = gcd(rem_num, rem_den)
            rem_num //= g
            rem_den //= g
            
            # Find 1/b + 1/c = rem_num/rem_den with a < b < c < best_max_den
            # c = b * rem_den / (rem_num * b - rem_den)
            # Need rem_num * b - rem_den > 0, so b > rem_den / rem_num
            # Also need (rem_num * b - rem_den) | (b * rem_den)
            
            b_min = max(a + 1, rem_den // rem_num + 1)
            b_max = best_max_den - 2  # Need c > b and c < best_max_den
            
            # Upper bound on b: 1/b + 1/(b+1) >= rem_num/rem_den (minimum c is b+1)
            # Actually we need 1/b + 1/c = rem_num/rem_den with c > b
            # So 1/b + 1/c < 1/b + 1/(b+1)
            # Thus rem_num/rem_den < 1/b + 1/(b+1) = (2b+1)/(b(b+1))
            # This gives a lower bound on b, not upper.
            # Upper bound: 2/b > rem_num/rem_den (since c > b, 1/c < 1/b)
            # So b < 2*rem_den/rem_num
            b_max = min(b_max, (2 * rem_den) // rem_num)
            
            if b_min > b_max:
                continue
                
            for b in range(b_min, b_max + 1):
                if time.time() > deadline:
                    return
                if b >= best_max_den:
                    break
                    
                denom_c = rem_num * b - rem_den
                if denom_c <= 0:
                    continue
                    
                num_c = b * rem_den
                if num_c % denom_c != 0:
                    continue
                    
                c = num_c // denom_c
                
                if c <= b or c >= best_max_den:
                    continue
                    
                # Found valid 3-term solution
                if c < best_max_den:
                    best_max_den = c
                    best_solution = [a, b, c]
                    # Tighten bounds
                    b_max = min(b_max, best_max_den - 2)
                    if b_min > b_max:
                        break
        
        return
    
    def search(rem_num, rem_den, min_d, max_d_limit, k, path):
        nonlocal best_solution, best_max_den
        
        if time.time() > deadline:
            return
        
        # If we need 1 more term
        if k == 1:
            if rem_num == 0:
                return
            if rem_den % rem_num != 0:
                return
            d = rem_den // rem_num
            if d > min_d and d < best_max_den:
                if d < best_max_den:
                    best_max_den = d
                    best_solution = path + [d]
            return
        
        # Lower bound for d
        d_min = max(min_d + 1, (rem_den + rem_num - 1) // rem_num)
        
        # Upper bound for d
        d_max = (k * rem_den) // rem_num
        
        if d_min >= best_max_den:
            return
            
        d_max = min(d_max, best_max_den - 1)
        
        if d_min > d_max:
            return
            
        for d in range(d_min, d_max + 1):
            if time.time() > deadline:
                return
            
            new_num = rem_num * d - rem_den
            new_den = rem_den * d
            
            if new_num <= 0:
                if new_num == 0:
                    return
                else:
                    continue
            
            g = gcd(new_num, new_den)
            new_num //= g
            new_den //= g
            
            # Pruning: max possible sum of remaining k-1 terms
            max_possible = (k - 1) / (d + 1)
            if max_possible < new_num / new_den:
                continue
            
            # Pruning: min possible sum of remaining k-1 terms
            # 1/(d+1) + 1/(d+2) + ... + 1/(d+k-1)
            min_possible = sum(1.0 / (d + i) for i in range(1, k))
            if min_possible > new_num / new_den:
                continue
            
            search(new_num, new_den, d, best_max_den, k - 1, path + [d])
            
            if d >= best_max_den:
                break

    # Run specialized 3-term search first
    search_3terms()
    
    # Then general search for 3-7 terms if time permits
    if time.time() < deadline:
        search(5, 31, 6, best_max_den, 3, [])
    
    if time.time() < deadline:
        search(5, 31, 6, best_max_den, 4, [])
        
    if time.time() < deadline:
        search(5, 31, 6, best_max_den, 5, [])
        
    if time.time() < deadline:
        search(5, 31, 6, best_max_den, 6, [])
        
    if time.time() < deadline:
        search(5, 31, 6, best_max_den, 7, [])

    # Verify the solution
    if verify_solution(best_solution):
        return best_solution
    else:
        return [8, 31, 248]
