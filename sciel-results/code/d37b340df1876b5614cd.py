# EVOLVE-BLOCK-START
import time
import math
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    # Try to find a solution with 8 terms
    result = _search_k8(deadline)
    if result:
        return result
    
    # Fallback to larger k if 8 fails
    for k in range(9, 13):
        if time.time() > deadline - 5:
            break
        result = _search_k(k, deadline)
        if result:
            return result
    
    raise RuntimeError("no valid representation found")

def _max_sum_float(m, start):
    """Approximate max sum of m unit fractions with denominators >= start"""
    s = 0.0
    for i in range(m):
        s += 1.0 / (start + i)
    return s

def _search_k8(deadline):
    """Search for 8 distinct denominators >= 5 summing to 1 using integer arithmetic"""
    k = 8
    d = []
    
    def dfs(idx, cur_num, cur_den, last_denom):
        if time.time() > deadline:
            return None
        
        if idx == k:
            if cur_num == cur_den:
                return d[:]
            return None
        
        remaining_terms = k - idx
        
        # remaining fraction = (cur_den - cur_num) / cur_den
        rem_num = cur_den - cur_num
        rem_den = cur_den
        
        if rem_num <= 0:
            return None
        
        # Prune: max possible sum of remaining terms
        max_possible_float = _max_sum_float(remaining_terms, last_denom + 1)
        if rem_num / rem_den > max_possible_float + 1e-12:
            return None
        
        # Special case: last term
        if remaining_terms == 1:
            # Need 1/d = rem_num/rem_den => d = rem_den/rem_num
            if rem_den % rem_num == 0:
                d_val = rem_den // rem_num
                if d_val > last_denom and d_val >= 5 and d_val <= 10000000:
                    d.append(d_val)
                    if dfs(idx + 1, cur_den, cur_den, d_val):
                        return d[:]
                    d.pop()
            return None
        
        # Special case: two terms left
        if remaining_terms == 2:
            # Need 1/a + 1/b = rem_num/rem_den where a > last_denom, b > a
            # (rem_den*a + rem_den*b) / (a*b) = rem_num/rem_den
            # rem_den^2 * (a+b) = rem_num * a * b
            # rem_num * a * b - rem_den^2 * a - rem_den^2 * b = 0
            # (rem_num*a - rem_den^2)(rem_num*b - rem_den^2) = rem_den^4
            p = rem_num
            q = rem_den
            q2 = q * q
            q4 = q2 * q2
            
            # Get divisors of q4
            divisors = []
            isqrt_q4 = int(math.isqrt(q4))
            for i in range(1, isqrt_q4 + 1):
                if q4 % i == 0:
                    divisors.append(i)
                    if i != q4 // i:
                        divisors.append(q4 // i)
            
            for div in divisors:
                if time.time() > deadline:
                    return None
                # pa - q2 = div => a = (div + q2) / p
                num_a = div + q2
                if num_a % p != 0:
                    continue
                a = num_a // p
                if a <= last_denom:
                    continue
                if a > 10000000:
                    continue
                other_div = q4 // div
                num_b = other_div + q2
                if num_b % p != 0:
                    continue
                b = num_b // p
                if b <= a:
                    continue
                if b > 10000000:
                    continue
                # Verify
                if p * q * (a + b) == q * q * (a * b):
                    d.append(a)
                    d.append(b)
                    if dfs(idx + 2, cur_den, cur_den, b):
                        return d[:]
                    d.pop()
                    d.pop()
            return None
        
        # Determine range for next denominator
        # 1/d <= rem_num/rem_den => d >= rem_den/rem_num
        min_d_float = rem_den / rem_num
        min_d = max(last_denom + 1, int(math.ceil(min_d_float)))
        
        # Upper bound: d <= remaining_terms * rem_den / rem_num
        max_d_float = remaining_terms * rem_den / rem_num
        max_d = int(math.floor(max_d_float)) + 20
        
        if min_d > max_d:
            return None
        
        for d_val in range(min_d, max_d + 1):
            if time.time() > deadline:
                return None
            
            if d_val <= last_denom:
                continue
            
            # new fraction = cur_num/cur_den + 1/d_val = (cur_num*d_val + cur_den) / (cur_den*d_val)
            new_num = cur_num * d_val + cur_den
            new_den = cur_den * d_val
            
            # Reduce
            g = gcd(new_num, new_den)
            new_num //= g
            new_den //= g
            
            if new_num >= new_den:
                continue
            
            # Check if remaining is feasible
            rem2_num = new_den - new_num
            rem2_den = new_den
            rem2_float = rem2_num / rem2_den
            
            if remaining_terms - 1 > 0:
                max_rem_float = _max_sum_float(remaining_terms - 1, d_val + 1)
                if rem2_float > max_rem_float + 1e-12:
                    continue
            
            d.append(d_val)
            res = dfs(idx + 1, new_num, new_den, d_val)
            if res:
                return res
            d.pop()
        
        return None
    
    return dfs(0, 0, 1, 4)

def _search_k(k, deadline):
    """General search for k denominators using integer arithmetic"""
    d = []
    
    def dfs(idx, cur_num, cur_den, last_denom):
        if time.time() > deadline:
            return None
        
        if idx == k:
            if cur_num == cur_den:
                return d[:]
            return None
        
        remaining_terms = k - idx
        rem_num = cur_den - cur_num
        rem_den = cur_den
        
        if rem_num <= 0:
            return None
        
        max_possible_float = _max_sum_float(remaining_terms, last_denom + 1)
        if rem_num / rem_den > max_possible_float + 1e-12:
            return None
        
        if remaining_terms == 1:
            if rem_den % rem_num == 0:
                d_val = rem_den // rem_num
                if d_val > last_denom and d_val >= 5 and d_val <= 10000000:
                    d.append(d_val)
                    if dfs(idx + 1, cur_den, cur_den, d_val):
                        return d[:]
                    d.pop()
            return None
        
        min_d_float = rem_den / rem_num
        min_d = max(last_denom + 1, int(math.ceil(min_d_float)))
        max_d_float = remaining_terms * rem_den / rem_num
        max_d = int(math.floor(max_d_float)) + 20
        
        if min_d > max_d:
            return None
        
        for d_val in range(min_d, max_d + 1):
            if time.time() > deadline:
                return None
            
            if d_val <= last_denom:
                continue
            
            new_num = cur_num * d_val + cur_den
            new_den = cur_den * d_val
            g = gcd(new_num, new_den)
            new_num //= g
            new_den //= g
            
            if new_num >= new_den:
                continue
            
            rem2_num = new_den - new_num
            rem2_den = new_den
            rem2_float = rem2_num / rem2_den
            
            if remaining_terms - 1 > 0:
                max_rem_float = _max_sum_float(remaining_terms - 1, d_val + 1)
                if rem2_float > max_rem_float + 1e-12:
                    continue
            
            d.append(d_val)
            res = dfs(idx + 1, new_num, new_den, d_val)
            if res:
                return res
            d.pop()
        
        return None
    
    return dfs(0, 0, 1, 4)
# EVOLVE-BLOCK-END
