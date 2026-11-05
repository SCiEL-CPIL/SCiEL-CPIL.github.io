import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    def max_sum_from(start_d, count):
        s = Fraction(0, 1)
        for i in range(1, count + 1):
            s += Fraction(1, start_d + i)
        return s

    def dfs(idx, target, last_d, current):
        if time.time() > deadline:
            return None
        
        if idx == 8:
            if target == 0:
                return current
            return None
        
        # Determine the minimum valid d
        # We need 1/d <= target => d >= 1/target
        # target = p/q. 1/target = q/p.
        # d_min = ceil(q/p)
        p = target.numerator
        q = target.denominator
        if p == 0:
            return None # Should not happen
        
        d_min = (q + p - 1) // p
        if d_min < last_d + 1:
            d_min = last_d + 1
            
        d = d_min
        
        while d <= 10000000:
            if time.time() > deadline:
                return None
            
            # 1/d is guaranteed <= target here
            
            rem_target = target - Fraction(1, d)
            
            if idx == 7:
                # Last term
                if rem_target == 0:
                    return current + [d]
                else:
                    # Need 1/d_next = rem_target
                    # rem_target must be 1/k
                    if rem_target.numerator == 1:
                        d_next = rem_target.denominator
                        if d_next > d and d_next <= 10000000:
                            return current + [d, d_next]
                    d += 1
                    continue
            
            # Pruning
            max_rem_sum = max_sum_from(d, 8 - idx - 1)
            if rem_target > max_rem_sum:
                break
            
            res = dfs(idx + 1, rem_target, d, current + [d])
            if res is not None:
                return res
            
            d += 1
            
        return None

    result = dfs(0, Fraction(1, 1), 4, [])
    
    if result:
        if len(result) == 8 and all(5 <= d <= 10000000 for d in result) and len(set(result)) == 8:
            s = sum(Fraction(1, d) for d in result)
            if s == 1:
                return result
        
    return []
