import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_val = float('inf')
    best_list = None
    
    # Check k=1: 210*x must be perfect square. x = 210*t^2, t>=1, x>210 => t>=2
    x1 = 840
    if 210 < x1 <= 10210:
        best_val = x1 - 210
        best_list = [x1]
    
    # Check k=2: a*b = 210*t^2, 211 <= a < b <= 10210
    for t in range(1, 220):
        if time.time() > deadline:
            break
        prod = 210 * t * t
        if prod > 10210 * 10210:
            break
        a_max = min(int(math.isqrt(prod)), 10210)
        for a in range(a_max, 210, -1):
            if prod % a == 0:
                b = prod // a
                if a < b and b <= 10210:
                    val = b - 210
                    if val < best_val:
                        best_val = val
                        best_list = [a, b]
                    break
    
    # Check k=3: a*b*c = 210*k^2, 211 <= a < b < c <= 10210
    for c in range(211, 10211):
        if time.time() > deadline:
            break
        if c - 210 >= best_val:
            break
        
        found = False
        k_max = int(math.isqrt(c * c * c // 210)) + 1
        k_min = max(1, int(math.isqrt(211 * 212 * c // 210)) + 1)
        k_cap = min(k_max, 5000)
        
        for k in range(k_min, k_cap + 1):
            if time.time() > deadline:
                break
            num = 210 * k * k
            if num % c != 0:
                continue
            prod = num // c
            if prod < 211 * 212:
                continue
            if prod >= c * c:
                break
            a_max = min(int(math.isqrt(prod)), c - 1)
            if a_max < 211:
                continue
            for a in range(211, a_max + 1):
                if prod % a == 0:
                    b = prod // a
                    if a < b < c:
                        val = c - 210
                        if val < best_val:
                            best_val = val
                            best_list = [a, b, c]
                        found = True
                        break
            if found:
                break
    
    if best_list is None:
        return [840]
        
    return best_list
