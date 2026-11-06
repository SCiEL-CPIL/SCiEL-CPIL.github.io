import time
import math
from fractions import Fraction
from math import gcd

def construct():
    start_time = time.time()
    time_limit = 85.0 # Safety margin for 90s limit
    
    best_solution = [8, 28, 1736]
    best_max_denom = 1736
    
    def update_best(sol):
        nonlocal best_solution, best_max_denom
        if not sol:
            return
        m = max(sol)
        if m < best_max_denom:
            best_solution = sol
            best_max_denom = m

    # 1. Search for 3-term solutions: 5/31 = 1/a + 1/b + 1/c
    # 1/c = 5/31 - 1/a - 1/b
    # c = 31ab / (5ab - 31b - 31a)
    
    # a >= 7
    # Prune a: 1/a + 1/(a+1) + 1/(a+2) < 5/31
    for a in range(7, 10000):
        if time.time() - start_time > time_limit - 5:
            break
        
        if (1/a + 1/(a+1) + 1/(a+2)) < 5/31:
            break
            
        for b in range(a + 1, 100000):
            if time.time() - start_time > time_limit - 5:
                break
                
            if (1/a + 1/b + 1/(b+1)) < 5/31:
                break
                
            num = 5 * a * b - 31 * b - 31 * a
            den = 31 * a * b
            
            if num <= 0:
                continue
                
            if den % num == 0:
                c = den // num
                if c > b:
                    update_best([a, b, c])

    # 2. Search for 4-term solutions: 5/31 = 1/a + 1/b + 1/c + 1/d
    # 1/d = 5/31 - 1/a - 1/b - 1/c
    # d = 31abc / (5abc - 31bc - 31ac - 31ab)
    
    if time.time() - start_time < time_limit - 10:
        for a in range(7, 1000):
            if time.time() - start_time > time_limit - 10:
                break
            if (1/a + 1/(a+1) + 1/(a+2) + 1/(a+3)) < 5/31:
                break
                
            for b in range(a + 1, 10000):
                if time.time() - start_time > time_limit - 10:
                    break
                if (1/a + 1/b + 1/(b+1) + 1/(b+2)) < 5/31:
                    break
                    
                num_ab = 5 * a * b - 31 * a - 31 * b
                if num_ab <= 0:
                    continue
                
                # c_min from 1/c < 5/31 - 1/a - 1/b
                # c > 31ab / (5ab - 31a - 31b)
                den_ab = 31 * a * b
                c_min = den_ab // num_ab + 1
                if c_min <= b:
                    c_min = b + 1
                    
                # c_max from 1/d > 0 and d > c
                # 5/31 - 1/a - 1/b - 1/c > 1/c
                # 5/31 - 1/a - 1/b > 2/c
                # c > 2 / (5/31 - 1/a - 1/b) is wrong direction?
                # 1/d = rem_ab - 1/c
                # d > c => 1/d < 1/c
                # rem_ab - 1/c < 1/c => rem_ab < 2/c => c < 2/rem_ab
                # c < 2 * den_ab / num_ab
                c_max = (2 * den_ab) // num_ab
                
                if c_max < c_min:
                    continue
                
                search_c_max = min(c_max, best_max_denom - 2)
                
                for c in range(c_min, search_c_max + 1):
                    if time.time() - start_time > time_limit - 10:
                        break
                        
                    num_abc = 5 * a * b * c - 31 * b * c - 31 * a * c - 31 * a * b
                    den_abc = 31 * a * b * c
                    
                    if num_abc <= 0:
                        continue
                        
                    if den_abc % num_abc == 0:
                        d = den_abc // num_abc
                        if d > c and d < best_max_denom:
                            update_best([a, b, c, d])

    return best_solution
