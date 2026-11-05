import time
import random
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin for 60s limit

    n = 30
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, n + 1))
    weights = [L // i for i in range(1, n + 1)]

    best_A = None
    best_coeffs = None

    def get_sum(coeffs):
        s = 0
        for i, c in enumerate(coeffs):
            if c != 0:
                s += c * weights[i]
        return s

    def local_search(coeffs, time_limit):
        nonlocal best_A, best_coeffs
        
        curr = list(coeffs)
        curr_sum = get_sum(curr)
        
        if curr_sum == 0:
            idx = random.randint(0, n-1)
            curr[idx] = random.choice([-1, 1])
            curr_sum = get_sum(curr)

        curr_abs = abs(curr_sum)
        
        if best_A is None or curr_abs < best_A:
            best_A = curr_abs
            best_coeffs = list(curr)
            
        if best_A == 1:
            return curr, curr_sum

        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            min_new_abs = curr_abs
            best_idx = -1
            best_val = 0
            
            for i in range(n):
                c_i = curr[i]
                w_i = weights[i]
                for new_val in (-1, 0, 1):
                    if new_val == c_i:
                        continue
                    delta = (new_val - c_i) * w_i
                    new_sum = curr_sum + delta
                    if new_sum == 0:
                        continue
                    new_abs = abs(new_sum)
                    if new_abs < min_new_abs:
                        min_new_abs = new_abs
                        best_idx = i
                        best_val = new_val
                        
            if best_idx != -1:
                curr[best_idx] = best_val
                curr_sum += (best_val - curr[best_idx]) * weights[best_idx] # Wait, curr[best_idx] is already updated? No, I update curr after.
                # Correction:
                # curr[best_idx] was old_val.
                # new_sum = curr_sum + (best_val - old_val) * w
                # I calculated new_sum in the loop. I should just use that.
                # Let's restructure to avoid confusion.
                
                # Actually, the loop calculates new_sum. I should store the best_new_sum too.
                pass # Placeholder for logic fix below

        return curr, curr_sum

    # Revised local search to avoid logic errors
    def local_search_v2(coeffs, time_limit):
        nonlocal best_A, best_coeffs
        
        curr = list(coeffs)
        curr_sum = get_sum(curr)
        
        if curr_sum == 0:
            idx = random.randint(0, n-1)
            curr[idx] = random.choice([-1, 1])
            curr_sum = get_sum(curr)

        curr_abs = abs(curr_sum)
        
        if best_A is None or curr_abs < best_A:
            best_A = curr_abs
            best_coeffs = list(curr)
            
        if best_A == 1:
            return curr, curr_sum

        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            min_new_abs = curr_abs
            best_idx = -1
            best_val = 0
            best_new_sum = curr_sum
            
            for i in range(n):
                c_i = curr[i]
                w_i = weights[i]
                for new_val in (-1, 0, 1):
                    if new_val == c_i:
                        continue
                    delta = (new_val - c_i) * w_i
                    new_sum = curr_sum + delta
                    if new_sum == 0:
                        continue
                    new_abs = abs(new_sum)
                    if new_abs < min_new_abs:
                        min_new_abs = new_abs
                        best_idx = i
                        best_val = new_val
                        best_new_sum = new_sum
                        
            if best_idx != -1:
                curr[best_idx] = best_val
                curr_sum = best_new_sum
                curr_abs = min_new_abs
                improved = True
                
                if curr_abs < best_A:
                    best_A = curr_abs
                    best_coeffs = list(curr)
                    if best_A == 1:
                        break

        return curr, curr_sum

    def kick(coeffs, num_kicks):
        curr = list(coeffs)
        for _ in range(num_kicks):
            idx = random.randint(0, n-1)
            curr[idx] = random.choice([-1, 0, 1])
        return curr

    random.seed(42)
    
    while (time.time() - start_time) < deadline - 2.0:
        coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        coeffs, _ = local_search_v2(coeffs, deadline - 2.0)
        
        if best_A == 1:
            break
            
        if (time.time() - start_time) < deadline - 5.0:
            kicked = kick(coeffs, 2)
            coeffs, _ = local_search_v2(kicked, deadline - 2.0)
            if best_A == 1:
                break

    if best_coeffs is None:
        return [0]*28 + [1, -1]
        
    return best_coeffs
