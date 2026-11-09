import time
from math import gcd
from functools import reduce
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 30
    # Calculate LCM of 1..30
    L = 1
    for i in range(1, 31):
        L = L * i // gcd(L, i)
    
    A = [0] * (N + 1)
    for i in range(1, N + 1):
        A[i] = L // i
        
    # Split into two halves
    half1 = list(range(1, 16))
    half2 = list(range(16, 31))
    
    # Generate all sums for first half
    def gen_sums_list(indices):
        current_sums = [0]
        for i in indices:
            a_i = A[i]
            add = [s + a_i for s in current_sums]
            sub = [s - a_i for s in current_sums]
            current_sums = add + sub + current_sums
        return current_sums

    sums1 = gen_sums_list(half1)
    sums1.sort()
    
    best_abs = None
    best_s1 = None
    best_s2 = None
    
    # Generate sums for second half iteratively
    def gen_sums_gen(indices, idx=0, current_sum=0):
        if idx == len(indices):
            yield current_sum
            return
        a_i = A[indices[idx]]
        yield from gen_sums_gen(indices, idx + 1, current_sum + a_i)
        yield from gen_sums_gen(indices, idx + 1, current_sum)
        yield from gen_sums_gen(indices, idx + 1, current_sum - a_i)
        
    for s2 in gen_sums_gen(half2):
        if time.time() > deadline - 5:
            break
        target = -s2
        pos = bisect.bisect_left(sums1, target)
        
        candidates = []
        if pos < len(sums1):
            candidates.append(sums1[pos])
        if pos > 0:
            candidates.append(sums1[pos-1])
            
        for s1 in candidates:
            total = s1 + s2
            if total != 0:
                abs_total = abs(total)
                if best_abs is None or abs_total < best_abs:
                    best_abs = abs_total
                    best_s1 = s1
                    best_s2 = s2
                    if best_abs == 1:
                        return reconstruct(best_s1, best_s2, half1, half2, A)
                        
    if best_s1 is None:
        return [0]*28 + [1, -1]
        
    return reconstruct(best_s1, best_s2, half1, half2, A)

def reconstruct(s1, s2, half1, half2, A):
    def find_coeffs(target, indices):
        n = len(indices)
        coeffs = [0] * n
        
        def dfs(idx, current_sum):
            if idx == n:
                return current_sum == target
            
            a_i = A[indices[idx]]
            
            coeffs[idx] = 1
            if dfs(idx + 1, current_sum + a_i):
                return True
            
            coeffs[idx] = 0
            if dfs(idx + 1, current_sum):
                return True
            
            coeffs[idx] = -1
            if dfs(idx + 1, current_sum - a_i):
                return True
            
            return False
            
        if dfs(0, 0):
            return coeffs
        return None

    c1 = find_coeffs(s1, half1)
    c2 = find_coeffs(s2, half2)
    
    if c1 is None or c2 is None:
        return [0]*28 + [1, -1]
        
    result = [0] * 30
    for i, idx in enumerate(half1):
        result[idx-1] = c1[i]
    for i, idx in enumerate(half2):
        result[idx-1] = c2[i]
        
    return result
