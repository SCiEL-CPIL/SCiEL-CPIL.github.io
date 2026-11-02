import math
import itertools
import time

def construct():
    deadline = time.time() + 140
    
    best_score = 0
    best_params = None
    
    # Search over base sets
    for max_elem in range(1, 50):
        if time.time() > deadline - 20:
            break
        for n in range(3, 8):
            for combo in itertools.combinations(range(1, max_elem + 1), n - 1):
                if time.time() > deadline - 20:
                    break
                B = [0] + list(combo)
                sums = set()
                diffs = set()
                for a in B:
                    for b in B:
                        sums.add(a + b)
                        diffs.add(a - b)
                s_size = len(sums)
                d_size = len(diffs)
                if s_size == 0:
                    continue
                ratio = d_size / s_size
                if ratio <= 1:
                    continue
                max_b = max(B)
                M = max_b + 1
                
                for k in range(1, 25):
                    max_u = max_b * (M**k - 1) // (M - 1)
                    if max_u < 1:
                        continue
                    score = 1 + k * math.log(ratio) / math.log(2 * max_u + 1)
                    if score > best_score:
                        best_score = score
                        best_params = (B, M, k)
    
    if best_params is None:
        return [0, 1, 3]
    
    B, M, k = best_params
    B = sorted(B)
    
    # Limit |U| to keep construction feasible
    u_size = len(B) ** k
    max_u_size = 300000
    while u_size > max_u_size and k > 1:
        k -= 1
        u_size = len(B) ** k
    
    U = set()
    for digits in itertools.product(B, repeat=k):
        val = 0
        for i, d in enumerate(digits):
            val += d * (M ** i)
        U.add(val)
    
    return sorted(U)
