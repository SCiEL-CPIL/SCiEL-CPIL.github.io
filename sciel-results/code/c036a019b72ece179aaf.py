"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import math
import time

def construct():
    start = time.time()
    deadline = start + 82
    
    best_score = 5
    best_result = [[1, 7, 13], [4, 10, 16]]
    
    def is_pairwise_coprime(vals):
        for i in range(len(vals)):
            for j in range(i+1, len(vals)):
                if math.gcd(vals[i], vals[j]) != 1:
                    return False
        return True
    
    # A all odd, B all even to avoid even-sum gcd collisions
    odd_vals = list(range(1, 129, 2))
    even_vals = list(range(2, 129, 2))
    
    A_search = odd_vals[:35]   # 1 to 69
    B_search = even_vals[:35]  # 2 to 70
    
    for A in itertools.combinations(A_search, 3):
        if time.time() > deadline:
            return best_result
        a1, a2, a3 = A
        d12 = a2 - a1
        d13 = a3 - a1
        d23 = a3 - a2
        
        valid_b = []
        for b in B_search:
            s1 = a1 + b
            s2 = a2 + b
            s3 = a3 + b
            if math.gcd(s1, d12) == 1 and math.gcd(s1, d13) == 1 and math.gcd(s2, d23) == 1:
                valid_b.append((b, s1, s2, s3))
        
        if len(valid_b) < 3:
            continue
        
        if len(valid_b) > 22:
            valid_b = valid_b[:22]
        
        n = len(valid_b)
        for i in range(n):
            if time.time() > deadline:
                return best_result
            b1, s1_1, s2_1, s3_1 = valid_b[i]
            sums1 = {s1_1, s2_1, s3_1}
            for j in range(i+1, n):
                b2, s1_2, s2_2, s3_2 = valid_b[j]
                sums12 = sums1 | {s1_2, s2_2, s3_2}
                if not is_pairwise_coprime(list(sums12)):
                    continue
                for k in range(j+1, n):
                    b3, s1_3, s2_3, s3_3 = valid_b[k]
                    all_sums = sums12 | {s1_3, s2_3, s3_3}
                    if len(all_sums) <= best_score:
                        continue
                    if is_pairwise_coprime(list(all_sums)):
                        if len(all_sums) > best_score:
                            best_score = len(all_sums)
                            best_result = [list(A), [b1, b2, b3]]
    
    return best_result
# EVOLVE-BLOCK-END
