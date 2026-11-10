import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 48  # Safety margin for 90s limit

    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_count = 5

    def is_pairwise_coprime(values):
        vals = list(set(values))
        for i in range(len(vals)):
            for j in range(i+1, len(vals)):
                if gcd(vals[i], vals[j]) != 1:
                    return False
        return True

    # Strategy: Use arithmetic progression families to ensure distinct sums
    # For A = {a, a+d, a+2d} and B = {b, b+e, b+2e}, all 9 sums are distinct
    # if the differences d, e, 2d, 2e, d-e, d+e, 2d-e, 2d+e, 2d-2e are all nonzero
    # and no collisions occur.
    
    # Try small arithmetic progressions first
    max_a = 50
    max_d = 20
    max_b = 50
    max_e = 20
    
    for a in range(1, max_a + 1):
        if time.time() > deadline:
            return [best_A, best_B]
        for d in range(1, max_d + 1):
            if a + 2*d > 128:
                continue
            A = (a, a+d, a+2*d)
            
            for b in range(1, max_b + 1):
                if time.time() > deadline:
                    return [best_A, best_B]
                for e in range(1, max_e + 1):
                    if b + 2*e > 128:
                        continue
                    B = (b, b+e, b+2*e)
                    
                    # Compute all 9 sums
                    sums = set()
                    for ai in A:
                        for bi in B:
                            sums.add(ai + bi)
                    
                    if len(sums) <= best_count:
                        continue
                    
                    if is_pairwise_coprime(sums):
                        best_count = len(sums)
                        best_A = list(A)
                        best_B = list(B)
                        
                        if best_count == 9:
                            return [best_A, best_B]
    
    # If AP families didn't find 9, try more general search with pruning
    for a1 in range(1, 31):
        if time.time() > deadline:
            return [best_A, best_B]
        for a2 in range(a1+1, 31):
            if time.time() > deadline:
                return [best_A, best_B]
            for a3 in range(a2+1, 31):
                if time.time() > deadline:
                    return [best_A, best_B]
                
                A = (a1, a2, a3)
                
                # Find valid b's that make a1+b, a2+b, a3+b pairwise coprime
                valid_bs = []
                for b in range(1, 129):
                    s1, s2, s3 = a1+b, a2+b, a3+b
                    if gcd(s1, s2) == 1 and gcd(s1, s3) == 1 and gcd(s2, s3) == 1:
                        valid_bs.append(b)
                
                if len(valid_bs) < 3:
                    continue
                
                # Try combinations, but limit to first 100 for speed
                for b1, b2, b3 in combinations(valid_bs[:100], 3):
                    B = (b1, b2, b3)
                    sums = set()
                    for a in A:
                        for b in B:
                            sums.add(a+b)
                    
                    if len(sums) <= best_count:
                        continue
                    
                    if is_pairwise_coprime(sums):
                        best_count = len(sums)
                        best_A = list(A)
                        best_B = list(B)
                        
                        if best_count == 9:
                            return [best_A, best_B]
    
    return [best_A, best_B]

if __name__ == "__main__":
    # Quick self-test
    A, B = construct()
    print(f"Found A={A}, B={B}")
    sums = set(a+b for a in A for b in B)
    print(f"Distinct sums: {len(sums)}, Values: {sorted(sums)}")
