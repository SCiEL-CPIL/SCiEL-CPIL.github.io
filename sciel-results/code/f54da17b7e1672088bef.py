import time
from math import gcd
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin for 90s limit

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

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return is_prime

    is_prime = sieve(256)
    
    # Strategy: Search for 9 distinct primes in A+B
    # For fixed A, find all b such that a1+b, a2+b, a3+b are all prime
    # Then pick 3 such b's where all 9 sums are distinct
    
    max_val = 128
    
    # Search smaller A values first (more likely to have many valid b's)
    for a1 in range(1, 30):
        if time.time() > deadline:
            return [best_A, best_B]
        for a2 in range(a1+1, 30):
            if time.time() > deadline:
                return [best_A, best_B]
            for a3 in range(a2+1, 30):
                if time.time() > deadline:
                    return [best_A, best_B]
                
                A = (a1, a2, a3)
                
                # Find all b in [1, 128] such that a1+b, a2+b, a3+b are prime
                valid_bs = []
                for b in range(1, max_val + 1):
                    if is_prime[a1+b] and is_prime[a2+b] and is_prime[a3+b]:
                        valid_bs.append(b)
                
                if len(valid_bs) < 3:
                    continue
                
                # Check ALL combinations of 3 valid_bs
                for b1, b2, b3 in combinations(valid_bs, 3):
                    B = (b1, b2, b3)
                    sums = set()
                    for ai in A:
                        for bi in B:
                            sums.add(ai + bi)
                    
                    if len(sums) == 9:
                        if is_pairwise_coprime(sums):
                            return [list(A), list(B)]
    
    # Fallback: Try AP search for 6-8 distinct sums
    max_a = 40
    max_d = 15
    max_b = 40
    max_e = 15
    
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
                    
                    sums = set()
                    for ai in A:
                        for bi in B:
                            sums.add(ai + bi)
                    
                    if len(sums) > best_count:
                        if is_pairwise_coprime(sums):
                            best_count = len(sums)
                            best_A = list(A)
                            best_B = list(B)
                            if best_count >= 8:
                                return [best_A, best_B]

    return [best_A, best_B]

if __name__ == "__main__":
    A, B = construct()
    print(f"Found A={A}, B={B}")
    sums = set(a+b for a in A for b in B)
    print(f"Distinct sums: {len(sums)}, Values: {sorted(sums)}")
