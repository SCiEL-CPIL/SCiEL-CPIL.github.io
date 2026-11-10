import itertools
import math
import time

def construct():
    start_time = time.time()
    limit = 128
    
    # Precompute primes up to 256
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    primes = [i for i in range(2, 257) if is_prime(i)]
    
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = 0
    
    # We want to find A, B such that |A+B| is maximized and sums are pairwise coprime.
    # Max size is 9.
    
    # To speed up, we can iterate over A and B.
    # But 128 choose 3 is ~340k. Pairs is ~10^11.
    # We need a smarter search.
    
    # Heuristic: 
    # 1. A and B should have disjoint difference sets to ensure distinct sums.
    # 2. Sums should be primes (or pairwise coprime).
    
    # Let's try to find a solution with 9 distinct sums.
    
    # Iterate over A
    for A in itertools.combinations(range(1, limit + 1), 3):
        if time.time() - start_time > 50:
            break
        A = list(A)
        D_A = set()
        for i in range(3):
            for j in range(3):
                if i != j:
                    D_A.add(abs(A[i] - A[j]))
        
        # Iterate over B
        for B in itertools.combinations(range(1, limit + 1), 3):
            if time.time() - start_time > 50:
                break
            B = list(B)
            D_B = set()
            for i in range(3):
                for j in range(3):
                    if i != j:
                        D_B.add(abs(B[i] - B[j]))
            
            if D_A & D_B:
                continue
            
            # Check distinct sums
            sums = [a + b for a in A for b in B]
            if len(set(sums)) != 9:
                continue
            
            # Check pairwise coprime
            valid = True
            for i in range(9):
                for j in range(i + 1, 9):
                    if math.gcd(sums[i], sums[j]) != 1:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                return [A, B]
                
    # If not found, return seed
    return [[1, 7, 13], [4, 10, 16]]
