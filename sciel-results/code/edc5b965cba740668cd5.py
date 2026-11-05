import itertools
from math import gcd
import time

def construct():
    deadline = time.time() + 85  # 5s safety margin
    
    def pairwise_coprime(values):
        distinct = list(set(values))
        n = len(distinct)
        for i in range(n):
            for j in range(i + 1, n):
                if gcd(distinct[i], distinct[j]) != 1:
                    return False
        return True
    
    best_score = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # Check initial seed
    sums = [a + b for a in best_A for b in best_B]
    distinct = list(set(sums))
    if pairwise_coprime(distinct):
        best_score = len(distinct)
        if best_score == 9:
            return [best_A, best_B]

    # Structured Search with optimized time checks
    # We search for A in a small range and B in a larger range.
    
    for a1 in range(1, 30):
        if time.time() > deadline:
            return [best_A, best_B]
        for a2 in range(a1 + 1, 40):
            if time.time() > deadline:
                return [best_A, best_B]
            for a3 in range(a2 + 1, 50):
                if time.time() > deadline:
                    return [best_A, best_B]
                A = [a1, a2, a3]
                
                for b1 in range(1, 129):
                    s1 = [a1 + b1, a2 + b1, a3 + b1]
                    if not pairwise_coprime(s1):
                        continue
                    
                    for b2 in range(1, 129):
                        if b2 == b1:
                            continue
                        s2 = [a1 + b2, a2 + b2, a3 + b2]
                        if not pairwise_coprime(s2):
                            continue
                        combined = s1 + s2
                        if not pairwise_coprime(combined):
                            continue
                        
                        for b3 in range(1, 129):
                            if b3 == b1 or b3 == b2:
                                continue
                            s3 = [a1 + b3, a2 + b3, a3 + b3]
                            if not pairwise_coprime(s3):
                                continue
                            combined_all = s1 + s2 + s3
                            if not pairwise_coprime(combined_all):
                                continue
                            
                            # If we get here, combined_all is pairwise coprime.
                            # The score is the number of distinct sums.
                            cnt = len(set(combined_all))
                            if cnt > best_score:
                                best_score = cnt
                                best_A = A[:]
                                best_B = [b1, b2, b3]
                                if best_score == 9:
                                    return [best_A, best_B]
    
    # If not found, try random
    import random
    random.seed(42)
    for _ in range(20000):
        if time.time() > deadline:
            break
        A = random.sample(range(1, 129), 3)
        B = random.sample(range(1, 129), 3)
        sums = [a + b for a in A for b in B]
        distinct = list(set(sums))
        if pairwise_coprime(distinct):
            cnt = len(distinct)
            if cnt > best_score:
                best_score = cnt
                best_A = A
                best_B = B
                if best_score == 9:
                    break

    return [best_A, best_B]
