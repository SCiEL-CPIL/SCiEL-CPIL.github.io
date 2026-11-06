import random
from math import gcd
from itertools import combinations
import time

def pairwise_coprime(lst):
    n = len(lst)
    for i in range(n):
        for j in range(i+1, n):
            if gcd(lst[i], lst[j]) != 1:
                return False
    return True

def score_of(A, B):
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    sl = list(sums)
    n = len(sl)
    if n < 2:
        return 0
    for i in range(n):
        for j in range(i+1, n):
            if gcd(sl[i], sl[j]) != 1:
                return 0
    return n

def construct():
    best_score = 5
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    start_time = time.time()
    time_limit = 55
    
    # Phase 1: Structured search in small range
    for A in combinations(range(1, 31), 3):
        if time.time() - start_time > time_limit:
            break
        A = list(A)
        for B in combinations(range(1, 31), 3):
            B = list(B)
            s = score_of(A, B)
            if s > best_score:
                best_score = s
                best_A = A[:]
                best_B = B[:]
                if best_score == 9:
                    return [best_A, best_B]
    
    # Phase 2: Random search in full range
    random.seed(42)
    for _ in range(3000000):
        if time.time() - start_time > time_limit:
            break
        A = random.sample(range(1, 129), 3)
        B = random.sample(range(1, 129), 3)
        s = score_of(A, B)
        if s > best_score:
            best_score = s
            best_A = A[:]
            best_B = B[:]
            if best_score == 9:
                break
    
    return [best_A, best_B]
