import sys
from math import isqrt

sys.setrecursionlimit(2000)

def construct():
    N = 5 * 10**6
    # Sieve to compute sigma(k) for all k <= N
    sigma = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            sigma[j] += i
            
    solutions = {}
    best_M = 0
    best_count = 0
    
    for k in range(1, N + 1):
        M = k * sigma[k]
        if M <= 10**18:
            if M not in solutions:
                solutions[M] = []
            solutions[M].append(k)
            if len(solutions[M]) > best_count:
                best_count = len(solutions[M])
                best_M = M

    return {'M': best_M, 'k': solutions[best_M]}
