import sys
import numpy as np
from math import isqrt

def construct():
        N = 4 * 10**6
        sigma = np.zeros(N + 1, dtype=np.int64)
        for i in range(1, N + 1):
            sigma[i::i] += i
            
        solutions = {}
        best_M = 0
        best_count = 0
        best_list = []
        
        for k in range(1, N + 1):
            M = k * sigma[k]
            if M <= 10**18:
                if M not in solutions:
                    solutions[M] = []
                solutions[M].append(k)
                count = len(solutions[M])
                if count > best_count:
                    best_count = count
                    best_M = M
                    best_list = solutions[M]
        return {'M': int(best_M), 'k': best_list}
