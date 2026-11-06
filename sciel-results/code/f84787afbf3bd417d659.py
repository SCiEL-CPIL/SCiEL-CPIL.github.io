import numpy as np
from math import gcd
from functools import reduce
from bisect import bisect_left

def construct():
    n = 30
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, n + 1))
    w = [L // i for i in range(1, n + 1)]
    
    # Split indices into two halves
    mid = n // 2
    indices1 = range(mid)
    indices2 = range(mid, n)
    
    # Generate all possible sums for the first half
    # c_i in {-1, 0, 1}
    sums1 = []
    coeffs1 = []
    
    # Recursive generation or iterative
    # 3^15 = 14,348,907. This might be slow in Python if done recursively with lists.
    # Better to use a loop or numpy broadcasting?
    # 3^15 is ~14M. Storing 14M tuples is memory heavy but doable (~1GB?).
    # Maybe too slow for 60s?
    # 14M operations in Python loop ~ 5-10s.
    # Sorting 14M elements ~ 5-10s.
    # Total ~ 20s. Risky.
    
    # Let's stick to LLL. It's faster.
    # I will fix the LLL implementation.
    
    # Actually, I can use a simpler heuristic: 
    # The minimal non-zero sum is likely small.
    # I will use the LLL code from above, but ensure it's correct.
    
    # Let's refine the LLL code to be safer.
    
    B = np.zeros((n + 1, n + 1), dtype=np.int64)
    K = 2 * sum(w) + 1
    M = K * sum(w) + 1
    
    for i in range(n):
        B[i, 0] = w[i]
        B[i, i + 1] = K
    B[n, 0] = M
    
    # Use a robust LLL implementation
    def lll_robust(B, delta=0.99):
        d = B.shape[1]
        n = B.shape[0]
        B = B.astype(np.float64) # Use float for mu calculations? 
        # No, keep B as int, use float for mu.
        
        mu = np.zeros((n, n))
        lambda_ = np.zeros((n, n))
        
        for i in range(n):
            lambda_[i][i] = np.dot(B[i], B[i])
            for j in range(i):
                mu[i][j] = np.dot(B[i], B[j]) / lambda_[j][j]
                q = int(round(mu[i][j]))
                B[i] = B[i] - q * B[j]
                lambda_[i][i] = np.dot(B[i], B[i])
                for j2 in range(j):
                    mu[i][j2] = np.dot(B[i], B[j2]) / lambda_[j2][j2]
        
        for k in range(1, n):
            while True:
                # Size reduction
                for j in range(k - 1, -1, -1):
                    q = int(round(mu[k][j]))
                    if q != 0:
                        B[k] = B[k] - q * B[j]
                        lambda_[k][k] = np.dot(B[k], B[k])
                        for i in range(j):
                            mu[k][i] -= q * mu[j][i]
                        mu[k][j] -= q
                        # Recalculate mu[k][j]
                        mu[k][j] = np.dot(B[k], B[j]) / lambda_[j][j]
                
                # Lovasz
                if lambda_[k][k] >= (delta - mu[k][k-1]**2) * lambda_[k-1][k-1]:
                    break
                else:
                    # Swap
                    B[k], B[k-1] = B[k-1], B[k].copy()
                    lambda_[k], lambda_[k-1] = lambda_[k-1], lambda_[k]
                    
                    # Update mu
                    # Recalculate mu for the swapped rows
                    for j in range(k-1):
                        mu[k-1][j] = np.dot(B[k-1], B[j]) / lambda_[j][j]
                    for j in range(k):
                        mu[k][j] = np.dot(B[k], B[j]) / lambda_[j][j]
                    
                    for i in range(k+1, n):
                        mu[i][k-1] = np.dot(B[i], B[k-1]) / lambda_[k-1][k-1]
                        mu[i][k] = np.dot(B[i], B[k]) / lambda_[k][k]
                        
        return B.astype(np.int64)

    reduced = lll_robust(B)
    
    best_sum = float('inf')
    best_coeffs = None
    
    for v in reduced[:10]:
        coeffs = []
        valid = True
        for i in range(n):
            c = v[i+1] / K
            if abs(c - round(c)) > 1e-5:
                valid = False
                break
            c = int(round(c))
            if c not in [-1, 0, 1]:
                valid = False
                break
            coeffs.append(c)
        if valid:
            s = sum(c * w[i] for i, c in enumerate(coeffs))
            if s != 0 and abs(s) < best_sum:
                best_sum = abs(s)
                best_coeffs = coeffs
                
    if best_coeffs is None:
        best_coeffs = [1, -1, -1] + [0]*27
        
    return best_coeffs

# EVOLVE-BLOCK-END
