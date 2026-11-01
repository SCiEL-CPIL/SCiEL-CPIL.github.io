import numpy as np
from itertools import product
import time

DIMENSIONS = [2, 2, 14]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 10 second safety margin
    
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    best = {'U': U_sb, 'V': V_sb, 'W': W_sb, 'rank': rank_sb}
    
    # Try to find lower rank using numerical optimization
    # The tensor constraint: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # And for all other index combos: sum = 0
    
    def residual(U, V, W, r):
        """Compute the residual of the tensor decomposition."""
        res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    res += (s - 1)**2
        # Zero constraints (sample a subset for speed)
        # Total zero constraints: n*m*m*p*p*n - n*m*p = 4*28*28 - 56 = 3080
        # Sample some
        import random
        if time.time() > deadline - 5:
            return res
        for _ in range(100):  # Sample 100 zero constraints
            i = random.randint(0, n-1)
            j = random.randint(0, m-1)
            jp = random.randint(0, m-1)
            k = random.randint(0, p-1)
            ip = random.randint(0, n-1)
            if i == ip and j == jp:
                continue  # This is a non-zero constraint
            s = 0.0
            for c in range(r):
                s += U[i*m+j][c] * V[jp*p+k][c] * W[k*n+ip][c]
            res += s**2
        return res
    
    # Try ranks from 28 up to 55
    for r in range(28, min(56, rank_sb)):
        if time.time() > deadline - 10:
            break
        # Try a few random initializations
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            np.random.seed(42 + trial)
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1
            
            # Simple gradient descent (very crude)
            lr = 0.01
            for step in range(1000):
                if time.time() > deadline - 5:
                    break
                # Compute gradient (crude)
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                            err = s - 1
                            for c in range(r):
                                dU[i*m+j][c] += 2 * err * V[j*p+k][c] * W[k*n+i][c]
                                dV[j*p+k][c] += 2 * err * U[i*m+j][c] * W[k*n+i][c]
                                dW[k*n+i][c] += 2 * err * U[i*m+j][c] * V[j*p+k][c]
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            # Check if close to solution
            res = residual(U.tolist(), V.tolist(), W.tolist(), r)
            if res < 1e-6:
                # Found a solution! Round to half-integers and verify
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                res_r = residual(U_r.tolist(), V_r.tolist(), W_r.tolist(), r)
                if res_r < 1e-6:
                    best = {'U': U_r.tolist(), 'V': V_r.tolist(), 'W': W_r.tolist(), 'rank': r}
                    break
        if best['rank'] < rank_sb:
            break
    
    return {'U': best['U'], 'V': best['V'], 'W': best['W']}
