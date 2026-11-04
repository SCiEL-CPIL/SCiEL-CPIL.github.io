"""Matrix multiplication 3,3,4 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 36

    # Build schoolbook decomposition as fallback
    U_school = np.zeros((n * m, schoolbook_rank), dtype=object)
    V_school = np.zeros((m * p, schoolbook_rank), dtype=object)
    W_school = np.zeros((p * n, schoolbook_rank), dtype=object)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = Fraction(1)
                V_school[j * p + k, col] = Fraction(1)
                W_school[k * n + i, col] = Fraction(1)
                col += 1

    def verify_exact(U_mat, V_mat, W_mat, rank):
        """Verify exact decomposition using Fractions"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if s != Fraction(1):
                        return False
        return True

    def verify_float(U_f, V_f, W_f, rank, tol=1e-8):
        """Fast float verification"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_f[i*m+j, r] * V_f[j*p+k, r] * W_f[k*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def compute_error_and_grad(U_f, V_f, W_f, rank):
        """Compute error and gradients for U, V, W"""
        # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] = np.sum(U_f[i*m+j, :] * V_f[j*p+k, :] * W_f[k*n+i, :])
        
        err = np.sum((T - 1.0)**2)
        
        # Gradients
        # dE/dU[i,j,r] = 2 * sum_k (T[i,j,k]-1) * V[j,k,r] * W[k,i,r]
        dU = np.zeros_like(U_f)
        dV = np.zeros_like(V_f)
        dW = np.zeros_like(W_f)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    diff = 2.0 * (T[i,j,k] - 1.0)
                    # Contribution to U[i*m+j, :]
                    dU[i*m+j, :] += diff * V_f[j*p+k, :] * W_f[k*n+i, :]
                    # Contribution to V[j*p+k, :]
                    dV[j*p+k, :] += diff * U_f[i*m+j, :] * W_f[k*n+i, :]
                    # Contribution to W[k*n+i, :]
                    dW[k*n+i, :] += diff * U_f[i*m+j, :] * V_f[j*p+k, :]
        
        return err, dU, dV, dW

    def optimize_rank(R, seed, max_iters=2000, tol=1e-10):
        """Try to find a rank R decomposition using gradient descent"""
        np.random.seed(seed)
        
        # Initialize
        U_f = np.random.randn(n * m, R) * 0.5
        V_f = np.random.randn(m * p, R) * 0.5
        W_f = np.random.randn(p * n, R) * 0.5
        
        lr = 0.01
        momentum = 0.9
        vU = np.zeros_like(U_f)
        vV = np.zeros_like(V_f)
        vW = np.zeros_like(W_f)
        
        prev_err = np.inf
        plateau_count = 0
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            err, dU, dV, dW = compute_error_and_grad(U_f, V_f, W_f, R)
            
            # Adaptive learning rate
            if err < prev_err * 0.99:
                lr = min(lr * 1.1, 0.1)
                plateau_count = 0
            else:
                plateau_count += 1
                if plateau_count > 50:
                    lr *= 0.9
                    if lr < 1e-6:
                        break
            
            # Momentum update
            vU = momentum * vU - lr * dU
            vV = momentum * vV - lr * dV
            vW = momentum * vW - lr * dW
            
            U_f += vU
            V_f += vV
            W_f += vW
            
            prev_err = err
            
            if err < tol:
                break
        
        if err < tol:
            # Rational reconstruction
            U_rat = np.zeros((n*m, R), dtype=object)
            V_rat = np.zeros((m*p, R), dtype=object)
            W_rat = np.zeros((p*n, R), dtype=object)
            
            for i in range(n*m):
                for r in range(R):
                    U_rat[i,r] = Fraction(U_f[i,r]).limit_denominator(10**6)
            for i in range(m*p):
                for r in range(R):
                    V_rat[i,r] = Fraction(V_f[i,r]).limit_denominator(10**6)
            for i in range(p*n):
                for r in range(R):
                    W_rat[i,r] = Fraction(W_f[i,r]).limit_denominator(10**6)
            
            if verify_exact(U_rat, V_rat, W_rat, R):
                return U_rat, V_rat, W_rat, R
        
        return None

    # Search for minimal rank
    # Start from low ranks and go up
    # Known: Rank 23 for 3x3x3. 3x3x4 is likely higher.
    # Try 25 to 35.
    
    best_result = None
    for R in range(25, 36):
        if time.time() > deadline - 20:
            break
            
        found = False
        # Try multiple seeds
        for seed in range(10):
            if time.time() > deadline - 10:
                break
                
            result = optimize_rank(R, seed)
            if result:
                U_rat, V_rat, W_rat, rank = result
                best_result = (U_rat, V_rat, W_rat, rank)
                found = True
                break
        
        if found:
            break

    if best_result:
        U_rat, V_rat, W_rat, rank = best_result
        return {"U": [[float(x) for x in row] for row in U_rat], 
                "V": [[float(x) for x in row] for row in V_rat], 
                "W": [[float(x) for x in row] for row in W_rat]}
    else:
        # Fallback to schoolbook
        return {"U": [[float(x) for x in row] for row in U_school], 
                "V": [[float(x) for x in row] for row in V_school], 
                "W": [[float(x) for x in row] for row in W_school]}
# EVOLVE-BLOCK-END
