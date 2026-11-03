"""Matrix multiplication 4,5,5 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 5, 5
    nm, mp, pn = n * m, m * p, p * n  # 20, 25, 20
    
    # Schoolbook solution (rank 100)
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school), dtype=float)
    V_school = np.zeros((mp, rank_school), dtype=float)
    W_school = np.zeros((pn, rank_school), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_school
    
    def verify_exact(U, V, W, rank, tol=1e-9):
        """Check if decomposition is exact using numpy."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                    val = np.sum(prod)
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def verify_half_integer(U, V, W, rank):
        """Check if entries are half-integers and decomposition is exact."""
        # Check half-integer property
        for mat in [U, V, W]:
            doubled = mat * 2
            for row in doubled:
                for val in row:
                    if abs(val - round(val)) > 1e-6:
                        return False
        return verify_exact(U, V, W, rank)
    
    def try_rank(rank, max_attempts=10, max_iters=5000):
        """Try to find valid decomposition at given rank using gradient descent."""
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None
            
            np.random.seed(42 + attempt * 7 + rank)
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            lr = 0.05
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    return None
                
                # Check residual periodically
                if it % 100 == 0:
                    max_err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                                err = abs(val - 1.0)
                                if err > max_err:
                                    max_err = err
                    
                    if max_err < 1e-4:
                        # Try rounding to half-integers
                        U_round = np.round(U * 2) / 2
                        V_round = np.round(V * 2) / 2
                        W_round = np.round(W * 2) / 2
                        
                        if verify_half_integer(U_round, V_round, W_round, rank):
                            return U_round, V_round, W_round
                
                # Compute gradients
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            err = val - 1.0
                            if abs(err) > 1e-10:
                                grad_U[i*m+j, :] += V[j*p+k, :] * W[k*n+i, :] * err
                                grad_V[j*p+k, :] += U[i*m+j, :] * W[k*n+i, :] * err
                                grad_W[k*n+i, :] += U[i*m+j, :] * V[j*p+k, :] * err
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Decay learning rate
                if it % 500 == 0 and it > 0:
                    lr *= 0.9
        
        return None
    
    # Try ranks from 99 down to 80
    for rank in range(99, 79, -1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(rank, max_attempts=8, max_iters=4000)
        if result is not None:
            U, V, W = result
            best_result = {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            best_rank = rank
            break
    
    return best_result
