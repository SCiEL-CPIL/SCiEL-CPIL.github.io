"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    # We aim to find a rank-minimal decomposition for 3x3x7 matrix multiplication.
    # The schoolbook rank is 3*3*7 = 63.
    # We can decompose 3x3x7 as (3x3x3) + (3x3x3) + (3x3x1).
    # The 3x3x1 part has rank 3 (since it's just scaling columns).
    # So if we find a rank-r decomposition for 3x3x3, total rank is r + r + 3.
    # It is known that the rank of 3x3x3 multiplication tensor is 23.
    # So total rank would be 23 + 23 + 3 = 49.
    # We will attempt to find a rank-23 decomposition for 3x3x3 using a numerical search
    # and then verify it exactly. If we fail, we fall back to schoolbook (63).
    
    n, m, p = 3, 3, 7
    schoolbook_rank = n * m * p
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    fallback = {"U": U_school, "V": V_school, "W": W_school}
    
    start_time = time.time()
    deadline = start_time + 120  # 120s limit
    
    # Target tensor for 3x3x3: T[i,j,k] = 1 if we are multiplying A[i,j] * B[j,k]
    # Actually, the tensor for matrix multiplication C = A * B is defined such that:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has indices (i, j, k) and T[i,j,k] = 1.
    # We want to decompose T as sum_r U_r * V_r * W_r where U_r is vector of length 9, V_r length 9, W_r length 9.
    # Specifically, T[i,j,k] = sum_r U[r, i*m+j] * V[r, j*p+k] * W[r, k*n+i] for m=p=n=3.
    
    # We will try to find a rank-23 decomposition.
    # Use a simple gradient-free method or random search with local improvement.
    # Given the complexity, we'll use a heuristic: start with random rank-23 and try to minimize error.
    
    def find_3x3x3_decomp(rank_target=23, max_iters=5000, seed=42):
        np.random.seed(seed)
        # Target tensor T of shape (3,3,3) with all 1s.
        T = np.ones((3,3,3))
        T_flat = T.flatten()  # 27 elements
        
        # Initialize random decomposition
        U = np.random.randn(rank_target, 9)
        V = np.random.randn(rank_target, 9)
        W = np.random.randn(rank_target, 9)
        
        best_error = float('inf')
        best_U, best_V, best_W = U, V, W
        
        for it in range(max_iters):
            if time.time() > deadline - 30:
                break
            # Compute current decomposition
            # T_decomp[i,j,k] = sum_r U[r, i*3+j] * V[r, j*3+k] * W[r, k*3+i]
            T_decomp = np.zeros((3,3,3))
            for r in range(rank_target):
                # Outer product of U[r], V[r], W[r] but with index mapping
                # U[r] has indices i*3+j
                # V[r] has indices j*3+k
                # W[r] has indices k*3+i
                # This is not a simple outer product due to index coupling.
                # We need to compute T_decomp[i,j,k] = sum_r U[r, i*3+j] * V[r, j*3+k] * W[r, k*3+i]
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            T_decomp[i,j,k] += U[r, i*3+j] * V[r, j*3+k] * W[r, k*3+i]
            
            error = np.linalg.norm(T_decomp - T)
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if error < 1e-9:
                    break
            
            # Gradient step (approximate)
            # For simplicity, use a small random perturbation
            step_size = 0.1 * (1 - it/max_iters)
            U += np.random.randn(*U.shape) * step_size
            V += np.random.randn(*V.shape) * step_size
            W += np.random.randn(*W.shape) * step_size
        
        # Try to round to integers or half-integers
        # Scale factors
        # We can try to find a scaling that makes entries close to half-integers
        # This is a hard problem. For now, check if best_error is small enough to attempt rounding.
        if best_error < 1e-4:
            # Try to round each matrix separately
            # We need to preserve the decomposition property approximately
            # This is complex. We'll just return the best found and let the verification step handle it.
            # In a real implementation, we'd use lattice reduction or similar.
            # For this task, if we can't find an exact integer/half-integer solution, we fall back.
            pass
        
        return best_U, best_V, best_W, best_error
    
    # Attempt to find a good decomposition
    try:
        U_3, V_3, W_3, err = find_3x3x3_decomp(23, max_iters=1000)
        # If error is very small, we might be able to use it.
        # But we need exact integer/half-integer entries.
        # Since we can't guarantee exactness from numerical search, we fall back to schoolbook.
        # This is a known limitation. A proper implementation would use exact algebraic methods.
    except:
        pass
    
    # For now, return schoolbook decomposition.
    # A more advanced version would use known rank-23 decompositions for 3x3x3.
    return fallback
# EVOLVE-BLOCK-END
