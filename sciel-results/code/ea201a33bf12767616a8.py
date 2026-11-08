"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    rank = len(U[0])
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower rank decompositions
    # The tensor structure: for each (i,j,k), sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    
    # Strategy: use random initialization and iterative refinement
    # For a given rank R, we have R rank-1 terms
    # Each term r contributes U_r[i,j] * V_r[j,k] * W_r[k,i]
    
    # Let's try ranks from 20 up to schoolbook
    for R in range(20, rank_school):
        if time.time() > deadline:
            break
        
        # Multiple random restarts
        for restart in range(50):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = [[random.uniform(-1, 1) for _ in range(R)] for _ in range(n*m)]
            V = [[random.uniform(-1, 1) for _ in range(R)] for _ in range(m*p)]
            W = [[random.uniform(-1, 1) for _ in range(R)] for _ in range(p*n)]
            
            # Iterative refinement: alternate solving for each factor
            for it in range(200):
                if time.time() > deadline:
                    break
                
                # Update U: for each (i,j), solve for U[i,j,:] 
                # The residual for (i,j) is: 1 - sum_k sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
                # This is nonlinear, so use a simple update
                
                # Actually, let's use a different approach: 
                # For fixed V and W, the equation is linear in U
                # For each (i,j): sum_r U[i,j,r] * (sum_k V[j,k,r]*W[k,i,r]) = 1
                # Let A[i,j,r] = sum_k V[j,k,r]*W[k,i,r]
                # Then U[i,j,:] is a vector such that U[i,j,:] dot A[i,j,:] = 1
                # Minimum norm solution: U[i,j,r] = A[i,j,r] / ||A[i,j,:]||^2
                
                A = [[0]*R for _ in range(n*m)]
                for i in range(n):
                    for j in range(m):
                        for r in range(R):
                            s = 0.0
                            for k in range(p):
                                s += V[j*p+k][r] * W[k*n+i][r]
                            A[i*m+j][r] = s
                
                for i in range(n):
                    for j in range(m):
                        norm_sq = sum(A[i*m+j][r]**2 for r in range(R))
                        if norm_sq < 1e-15:
                            continue
                        for r in range(R):
                            U[i*m+j][r] = A[i*m+j][r] / norm_sq
                
                # Similarly update V
                B = [[0]*R for _ in range(m*p)]
                for j in range(m):
                    for k in range(p):
                        for r in range(R):
                            s = 0.0
                            for i in range(n):
                                s += U[i*m+j][r] * W[k*n+i][r]
                            B[j*p+k][r] = s
                
                for j in range(m):
                    for k in range(p):
                        norm_sq = sum(B[j*p+k][r]**2 for r in range(R))
                        if norm_sq < 1e-15:
                            continue
                        for r in range(R):
                            V[j*p+k][r] = B[j*p+k][r] / norm_sq
                
                # Similarly update W
                C = [[0]*R for _ in range(p*n)]
                for k in range(p):
                    for i in range(n):
                        for r in range(R):
                            s = 0.0
                            for j in range(m):
                                s += U[i*m+j][r] * V[j*p+k][r]
                            C[k*n+i][r] = s
                
                for k in range(p):
                    for i in range(n):
                        norm_sq = sum(C[k*n+i][r]**2 for r in range(R))
                        if norm_sq < 1e-15:
                            continue
                        for r in range(R):
                            W[k*n+i][r] = C[k*n+i][r] / norm_sq
                
                # Check convergence
                if it % 20 == 19:
                    if _verify(U, V, W, n, m, p, tol=1e-8):
                        # Found a valid decomposition at rank R
                        best_rank = R
                        best_U, best_V, best_W = U, V, W
                        break
            
            if best_rank == R:
                break
        
        if best_rank == R:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
