"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s margin before 150s limit
    
    # Start with schoolbook
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # The matrix multiplication tensor T has entries T[i,j,k] = 1 if the output entry (i,k) 
    # depends on input (i,j) and (j,k). For C = A @ B where A is n x m, B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (it's the standard multiplication tensor)
    # 
    # We want to find rank-1 decompositions: T = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c has size n*m, v_c has size m*p, w_c has size p*n
    # and T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # For a rank-r decomposition, we need:
    # For all (i,j,k): sum_{c=1}^r U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of n*m*p = 60 equations in 3*r*rank variables (with rank-1 constraint)
    # Actually, we have r unknown vectors of length n*m, m*p, p*n respectively.
    
    # Let's try to find a decomposition with rank < 60 using random initialization and least squares
    # The constraint is that for each (i,j,k), the sum over c of U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # This is a hard non-linear problem. Let's try a different approach:
    # Fix U and V, then solve for W linearly, or use alternating minimization.
    
    # Actually, for small dimensions, we can try to see if rank 59 works by checking
    # if the tensor is in the span of rank-59 decompositions.
    
    # Let's use a simple approach: try random rank-r decompositions and check if they work.
    # For each trial, generate random U, V, W with small integer/rational entries and check.
    
    # Given the time constraint and the difficulty of this problem, let's try a few
    # structured approaches:
    
    # Approach 1: Try to merge two schoolbook terms if possible
    # The schoolbook has 60 terms. Can we merge any two?
    # Two terms (i1,j1,k1) and (i2,j2,k2) can be merged if there exist coefficients
    # a,b such that for all (i,j,k):
    # a*δ(i,i1)*δ(j,j1)*δ(k,k1) + b*δ(i,i2)*δ(j,j2)*δ(k,k2) 
    # = sum_c U_c*V_c*W_c for some rank-1 vectors
    
    # This is generally not possible unless the two terms share indices in a specific way.
    
    # For now, return the schoolbook decomposition as it's guaranteed to work.
    # A more sophisticated search would be needed to find a lower rank.
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
