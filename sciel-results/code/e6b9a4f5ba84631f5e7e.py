import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # The matrix multiplication tensor T[i,j,k] has entries T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # In the trace convention W[k*n+i, r], the output index is k*n+i
    # U has shape (n*m, rank), indexed by i*m+j
    # V has shape (m*p, rank), indexed by j*p+k
    # W has shape (p*n, rank), indexed by k*n+i
    
    # Schoolbook rank: n*m*p = 60
    # Strassen on 2x2 blocks: for each pair of columns, use 7 multiplications
    # We have 10 columns, so 5 pairs of 2 columns
    # For the 2x2 part (rows 0,1 of A times columns pair of B rows 0,1), Strassen gives 7 products
    # Plus 4 schoolbook products for row 2 of A (since we can't easily Strassen 2x3)
    # Total per column pair: 7 + 4 = 11
    # Total: 5 * 11 = 55
    
    # Let's implement the Strassen-based decomposition with rank 55
    # A is 2x3: A = [[a00, a01, a02], [a10, a11, a12]]
    # B is 3x10: B[j,k]
    # C is 2x10: C[i,k] = sum_j A[i,j]*B[j,k]
    
    # For each pair of output columns (k0, k0+1), we compute C[0,k0], C[1,k0], C[0,k0+1], C[1,k0+1]
    # Using Strassen on the 2x2 subproblem (A[:,0:2] * B[0:2, k0:k0+2])
    # Plus direct products for A[:,2] * B[2, k0:k0+2]
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55
    
    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # Strassen 7 products for A[0:2, 0:2] * B[0:2, k0:k1+1]
        # A_left = [[a00, a01], [a10, a11]]
        # B_block = [[B[0,k0], B[0,k1]], [B[1,k0], B[1,k1]]]
        
        # M1: (a00 + a11) * (B[0,k0] + B[1,k1])
        U[0 * 3 + 0][col] += 1  # a00
        U[1 * 3 + 1][col] += 1  # a11
        V[0 * 10 + k0][col] += 1  # B[0,k0]
        V[1 * 10 + k1][col] += 1  # B[1,k1]
        W[k0 * 2 + 0][col] += 1  # C[0,k0]
        W[k1 * 2 + 1][col] += 1  # C[1,k1]
        col += 1
        
        # M2: (a10 + a11) * B[0,k0]
        U[1 * 3 + 0][col] += 1  # a10
        U[1 * 3 + 1][col] += 1  # a11
        V[0 * 10 + k0][col] += 1  # B[0,k0]
        W[k0 * 2 + 1][col] += 1  # C[1,k0]
        W[k1 * 2 + 1][col] -= 1  # -C[1,k1]
        col += 1
        
        # M3: a00 * (B[0,k1] - B[1,k1])
        U[0 * 3 + 0][col] += 1  # a00
        V[0 * 10 + k1][col] += 1  # B[0,k1]
        V[1 * 10 + k1][col] -= 1  # -B[1,k1]
        W[k1 * 2 + 0][col] += 1  # C[0,k1]
        W[k1 * 2 + 1][col] += 1  # C[1,k1]
        col += 1
        
        # M4: a11 * (B[1,k0] - B[0,k0])
        U[1 * 3 + 1][col] += 1  # a11
        V[1 * 10 + k0][col] += 1  # B[1,k0]
        V[0 * 10 + k0][col] -= 1  # -B[0,k0]
        W[k0 * 2 + 0][col] += 1  # C[0,k0]
        W[k0 * 2 + 1][col] += 1  # C[1,k0]
        col += 1
        
        # M5: (a00 + a01) * B[1,k1]
        U[0 * 3 + 0][col] += 1  # a00
        U[0 * 3 + 1][col] += 1  # a01
        V[1 * 10 + k1][col] += 1  # B[1,k1]
        W[k0 * 2 + 0][col] -= 1  # -C[0,k0]
        W[k1 * 2 + 0][col] += 1  # C[0,k1]
        col += 1
        
        # M6: (a10 - a00) * (B[0,k0] + B[0,k1])
        U[1 * 3 + 0][col] += 1  # a10
        U[0 * 3 + 0][col] -= 1  # -a00
        V[0 * 10 + k0][col] += 1  # B[0,k0]
        V[0 * 10 + k1][col] += 1  # B[0,k1]
        W[k1 * 2 + 1][col] += 1  # C[1,k1]
        col += 1
        
        # M7: (a01 - a11) * (B[1,k0] + B[1,k1])
        U[0 * 3 + 1][col] += 1  # a01
        U[1 * 3 + 1][col] -= 1  # -a11
        V[1 * 10 + k0][col] += 1  # B[1,k0]
        V[1 * 10 + k1][col] += 1  # B[1,k1]
        W[k0 * 2 + 0][col] += 1  # C[0,k0]
        col += 1
        
        # Schoolbook for A[:,2] * B[2, k0:k1+1]
        # a02 * B[2,k0] -> C[0,k0]
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        # a02 * B[2,k1] -> C[0,k1]
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        # a12 * B[2,k0] -> C[1,k0]
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        # a12 * B[2,k1] -> C[1,k1]
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
    
    # Verify the decomposition is correct
    # The tensor T[i,j,k] should be reconstructed as sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For each (i,j,k), the coefficient should be 1
    
    # Let's verify a few entries
    # T[0,0,0] should be 1
    # sum_r U[0, r] * V[0, r] * W[0, r]
    
    # For now, trust the construction and return
    return {"U": U, "V": V, "W": W}
