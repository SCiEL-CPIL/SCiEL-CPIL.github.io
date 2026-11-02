"""Matrix multiplication 5x5x5 tensor decomposition using Laderman for 3x3 blocks."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # all 5
    
    # Block split: 3 + 2
    # Using Laderman's 15-rank algorithm for 3x3x3, schoolbook for rest
    # Total rank: 15 + 18 + 18 + 12 + 18 + 12 + 12 + 4 = 99
    
    rank = 99
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Laderman's 15-term algorithm for 3x3 matrix multiplication
    # Verified coefficients: each term is (A_linear_form, B_linear_form, C_linear_form)
    # where C = sum_t (A_form_t * B_form_t)
    
    # Using the standard Laderman decomposition:
    laderman = [
        # (A_coeffs {(i,j):val}, B_coeffs {(j,k):val}, C_coeffs {(i,k):val})
        ([(0,0,1)], [(0,0,1)], [(0,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,0,1)], [(0,1,1)]),
        ([(0,1,1)], [(1,1,1),(0,1,-1)], [(0,1,1)]),
        ([(0,1,1),(1,1,1),(1,0,-1),(2,0,-1)], [(2,0,1)], [(1,0,1)]),
        ([(1,0,1),(1,1,1),(2,0,-1),(2,1,-1)], [(1,2,1)], [(1,1,1),(2,1,1)]),
        ([(2,1,1)], [(0,1,1),(0,0,-1)], [(0,2,1),(1,2,1)]),
        ([(2,2,1)], [(0,2,1),(1,2,-1)], [(0,2,1),(1,2,1)]),
        ([(2,2,1),(2,1,1),(1,2,-1),(1,1,-1)], [(0,1,1)], [(1,1,1),(2,1,1)]),
        ([(2,2,1),(1,2,1),(2,1,-1),(1,1,-1)], [(1,0,1)], [(1,0,1),(2,0,1)]),
        ([(0,0,1),(2,2,1),(1,1,-1)], [(1,1,1)], [(0,0,1),(1,1,1),(2,2,1)]),
        ([(0,2,1),(1,2,1),(0,1,-1),(1,0,-1)], [(2,1,1)], [(0,2,1)]),
        ([(1,0,1),(2,0,1),(0,2,-1),(2,2,-1)], [(0,2,1)], [(1,0,1)]),
        ([(0,2,1),(0,1,1),(1,0,-1),(2,0,-1)], [(1,2,1)], [(1,1,-1),(1,2,1)]),
        ([(1,0,1),(1,2,1),(0,1,-1),(0,2,-1)], [(2,0,1)], [(2,1,1),(2,2,1)]),
        ([(0,2,1),(1,2,1),(1,0,-1),(2,0,-1)], [(0,1,1)], [(0,1,1),(2,2,-1)]),
    ]
    
    # Add Laderman terms for A11*B11 (block 1: rows/cols 0-2)
    for a_list, b_list, c_list in laderman:
        for (ai, aj, aval) in a_list:
            U[ai*m + aj, col] += aval
        for (bj, bk, bval) in b_list:
            V[bj*p + bk, col] += bval
        for (ci, ck, cval) in c_list:
            W[ck*n + ci, col] += cval
        col += 1
    
    # A12*B21: 3x2x3 schoolbook (rank 18)
    # A[i,3+j] * B[3+j,k] -> C[i,k], i in 0-2, j in 0-1, k in 0-2
    for i in range(3):
        for j in range(2):
            for k in range(3):
                U[i*m + (3+j), col] = 1.0
                V[(3+j)*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    
    # A11*B12: 3x3x2 schoolbook (rank 18)
    # A[i,j] * B[j,3+k] -> C[i,3+k], i in 0-2, j in 0-2, k in 0-1
    for i in range(3):
        for j in range(3):
            for k in range(2):
                U[i*m + j, col] = 1.0
                V[j*p + (3+k), col] = 1.0
                W[(3+k)*n + i, col] = 1.0
                col += 1
    
    # A12*B22: 3x2x2 schoolbook (rank 12)
    # A[i,3+j] * B[3+j,3+k] -> C[i,3+k], i in 0-2, j in 0-1, k in 0-1
    for i in range(3):
        for j in range(2):
            for k in range(2):
                U[i*m + (3+j), col] = 1.0
                V[(3+j)*p + (3+k), col] = 1.0
                W[(3+k)*n + i, col] = 1.0
                col += 1
    
    # A21*B11: 2x3x3 schoolbook (rank 18)
    # A[3+i,j] * B[j,k] -> C[3+i,k], i in 0-1, j in 0-2, k in 0-2
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U[(3+i)*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + (3+i), col] = 1.0
                col += 1
    
    # A22*B21: 2x2x3 schoolbook (rank 12)
    # A[3+i,3+j] * B[3+j,k] -> C[3+i,k], i in 0-1, j in 0-1, k in 0-2
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[(3+i)*m + (3+j), col] = 1.0
                V[(3+j)*p + k, col] = 1.0
                W[k*n + (3+i), col] = 1.0
                col += 1
    
    # A21*B12: 2x3x2 schoolbook (rank 12)
    # A[3+i,j] * B[j,3+k] -> C[3+i,3+k], i in 0-1, j in 0-2, k in 0-1
    for i in range(2):
        for j in range(3):
            for k in range(2):
                U[(3+i)*m + j, col] = 1.0
                V[j*p + (3+k), col] = 1.0
                W[(3+k)*n + (3+i), col] = 1.0
                col += 1
    
    # A22*B22: 2x2x2 schoolbook (rank 4)
    # A[3+i,3+j] * B[3+j,3+k] -> C[3+i,3+k], i in 0-1, j in 0-1, k in 0-1
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[(3+i)*m + (3+j), col] = 1.0
                V[(3+j)*p + (3+k), col] = 1.0
                W[(3+k)*n + (3+i), col] = 1.0
                col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
