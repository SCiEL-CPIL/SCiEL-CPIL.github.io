import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # We aim for rank 30
    # 4 Strassen blocks (rank 28) + 2 Rank-1 blocks (rank 2)
    rank = 30
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block
    def add_strassen(i0, j0, k0, c_start):
        nonlocal col
        # A sub-block: rows i0, i0+1; cols j0, j0+1
        # B sub-block: rows j0, j0+1; cols k0, k0+1
        # C sub-block: rows i0, i0+1; cols k0, k0+1
        
        # M1 = (A00 + A11)(B00 + B11)
        # U: A00, A11. V: B00, B11. W: C00, C11 (with signs)
        
        # We need to be careful with the indices in U, V, W.
        # U[i*m+j], V[j*p+k], W[k*n+i]
        
        # Term 1: M1
        # U[A00] += 1, U[A11] += 1
        # V[B00] += 1, V[B11] += 1
        # W[C00] += 1, W[C11] += 1
        
        # Wait, Strassen computes M's as intermediate products.
        # The decomposition is sum of rank-1 tensors.
        # M1 corresponds to 7 rank-1 tensors? No, M1 is one product of sums.
        # (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
        # This is not rank 1.
        # Strassen algorithm reduces the number of multiplications.
        # The tensor decomposition uses the 7 products as the "rank-1" terms?
        # No, the rank is the number of terms in the sum.
        # Each Strassen term $M_k$ is a product of a sum of A entries and a sum of B entries.
        # $M_k = (\sum u_{ij} A_{ij}) (\sum v_{jk} B_{jk})$.
        # This is a rank-1 tensor in the A and B spaces, but combined with W?
        # The term in the decomposition is $W_{ki} \cdot (\sum u A) \cdot (\sum v B)$.
        # So each $M_k$ contributes 1 to the rank.
        
        # So for each Strassen block, we add 7 columns to U, V, W.
        
        # M1
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C11=1
        U[i0*m + j0][c_start] = 1
        U[(i0+1)*m + (j0+1)][c_start] = 1
        V[j0*p + k0][c_start] = 1
        V[(j0+1)*p + (k0+1)][c_start] = 1
        W[k0*n + i0][c_start] = 1
        W[(k0+1)*n + (i0+1)][c_start] = 1
        col += 1
        
        # M2 = (A10 + A11)B00
        U[(i0+1)*m + j0][c_start] = 1
        U[(i0+1)*m + (j0+1)][c_start] = 1
        V[j0*p + k0][c_start] = 1
        W[k0*n + (i0+1)][c_start] = 1
        W[(k0+1)*n + (i0+1)][c_start] = 1
        col += 1
        
        # M3 = A00(B10 - B11)
        U[i0*m + j0][c_start] = 1
        V[(j0+1)*p + k0][c_start] = 1
        V[(j0+1)*p + (k0+1)][c_start] = -1
        W[k0*n + i0][c_start] = 1
        W[(k0+1)*n + i0][c_start] = -1
        col += 1
        
        # M4 = A11(B01 - B00)
        U[(i0+1)*m + (j0+1)][c_start] = 1
        V[j0*p + (k0+1)][c_start] = 1
        V[j0*p + k0][c_start] = -1
        W[(k0+1)*n + (i0+1)][c_start] = 1
        W[k0*n + (i0+1)][c_start] = -1
        col += 1
        
        # M5 = (A00 + A01)B11
        U[i0*m + j0][c_start] = 1
        U[i0*m + (j0+1)][c_start] = 1
        V[(j0+1)*p + (k0+1)][c_start] = 1
        W[(k0+1)*n + i0][c_start] = 1
        W[(k0+1)*n + (i0+1)][c_start] = 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B10)
        U[(i0+1)*m + j0][c_start] = 1
        U[i0*m + j0][c_start] = -1
        V[j0*p + k0][c_start] = 1
        V[(j0+1)*p + k0][c_start] = 1
        W[k0*n + (i0+1)][c_start] = 1
        W[k0*n + i0][c_start] = -1
        col += 1
        
        # M7 = (A01 - A11)(B11 + B01)
        U[i0*m + (j0+1)][c_start] = 1
        U[(i0+1)*m + (j0+1)][c_start] = -1
        V[(j0+1)*p + (k0+1)][c_start] = 1
        V[j0*p + (k0+1)][c_start] = 1
        W[(k0+1)*n + i0][c_start] = 1
        W[(k0+1)*n + (i0+1)][c_start] = -1
        col += 1

    # Block 1: i=0,1 j=0,1 k=0,1
    add_strassen(0, 0, 0, col)
    # Block 2: i=0,1 j=2,3 k=0,1
    add_strassen(0, 2, 0, col)
    # Block 3: i=0,1 j=0,1 k=2,3
    add_strassen(0, 0, 2, col)
    # Block 4: i=0,1 j=2,3 k=2,3
    add_strassen(0, 2, 2, col)
    
    # Remainder 1: i=0,1 j=0..3 k=4 (Rank 1)
    # Sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
    # Set r = col
    # U[i,j,col] = 1 for i in {0,1}, j in {0,1,2,3}
    # V[j,4,col] = 1 for j in {0,1,2,3}
    # W[4,i,col] = 1 for i in {0,1}
    for i in [0, 1]:
        for j in range(4):
            U[i*m + j][col] = 1
    for j in range(4):
        V[j*p + 4][col] = 1
    for i in [0, 1]:
        W[4*n + i][col] = 1
    col += 1
    
    # Remainder 2: i=2 j=0..3 k=0..4 (Rank 1)
    # U[2,j,col] = 1 for j in {0..3}
    # V[j,k,col] = 1 for j in {0..3}, k in {0..4}
    # W[k,2,col] = 1 for k in {0..4}
    for j in range(4):
        U[2*m + j][col] = 1
    for j in range(4):
        for k in range(5):
            V[j*p + k][col] = 1
    for k in range(5):
        W[k*n + 2][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
