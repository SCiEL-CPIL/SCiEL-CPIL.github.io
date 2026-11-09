import numpy as np
import time
import random

DIMENSIONS = [3, 4, 7]

def _verify(U, V, W, n, m, p, tol=1e-10):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def _strassen_2x2():
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # M1 = A11(B22-B12)
    U[0][0] = 1; V[3][0] = 1; V[2][0] = -1; W[2][0] = 1; W[3][0] = 1
    # M2 = (A11+A12)B12
    U[0][1] = 1; U[1][1] = 1; V[2][1] = 1; W[3][1] = 1
    # M3 = (A21+A22)B11
    U[2][2] = 1; U[3][2] = 1; V[0][2] = 1; W[2][2] = 1
    # M4 = A22(B21-B11)
    U[3][3] = 1; V[1][3] = 1; V[0][3] = -1; W[2][3] = 1; W[3][3] = 1
    # M5 = A11(B11+B22)
    U[0][4] = 1; V[0][4] = 1; V[3][4] = 1; W[2][4] = 1
    # M6 = (A21-A11)(B11+B12)
    U[2][5] = 1; U[0][5] = -1; V[0][5] = 1; V[2][5] = 1; W[3][5] = 1
    # M7 = (A12-A22)(B21+B22)
    U[1][6] = 1; U[3][6] = -1; V[1][6] = 1; V[3][6] = 1; W[2][6] = 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_result = {"U": U, "V": V, "W": W}
    best_rank = n * m * p
    
    # Greedy packing of 2x2x2 Strassen blocks
    # We represent the tensor as a set of "active" indices
    # If an index is covered by a Strassen block, it's not active for schoolbook
    # But we need to construct the actual matrices.
    
    # This approach is complex to implement dynamically.
    # Instead, we will explicitly construct the solution for the known 78 rank case.
    # And then try to improve via local search if time permits.
    
    # For now, return the schoolbook to be safe, but we can do better.
    # Let's implement the 78 rank construction explicitly.
    
    # Indices for 6 disjoint 2x2x2 blocks in 3x4x7
    # i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5,6}
    # Blocks:
    # 1: i=0,1; j=0,1; k=0,1
    # 2: i=0,1; j=0,1; k=2,3
    # 3: i=0,1; j=0,1; k=4,5
    # 4: i=0,1; j=2,3; k=0,1
    # 5: i=0,1; j=2,3; k=2,3
    # 6: i=0,1; j=2,3; k=4,5
    
    # All these blocks use i=0,1.
    # i=2 is left for schoolbook.
    # j=3 is used in blocks 4,5,6.
    # k=6 is left for schoolbook.
    
    # We will build the matrices.
    # Rank = 6 * 7 (Strassen) + (84 - 6*8) (Schoolbook)
    # Rank = 42 + 36 = 78.
    
    U_new = [[0] * 78 for _ in range(n * m)]
    V_new = [[0] * 78 for _ in range(m * p)]
    W_new = [[0] * 78 for _ in range(p * n)]
    
    col_offset = 0
    blocks = [
        (0, 0, 0), # i0, j0, k0
        (0, 0, 2), # i0, j0, k2
        (0, 0, 4), # i0, j0, k4
        (0, 2, 0), # i0, j2, k0
        (0, 2, 2), # i0, j2, k2
        (0, 2, 4), # i0, j2, k4
    ]
    
    for i0, j0, k0 in blocks:
        sU, sV, sW = _strassen_2x2()
        # Map local indices (0,1) to global (i0, i0+1) etc.
        for i_local in range(2):
            for j_local in range(2):
                for c in range(7):
                    U_new[(i0 + i_local) * m + (j0 + j_local)][col_offset + c] = sU[i_local * 2 + j_local][c]
        for j_local in range(2):
            for k_local in range(2):
                for c in range(7):
                    V_new[(j0 + j_local) * p + (k0 + k_local)][col_offset + c] = sV[j_local * 2 + k_local][c]
        for k_local in range(2):
            for i_local in range(2):
                for c in range(7):
                    W_new[(k0 + k_local) * n + (i0 + i_local)][col_offset + c] = sW[k_local * 2 + i_local][c]
        col_offset += 7
    
    # Fill remaining with schoolbook
    # We need to track which (i,j,k) are already covered.
    covered = set()
    for i0, j0, k0 in blocks:
        for i in range(i0, i0+2):
            for j in range(j0, j0+2):
                for k in range(k0, k0+2):
                    covered.add((i,j,k))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i,j,k) not in covered:
                    U_new[i * m + j][col_offset] = 1
                    V_new[j * p + k][col_offset] = 1
                    W_new[k * n + i][col_offset] = 1
                    col_offset += 1
    
    # Verify
    if _verify(U_new, V_new, W_new, n, m, p):
        best_result = {"U": U_new, "V": V_new, "W": W_new}
        best_rank = 78
    
    return best_result
