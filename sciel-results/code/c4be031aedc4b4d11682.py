"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _add_strassen_2x2(U, V, W, col, a_r0, a_r1, b_c0, b_c1, c_r0, c_r1, c_c0, c_c1):
    """
    Add Strassen decomposition for computing C[c_r0:c_r1+1, c_c0:c_c1+1] 
    = A[a_r0:a_r1+1, b_c0:b_c1+1] * B[b_c0:b_c1+1, c_c0:c_c1+1]
    
    A is n x m, B is m x p, C is n x p
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    W index: k*n + i (trace convention W[k*n+i, r])
    
    a_r0, a_r1: rows of A (and C)
    b_c0, b_c1: cols of A (and rows of B)  
    c_c0, c_c1: cols of B (and C)
    """
    # Strassen's 7 products for 2x2 * 2x2
    # A_block = [[A[a_r0,b_c0], A[a_r0,b_c1]], [A[a_r1,b_c0], A[a_r1,b_c1]]]
    # B_block = [[B[b_c0,c_c0], B[b_c0,c_c1]], [B[b_c1,c_c0], B[b_c1,c_c1]]]
    # C_block = [[C[a_r0,c_c0], C[a_r0,c_c1]], [C[a_r1,c_c0], C[a_r1,c_c1]]]
    
    # M1 = (A[a_r0,b_c0] + A[a_r1,b_c1]) * (B[b_c0,c_c0] + B[b_c1,c_c1])
    U[a_r0 * 4 + b_c0][col] += 1
    U[a_r1 * 4 + b_c1][col] += 1
    V[b_c0 * 4 + c_c0][col] += 1
    V[b_c1 * 4 + c_c1][col] += 1
    # C[a_r0,c_c0] += M1, C[a_r1,c_c1] += M1
    W[c_c0 * 3 + a_r0][col] += 1
    W[c_c1 * 3 + a_r1][col] += 1
    col += 1
    
    # M2 = (A[a_r1,b_c0] + A[a_r1,b_c1]) * B[b_c0,c_c0]
    U[a_r1 * 4 + b_c0][col] += 1
    U[a_r1 * 4 + b_c1][col] += 1
    V[b_c0 * 4 + c_c0][col] += 1
    # C[a_r1,c_c0] += M2, C[a_r1,c_c1] -= M2
    W[c_c0 * 3 + a_r1][col] += 1
    W[c_c1 * 3 + a_r1][col] -= 1
    col += 1
    
    # M3 = A[a_r0,b_c0] * (B[b_c0,c_c1] - B[b_c1,c_c1])
    U[a_r0 * 4 + b_c0][col] += 1
    V[b_c0 * 4 + c_c1][col] += 1
    V[b_c1 * 4 + c_c1][col] -= 1
    # C[a_r0,c_c1] += M3, C[a_r1,c_c1] += M3
    W[c_c1 * 3 + a_r0][col] += 1
    W[c_c1 * 3 + a_r1][col] += 1
    col += 1
    
    # M4 = A[a_r1,b_c1] * (B[b_c1,c_c0] - B[b_c0,c_c0])
    U[a_r1 * 4 + b_c1][col] += 1
    V[b_c1 * 4 + c_c0][col] += 1
    V[b_c0 * 4 + c_c0][col] -= 1
    # C[a_r0,c_c0] += M4, C[a_r1,c_c0] += M4
    W[c_c0 * 3 + a_r0][col] += 1
    W[c_c0 * 3 + a_r1][col] += 1
    col += 1
    
    # M5 = (A[a_r0,b_c0] + A[a_r0,b_c1]) * B[b_c1,c_c1]
    U[a_r0 * 4 + b_c0][col] += 1
    U[a_r0 * 4 + b_c1][col] += 1
    V[b_c1 * 4 + c_c1][col] += 1
    # C[a_r0,c_c0] -= M5, C[a_r1,c_c1] += M5
    W[c_c0 * 3 + a_r0][col] -= 1
    W[c_c1 * 3 + a_r1][col] += 1
    col += 1
    
    # M6 = (A[a_r1,b_c0] - A[a_r0,b_c0]) * (B[b_c0,c_c0] + B[b_c0,c_c1])
    U[a_r1 * 4 + b_c0][col] += 1
    U[a_r0 * 4 + b_c0][col] -= 1
    V[b_c0 * 4 + c_c0][col] += 1
    V[b_c0 * 4 + c_c1][col] += 1
    # C[a_r1,c_c1] += M6
    W[c_c1 * 3 + a_r1][col] += 1
    col += 1
    
    # M7 = (A[a_r0,b_c1] - A[a_r1,b_c1]) * (B[b_c1,c_c0] + B[b_c1,c_c1])
    U[a_r0 * 4 + b_c1][col] += 1
    U[a_r1 * 4 + b_c1][col] -= 1
    V[b_c1 * 4 + c_c0][col] += 1
    V[b_c1 * 4 + c_c1][col] += 1
    # C[a_r0,c_c0] += M7
    W[c_c0 * 3 + a_r0][col] += 1
    col += 1
    
    return col

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    t0 = time.time()
    deadline = t0 + 100  # 100s safety margin within 120s limit
    
    # Strategy: Split the 3x4 matrix into a 2x4 block and a 1x4 block.
    # 2x4 x 4x4: Use Strassen on four 2x2 sub-blocks -> rank 4*7 = 28
    # 1x4 x 4x4: Schoolbook -> rank 16
    # Total rank: 44
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # (12, 44)
    V = [[0] * rank for _ in range(m * p)]  # (16, 44)
    W = [[0] * rank for _ in range(p * n)]  # (12, 44)
    
    col = 0
    
    # Part 1: 2x4 x 4x4 using Strassen (rows 0,1 of A)
    # Split into four 2x2 * 2x2 blocks:
    for j in range(2):  # A col pair / B row pair
        for i in range(2):  # B col pair / C col pair
            a_r0, a_r1 = 0, 1
            b_c0, b_c1 = 2*j, 2*j+1
            c_c0, c_c1 = 2*i, 2*i+1
            c_r0, c_r1 = 0, 1
            col = _add_strassen_2x2(U, V, W, col, a_r0, a_r1, b_c0, b_c1, c_r0, c_r1, c_c0, c_c1)
    
    # Part 2: 1x4 x 4x4 schoolbook (row 2 of A)
    for j in range(4):
        for k in range(4):
            U[2 * 4 + j][col] = 1
            V[j * 4 + k][col] = 1
            W[k * 3 + 2][col] = 1
            col += 1
    
    assert col == rank, f"Expected rank {rank} but used {col} columns"
    
    # Verify correctness using numpy for speed
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # T[i*m+j, j*p+k, k*n+i] should all be 1
    # Check: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
    errors = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                if abs(val - 1.0) > 1e-10:
                    errors += 1
    
    if errors == 0:
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank_sb = n * m * p
    U = [[0] * rank_sb for _ in range(n * m)]
    V = [[0] * rank_sb for _ in range(m * p)]
    W = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
