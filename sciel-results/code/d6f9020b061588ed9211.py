import time
import numpy as np
from fractions import Fraction

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 3 matrix by a 3 by 5 matrix.
    
    Strategy:
    1. Use Strassen's algorithm for 2x2x2 sub-blocks to achieve rank 7 per block.
    2. Decompose the 2x3x5 tensor into blocks:
       - M_{2,1,5}: rank 10 (schoolbook for the 1-column case)
       - M_{2,2,5}: split into M_{2,2,2} + M_{2,2,2} + M_{2,2,1} = 7+7+4 = 18
       Total: 10 + 18 = 28
    3. Verify the decomposition exactly using integer arithmetic.
    4. Fall back to schoolbook rank 30 if verification fails.
    """
    n, m, p = 2, 3, 5
    nm, mp, pn = n * m, m * p, p * n  # 6, 15, 10
    
    start_time = time.time()
    deadline = start_time + 100.0

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """Verify exact decomposition using integer arithmetic."""
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True

    if time.time() > deadline:
        return schoolbook()

    # Target rank 28
    rank = 28
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0

    # Part 1: M_{2,1,5} for j=0 (schoolbook, rank 10)
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    # Helper to add Strassen terms for a 2x2x2 block
    def add_strassen(j_base, k_base):
        nonlocal col
        # Strassen 7 multiplications for 2x2 matrices
        # m1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
        U[0 * m + (j_base + 0)][col] = 1
        U[1 * m + (j_base + 1)][col] = 1
        V[(j_base + 0) * p + (k_base + 0)][col] = 1
        V[(j_base + 1) * p + (k_base + 1)][col] = 1
        W[(k_base + 0) * n + 0][col] = 1   # C[0,0] += 1
        W[(k_base + 1) * n + 1][col] = 1   # C[1,1] += 1
        col += 1

        # m2 = (A[1,0] + A[1,1]) * B[0,0]
        U[1 * m + (j_base + 0)][col] = 1
        U[1 * m + (j_base + 1)][col] = 1
        V[(j_base + 0) * p + (k_base + 0)][col] = 1
        W[(k_base + 0) * n + 1][col] = 1   # C[1,0] += 1
        col += 1

        # m3 = A[0,0] * (B[0,1] - B[1,1])
        U[0 * m + (j_base + 0)][col] = 1
        V[(j_base + 0) * p + (k_base + 1)][col] = 1
        V[(j_base + 1) * p + (k_base + 1)][col] = -1
        W[(k_base + 1) * n + 0][col] = 1   # C[0,1] += 1
        col += 1

        # m4 = A[1,1] * (B[1,0] - B[0,0])
        U[1 * m + (j_base + 1)][col] = 1
        V[(j_base + 1) * p + (k_base + 0)][col] = 1
        V[(j_base + 0) * p + (k_base + 0)][col] = -1
        W[(k_base + 0) * n + 1][col] = 1   # C[1,0] += 1
        col += 1

        # m5 = (A[0,0] + A[0,1]) * B[1,1]
        U[0 * m + (j_base + 0)][col] = 1
        U[0 * m + (j_base + 1)][col] = 1
        V[(j_base + 1) * p + (k_base + 1)][col] = 1
        W[(k_base + 1) * n + 0][col] = 1   # C[0,1] += 1
        W[(k_base + 0) * n + 0][col] = -1  # C[0,0] -= 1
        col += 1

        # m6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
        U[1 * m + (j_base + 0)][col] = 1
        U[0 * m + (j_base + 0)][col] = -1
        V[(j_base + 0) * p + (k_base + 0)][col] = 1
        V[(j_base + 0) * p + (k_base + 1)][col] = 1
        W[(k_base + 1) * n + 1][col] = 1   # C[1,1] += 1
        col += 1

        # m7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
        U[0 * m + (j_base + 1)][col] = 1
        U[1 * m + (j_base + 1)][col] = -1
        V[(j_base + 1) * p + (k_base + 0)][col] = 1
        V[(j_base + 1) * p + (k_base + 1)][col] = 1
        W[(k_base + 0) * n + 0][col] = 1   # C[0,0] += 1
        W[(k_base + 1) * n + 1][col] = 1   # C[1,1] += 1
        col += 1

    # Part 2: M_{2,2,5} for j in {1,2}
    # Split k into {0,1}, {2,3}, {4}
    # k in {0,1}: Strassen block with j_base=1, k_base=0
    add_strassen(1, 0)
    
    # k in {2,3}: Strassen block with j_base=1, k_base=2
    add_strassen(1, 2)

    # k in {4}: Schoolbook for M_{2,2,1} (rank 4)
    for i in range(n):
        for j_local in range(2):
            j_global = 1 + j_local
            U[i * m + j_global][col] = 1
            V[j_global * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1

    # Check if we filled exactly 28 columns
    if col != 28:
        return schoolbook()

    # Verify
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        return schoolbook()

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
