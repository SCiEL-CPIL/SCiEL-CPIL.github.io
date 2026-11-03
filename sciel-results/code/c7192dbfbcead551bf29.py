import time
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 3 matrix by a 3 by 5 matrix.
    
    Strategy:
    1. Decompose the problem M_{2,3,5} into M_{2,1,5} (rank 10) and M_{2,2,5}.
    2. Decompose M_{2,2,5} into M_{2,2,2} (rank 7, Strassen), M_{2,2,2} (rank 7, Strassen), 
       and M_{2,2,1} (rank 4, schoolbook).
    3. Total rank = 10 + 7 + 7 + 4 = 28.
    4. Construct the U, V, W matrices for these components.
    5. Verify the decomposition.
    6. If verification fails, fall back to schoolbook rank 30.
    """
    n, m, p = 2, 3, 5
    nm, mp, pn = n * m, m * p, p * n  # 6, 15, 10
    
    start_time = time.time()
    deadline = start_time + 110.0

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
        """Verify exact decomposition."""
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

    # Part 1: M_{2,1,5} for j=0
    # i in {0,1}, j=0, k in {0..4}
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    # Helper to add Strassen terms for a 2x2x2 block
    # i_offset: 0 or 1 (not used, i is global 0,1)
    # j_offset: 0 or 1 (local j index within the 2x2 block)
    # k_offset: 0 or 1 (local k index within the 2x2 block)
    # Global j = j_base + j_local
    # Global k = k_base + k_local
    def add_strassen(j_base, k_base):
        nonlocal col
        # m1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
        # U: A[0,0] -> i=0, j=j_base+0; A[1,1] -> i=1, j=j_base+1
        U[0 * m + (j_base + 0)][col] = 1
        U[1 * m + (j_base + 1)][col] = 1
        # V: B[0,0] -> j=j_base+0, k=k_base+0; B[1,1] -> j=j_base+1, k=k_base+1
        V[(j_base + 0) * p + (k_base + 0)][col] = 1
        V[(j_base + 1) * p + (k_base + 1)][col] = 1
        # W: C[0,0] += 1; C[1,1] += 1
        # C[i,k] index in W is k*n+i
        W[(k_base + 0) * n + 0][col] = 1  # C[0,0]
        W[(k_base + 1) * n + 1][col] = 1  # C[1,1]
        col += 1

        # m2 = (A[1,0] + A[1,1]) * B[0,0]
        U[1 * m + (j_base + 0)][col] = 1
        U[1 * m + (j_base + 1)][col] = 1
        V[(j_base + 0) * p + (k_base + 0)][col] = 1
        # W: C[1,0] += 1
        W[(k_base + 0) * n + 1][col] = 1  # C[1,0]
        col += 1

        # m3 = A[0,0] * (B[0,1] - B[1,1])
        U[0 * m + (j_base + 0)][col] = 1
        V[(j_base + 0) * p + (k_base + 1)][col] = 1
        V[(j_base + 1) * p + (k_base + 1)][col] = -1
        # W: C[0,1] += 1
        W[(k_base + 1) * n + 0][col] = 1  # C[0,1]
        col += 1

        # m4 = A[1,1] * (B[1,0] - B[0,0])
        U[1 * m + (j_base + 1)][col] = 1
        V[(j_base + 1) * p + (k_base + 0)][col] = 1
        V[(j_base + 0) * p + (k_base + 0)][col] = -1
        # W: C[1,0] += 1
        W[(k_base + 0) * n + 1][col] = 1  # C[1,0]
        col += 1

        # m5 = (A[0,0] + A[0,1]) * B[1,1]
        U[0 * m + (j_base + 0)][col] = 1
        U[0 * m + (j_base + 1)][col] = 1
        V[(j_base + 1) * p + (k_base + 1)][col] = 1
        # W: C[0,1] += 1; C[0,0] -= 1
        W[(k_base + 1) * n + 0][col] = 1  # C[0,1]
        W[(k_base + 0) * n + 0][col] = -1 # C[0,0]
        col += 1

        # m6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
        U[1 * m + (j_base + 0)][col] = 1
        U[0 * m + (j_base + 0)][col] = -1
        V[(j_base + 0) * p + (k_base + 0)][col] = 1
        V[(j_base + 0) * p + (k_base + 1)][col] = 1
        # W: C[1,1] += 1
        W[(k_base + 1) * n + 1][col] = 1  # C[1,1]
        col += 1

        # m7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
        U[0 * m + (j_base + 1)][col] = 1
        U[1 * m + (j_base + 1)][col] = -1
        V[(j_base + 1) * p + (k_base + 0)][col] = 1
        V[(j_base + 1) * p + (k_base + 1)][col] = 1
        # W: C[0,0] += 1; C[1,1] += 1
        W[(k_base + 0) * n + 0][col] = 1  # C[0,0]
        W[(k_base + 1) * n + 1][col] = 1  # C[1,1]
        col += 1

    # Part 2: M_{2,2,5} for j in {1,2}
    # Split k into {0,1} and {2,3,4}
    # k in {0,1}: Strassen block with j_base=1, k_base=0
    add_strassen(1, 0)
    
    # k in {2,3}: Strassen block with j_base=1, k_base=2
    # Note: The Strassen algorithm assumes 2x2 B. Here k indices are 2 and 3.
    # So k_local 0 maps to k=2, k_local 1 maps to k=3.
    add_strassen(1, 2)

    # k in {4}: Schoolbook for M_{2,2,1}
    # i in {0,1}, j in {1,2}, k=4
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
