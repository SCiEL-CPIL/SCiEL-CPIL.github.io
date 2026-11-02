import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Strategy: Use multiple non-overlapping 2x2x2 Strassen blocks
    # The 3x5x5 tensor can be decomposed into:
    # - For i in {0,1}, j in {0,1}, k in {0,1}: 2x2x2 block -> Strassen (7 terms instead of 8)
    # - For i in {0,1}, j in {2,3}, k in {0,1}: another 2x2x2 block -> Strassen
    # - For i in {0,1}, j in {0,1}, k in {2,3}: another 2x2x2 block -> Strassen
    # - All remaining terms: schoolbook (1 term each)
    
    # Let's identify non-overlapping 2x2x2 blocks:
    # Block 1: i∈{0,1}, j∈{0,1}, k∈{0,1}
    # Block 2: i∈{0,1}, j∈{2,3}, k∈{0,1}
    # Block 3: i∈{0,1}, j∈{0,1}, k∈{2,3}
    # Block 4: i∈{0,1}, j∈{2,3}, k∈{2,3}
    
    # These 4 blocks are non-overlapping and each saves 1 rank term (8 schoolbook -> 7 Strassen)
    # Total saved: 4 rank terms
    # Schoolbook rank: 3*5*5 = 75
    # With 4 Strassen blocks: 75 - 4 = 71
    
    # Let's verify the blocks don't overlap:
    # Block 1: i=0,1; j=0,1; k=0,1
    # Block 2: i=0,1; j=2,3; k=0,1
    # Block 3: i=0,1; j=0,1; k=2,3
    # Block 4: i=0,1; j=2,3; k=2,3
    # Yes, they are disjoint in the (i,j,k) space.
    
    # However, we need to be careful: Strassen's algorithm works on 2x2 matrices.
    # For each block, we're replacing the 8 scalar multiplications with 7.
    
    # Let's construct the decomposition:
    # Total rank = 4 * 7 (Strassen blocks) + (75 - 4*8) schoolbook terms
    # = 28 + (75 - 32) = 28 + 43 = 71
    
    total_rank = 4 * 7 + (n * m * p - 4 * 8)
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Define the 4 Strassen blocks
    # Each block is defined by (i0, i1, j0, j1, k0, k1)
    blocks = [
        (0, 1, 0, 1, 0, 1),  # Block 1
        (0, 1, 2, 3, 0, 1),  # Block 2
        (0, 1, 0, 1, 2, 3),  # Block 3
        (0, 1, 2, 3, 2, 3),  # Block 4
    ]
    
    # Strassen formulas for a 2x2 block:
    # Let A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A * B
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b01 - b11)
    # M4 = a11(b10 - b00)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    # C00 = M1 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    for (i0, i1, j0, j1, k0, k1) in blocks:
        # Local indices: i_local in {0,1}, j_local in {0,1}, k_local in {0,1}
        # Global indices: i = i0 + i_local, j = j0 + j_local, k = k0 + k_local
        
        # M1 = (a00 + a11)(b00 + b11)
        c = col
        U[(i0)*m + j0][c] += 1
        U[(i1)*m + j1][c] += 1
        V[j0*p + k0][c] += 1
        V[j1*p + k1][c] += 1
        W[k0*n + i0][c] += 1   # C00
        W[k1*n + i1][c] += 1   # C11
        col += 1
        
        # M2 = (a10 + a11)b00
        c = col
        U[(i1)*m + j0][c] += 1
        U[(i1)*m + j1][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + i1][c] += 1   # C10
        W[k1*n + i1][c] += -1  # C11
        col += 1
        
        # M3 = a00(b01 - b11)
        c = col
        U[(i0)*m + j0][c] += 1
        V[j0*p + k1][c] += 1
        V[j1*p + k1][c] += -1
        W[k1*n + i0][c] += 1   # C01
        W[k1*n + i1][c] += 1   # C11
        col += 1
        
        # M4 = a11(b10 - b00)
        c = col
        U[(i1)*m + j1][c] += 1
        V[j1*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1   # C00
        W[k0*n + i1][c] += 1   # C10
        col += 1
        
        # M5 = (a00 + a01)b11
        c = col
        U[(i0)*m + j0][c] += 1
        U[(i0)*m + j1][c] += 1
        V[j1*p + k1][c] += 1
        W[k0*n + i0][c] += -1  # C00
        W[k1*n + i0][c] += 1   # C01
        col += 1
        
        # M6 = (a10 - a00)(b00 + b01)
        c = col
        U[(i1)*m + j0][c] += 1
        U[(i0)*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + k1][c] += 1
        W[k1*n + i1][c] += 1   # C11
        col += 1
        
        # M7 = (a01 - a11)(b10 + b11)
        c = col
        U[(i0)*m + j1][c] += 1
        U[(i1)*m + j1][c] += -1
        V[j1*p + k0][c] += 1
        V[j1*p + k1][c] += 1
        W[k0*n + i0][c] += 1   # C00
        col += 1
    
    # Schoolbook for the rest
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this (i,j,k) is in any of the Strassen blocks
                in_block = False
                for (i0, i1, j0, j1, k0, k1) in blocks:
                    if i0 <= i <= i1 and j0 <= j <= j1 and k0 <= k <= k1:
                        in_block = True
                        break
                if in_block:
                    continue
                U[i*m + j][col] += 1
                V[j*p + k][col] += 1
                W[k*n + i][col] += 1
                col += 1
    
    # Verify we used all columns
    assert col == total_rank, f"Expected {total_rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
    
    # Verify correctness by checking the tensor decomposition
    # T_ijk = sum_c U_{i*m+j, c} * V_{j*p+k, c} * W_{k*n+i, c}
    # should equal 1 for all (i,j,k) that correspond to multiplication terms
    # Actually, for matrix multiplication, the tensor is:
    # T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # which is always the case, so T[i,j,k] = 1 for all i,j,k
    
    # Let's verify a few entries
    T_reconstructed = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_reconstructed[i,j,k] = np.dot(U[i*m+j], V[j*p+k]) @ W[k*n+i]
                # Wait, this isn't right. The tensor decomposition is:
                # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # Let me recompute
    
    T_reconstructed = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_reconstructed[i,j,k] = 0.0
                for c in range(U.shape[1]):
                    T_reconstructed[i,j,k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # The expected tensor is all 1s
    T_expected = np.ones((n, m, p))
    max_err = np.max(np.abs(T_reconstructed - T_expected))
    print(f"Max error: {max_err}")
    
    if max_err < 1e-10:
        print("Decomposition is correct!")
    else:
        print("Decomposition has errors!")
        # Print some entries with errors
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(T_reconstructed[i,j,k] - 1.0) > 1e-10:
                        print(f"  T[{i},{j},{k}] = {T_reconstructed[i,j,k]} (expected 1.0)")
