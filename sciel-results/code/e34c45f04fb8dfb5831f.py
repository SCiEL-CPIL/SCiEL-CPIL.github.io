"""Matrix multiplication 4x4 * 4x6 tensor decomposition.

Strategy: Split B into three 4x2 column blocks.
For each block, compute A * B_block (4x4 * 4x2 = 4x2) using:
  - Split A into 2x2 blocks, B_block into 2x2 blocks
  - 4 sub-products of 2x2 * 2x2, each using Strassen (rank 7)
  - Total per block: 4 * 7 = 28
Total rank: 3 * 28 = 84

This is better than the previous 88-rank approach.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = 4, 4, 6  # A is n x m, B is m x p, C is n x p
    
    U_rows = n * m  # 16
    V_rows = m * p  # 24
    W_rows = p * n  # 24
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    def strassen_2x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        """Add 7 rank-1 terms for Strassen 2x2*2x2.
        A[a_r:a_r+2, a_c:a_c+2] * B[b_r:b_r+2, b_c:b_c+2] -> C[c_r:c_r+2, c_c:c_c+2]
        """
        def ai(i,j): return (a_r+i)*m + (a_c+j)
        def bi(i,j): return (b_r+i)*p + (b_c+j)
        def ci(i,j): return (c_c+j)*n + (c_r+i)
        
        def term(a_entries, b_entries, c_entries):
            u = [0]*U_rows; v = [0]*V_rows; w = [0]*W_rows
            for idx, val in a_entries:
                u[idx] = val
            for idx, val in b_entries:
                v[idx] = val
            for idx, val in c_entries:
                w[idx] = val
            add_term(u, v, w)
        
        # M1 = (A11+A22)*(B11+B22)
        term([(ai(0,0),1),(ai(1,1),1)], [(bi(0,0),1),(bi(1,1),1)], [(ci(0,0),1),(ci(1,1),1)])
        # M2 = (A21+A22)*B11
        term([(ai(1,0),1),(ai(1,1),1)], [(bi(0,0),1)], [(ci(1,0),1),(ci(1,1),-1)])
        # M3 = A11*(B12-B22)
        term([(ai(0,0),1)], [(bi(0,1),1),(bi(1,1),-1)], [(ci(0,1),1),(ci(1,1),1)])
        # M4 = A22*(B21-B11)
        term([(ai(1,1),1)], [(bi(1,0),1),(bi(0,0),-1)], [(ci(0,0),1),(ci(1,0),1)])
        # M5 = (A11+A12)*B22
        term([(ai(0,0),1),(ai(0,1),1)], [(bi(1,1),1)], [(ci(0,0),-1),(ci(0,1),1)])
        # M6 = (A21-A11)*(B11+B12)
        term([(ai(1,0),1),(ai(0,0),-1)], [(bi(0,0),1),(bi(0,1),1)], [(ci(1,1),1)])
        # M7 = (A12-A22)*(B21+B22)
        term([(ai(0,1),1),(ai(1,1),-1)], [(bi(1,0),1),(bi(1,1),1)], [(ci(0,0),1)])
    
    # For each of the 3 column blocks of B (each 4x2):
    for block in range(3):
        b_col_offset = block * 2
        c_col_offset = block * 2
        
        # A11 * B11 -> C11
        strassen_2x2_2x2(0, 0, 0, b_col_offset, 0, c_col_offset)
        # A12 * B21 -> C12
        strassen_2x2_2x2(0, 2, 2, b_col_offset, 0, c_col_offset)
        # A21 * B11 -> C21
        strassen_2x2_2x2(2, 0, 0, b_col_offset, 2, c_col_offset)
        # A22 * B21 -> C22
        strassen_2x2_2x2(2, 2, 2, b_col_offset, 2, c_col_offset)
    
    rank = len(U_cols)
    
    # Build output matrices
    U = [[U_cols[r][i] for r in range(rank)] for i in range(U_rows)]
    V = [[V_cols[r][i] for r in range(rank)] for i in range(V_rows)]
    W = [[W_cols[r][i] for r in range(rank)] for i in range(W_rows)]
    
    return {"U": U, "V": V, "W": W}

# Verification (not part of construct, but useful for debugging)
if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    
    n, m, p = 4, 4, 6
    U_rows, V_rows, W_rows = n*m, m*p, p*n
    rank = U.shape[1]
    
    # Build the multiplication tensor
    # T[i,j,k] = 1 if j == k (i.e., C[i,k] = sum_j A[i,j]*B[j,k])
    # In the decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    
    # Verify with random matrices
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_decomposed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for j in range(m):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                C_decomposed[i, k] += s
    
    error = np.max(np.abs(C_decomposed - C_expected))
    print(f"Rank: {rank}")
    print(f"Max error: {error}")
    if error < 1e-10:
        print("VALID: decomposition is correct")
    else:
        print("INVALID: decomposition has errors")
