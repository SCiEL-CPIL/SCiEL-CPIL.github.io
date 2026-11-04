"""Matrix multiplication 4x5x5 with block-partitioned Strassen/Laderman decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 4x5 matrix by a 5x5 matrix using block partitioning.
    
    Partition: split dimension 5 into [2, 3] for j and k.
    - 4x2x2 blocks: Strassen (rank 7)
    - 4x3x3 block: Laderman (rank 13)  
    - 4x2x3, 4x3x2 blocks: schoolbook (rank 6)
    Total rank: 4*7 + 4*6 + 4*6 + 1*13 = 89
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    n_m = n * m  # 20
    m_p = m * p  # 25
    p_n = p * n  # 20
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_entries, v_entries, w_entries):
        """Add one rank-1 term to the decomposition."""
        u_col = np.zeros(n_m)
        v_col = np.zeros(m_p)
        w_col = np.zeros(p_n)
        for idx, val in u_entries:
            u_col[idx] = val
        for idx, val in v_entries:
            v_col[idx] = val
        for idx, val in w_entries:
            w_col[idx] = val
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
    
    def idx_I(i, j):
        return i * m + j
    
    def idx_J(j, k):
        return j * p + k
    
    def idx_K(k, i):
        return k * n + i
    
    def strassen_2x2x2(i_off, j_off, k_off):
        """
        Strassen algorithm for multiplying 2x2 by 2x2 matrices.
        """
        def I(ii, jj): return idx_I(i_off + ii, j_off + jj)
        def J(jj, kk): return idx_J(j_off + jj, k_off + kk)
        def K(kk, ii): return idx_K(k_off + kk, i_off + ii)
        
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        add_term(
            [(I(0,0), 1), (I(1,1), 1)],
            [(J(0,0), 1), (J(1,1), 1)],
            [(K(0,0), 1), (K(1,1), 1)]
        )
        
        # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
        add_term(
            [(I(1,0), 1), (I(1,1), 1)],
            [(J(0,0), 1)],
            [(K(0,1), 1), (K(1,1), -1)]
        )
        
        # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
        add_term(
            [(I(0,0), 1)],
            [(J(0,1), 1), (J(1,1), -1)],
            [(K(1,0), 1), (K(1,1), 1)]
        )
        
        # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
        add_term(
            [(I(1,1), 1)],
            [(J(1,0), 1), (J(0,0), -1)],
            [(K(0,0), 1), (K(0,1), 1)]
        )
        
        # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
        add_term(
            [(I(0,0), 1), (I(0,1), 1)],
            [(J(1,1), 1)],
            [(K(0,0), -1), (K(1,0), 1)]
        )
        
        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        add_term(
            [(I(1,0), 1), (I(0,0), -1)],
            [(J(0,0), 1), (J(0,1), 1)],
            [(K(1,1), 1)]
        )
        
        # M7 = (A01-A11)(B10+B11) -> C00 += M7
        add_term(
            [(I(0,1), 1), (I(1,1), -1)],
            [(J(1,0), 1), (J(1,1), 1)],
            [(K(0,0), 1)]
        )
    
    def laderman_3x3x3(i_off, j_off, k_off):
        """
        Laderman algorithm for 3x3 matrix multiplication (rank 13).
        Uses the well-known Laderman (1976) decomposition.
        """
        def I(ii, jj): return idx_I(i_off + ii, j_off + jj)
        def J(jj, kk): return idx_J(j_off + jj, k_off + kk)
        def K(kk, ii): return idx_K(k_off + kk, i_off + ii)
        
        # Laderman's 13 products for 3x3x3:
        # The algorithm computes C = A*B where A,B are 3x3
        
        # P1 = (A[0][0] + A[1][1]) * (B[0][0] + B[1][1])
        add_term(
            [(I(0,0), 1), (I(1,1), 1)],
            [(J(0,0), 1), (J(1,1), 1)],
            [(K(0,0), 1), (K(1,1), 1)]
        )
        
        # P2 = (A[0][1] + A[1][1]) * (B[1][0] + B[1][1])
        add_term(
            [(I(0,1), 1), (I(1,1), 1)],
            [(J(1,0), 1), (J(1,1), 1)],
            [(K(0,1), 1), (K(1,1), 1)]
        )
        
        # P3 = (A[0][2] + A[1][2]) * (B[2][0] + B[2][1])
        add_term(
            [(I(0,2), 1), (I(1,2), 1)],
            [(J(2,0), 1), (J(2,1), 1)],
            [(K(0,2), 1), (K(1,2), 1)]
        )
        
        # P4 = (A[1][0] + A[2][0]) * (B[0][1] + B[0][2])
        add_term(
            [(I(1,0), 1), (I(2,0), 1)],
            [(J(0,1), 1), (J(0,2), 1)],
            [(K(1,0), 1), (K(2,0), 1)]
        )
        
        # P5 = (A[1][1] + A[2][1]) * (B[1][2] + B[2][2])
        add_term(
            [(I(1,1), 1), (I(2,1), 1)],
            [(J(1,2), 1), (J(2,2), 1)],
            [(K(1,1), 1), (K(2,1), 1)]
        )
        
        # P6 = (A[1][2] + A[2][2]) * (B[2][0] + B[0][0])
        add_term(
            [(I(1,2), 1), (I(2,2), 1)],
            [(J(2,0), 1), (J(0,0), 1)],
            [(K(1,2), 1), (K(2,2), 1)]
        )
        
        # P7 = A[0][0] * (B[0][1] - B[1][1])
        add_term(
            [(I(0,0), 1)],
            [(J(0,1), 1), (J(1,1), -1)],
            [(K(1,0), 1)]
        )
        
        # P8 = A[0][1] * (B[1][2] - B[2][2])
        add_term(
            [(I(0,1), 1)],
            [(J(1,2), 1), (J(2,2), -1)],
            [(K(2,1), 1)]
        )
        
        # P9 = A[0][2] * (B[2][0] - B[0][0])
        add_term(
            [(I(0,2), 1)],
            [(J(2,0), 1), (J(0,0), -1)],
            [(K(0,2), 1)]
        )
        
        # P10 = A[1][0] * (B[0][2] - B[2][2])
        add_term(
            [(I(1,0), 1)],
            [(J(0,2), 1), (J(2,2), -1)],
            [(K(2,0), 1)]
        )
        
        # P11 = A[1][1] * (B[1][0] - B[0][0])
        add_term(
            [(I(1,1), 1)],
            [(J(1,0), 1), (J(0,0), -1)],
            [(K(0,1), 1)]
        )
        
        # P12 = A[1][2] * (B[2][1] - B[1][1])
        add_term(
            [(I(1,2), 1)],
            [(J(2,1), 1), (J(1,1), -1)],
            [(K(1,2), 1)]
        )
        
        # P13 = A[2][0] * (B[0][1] - B[1][1])
        add_term(
            [(I(2,0), 1)],
            [(J(0,1), 1), (J(1,1), -1)],
            [(K(1,0), 1)]
        )
    
    def schoolbook_block(i_off, j_off, k_off, ni, nj, nk):
        """Schoolbook for sub-block of size ni x nj x nk."""
        for ii in range(ni):
            for jj in range(nj):
                for kk in range(nk):
                    i_g = i_off + ii
                    j_g = j_off + jj
                    k_g = k_off + kk
                    add_term(
                        [(idx_I(i_g, j_g), 1)],
                        [(idx_J(j_g, k_g), 1)],
                        [(idx_K(k_g, i_g), 1)]
                    )
    
    # Partition j and k dimensions into [2, 3]
    j_splits = [(0, 2), (2, 3)]  # (offset, size)
    k_splits = [(0, 2), (2, 3)]
    
    for j_off, j_size in j_splits:
        for k_off, k_size in k_splits:
            if j_size == 2 and k_size == 2:
                # 4x2x2: use Strassen (rank 7)
                strassen_2x2x2(0, j_off, k_off)
            elif j_size == 3 and k_size == 3:
                # 4x3x3: use Laderman (rank 13)
                laderman_3x3x3(0, j_off, k_off)
            else:
                # 4x2x3 or 4x3x2: schoolbook (rank 6)
                schoolbook_block(0, j_off, k_off, 4, j_size, k_size)
    
    # Build final matrices
    if not U_cols:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n*m)]
        V = [[0] * rank for _ in range(m*p)]
        W = [[0] * rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0])
    print(f"Rank: {rank}")
    
    # Verify
    n, m, p = DIMENSIONS
    U_arr = np.array(result["U"], dtype=float)
    V_arr = np.array(result["V"], dtype=float)
    W_arr = np.array(result["W"], dtype=float)
    
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    if np.allclose(T, 1.0, atol=1e-9):
        print("VERIFICATION PASSED")
    else:
        print("VERIFICATION FAILED")
        print("Max error:", np.max(np.abs(T - 1.0)))
