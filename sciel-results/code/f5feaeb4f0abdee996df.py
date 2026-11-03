"""
Rank-minimal exact tensor decomposition for 3x4 * 4x6 matrix multiplication.

We use a block-structured approach:
  A (3x4) = [[A11 (2x2), A12 (2x2)],
             [A21 (1x2), A22 (1x2)]]
  B (4x6) = [[Btop (2x6)],
             [Bbot (2x6)]]
  
  C = A*B where:
    C rows 0-1: A11*Btop + A12*Bbot  (2x6 result)
    C row 2:    A21*Btop + A22*Bbot  (1x6 result)

For the 2x2 x 6 blocks, we use Strassen on 2x2x2 sub-blocks of B.
For the 1x2 x 6 blocks, we use direct multiplication (rank 12 each).

Rank breakdown:
  2x2x6 via Strassen on 2x2x2: 3 blocks of 2x2x2 with rank 7 each = 21, but wait...
  
Actually, let me reconsider. For 2x2 matrix times 2x6 matrix:
  We can split the 2x6 into three 2x2 blocks.
  Each 2x2 * 2x2 uses Strassen rank 7.
  Total: 3 * 7 = 21? No, that's not right either.
  
Let me think more carefully. The tensor for 2x2 * 2x6 has rank at most 2*2*6 = 24.
We can decompose it as: sum over k=0,1,2 of (2x2 * 2x2) where each 2x2 block of B is a 2x2.
Each 2x2*2x2 via Strassen has rank 7. So 3 * 7 = 21.

For 1x2 * 2x6: rank is at most 1*2*6 = 12, and it's exactly 12 (direct).

So total:
  C rows 0-1: 2x2*2x6 (rank 21) + 2x2*2x6 (rank 21) = 42
  C row 2: 1x2*2x6 (rank 12) + 1x2*2x6 (rank 12) = 24
  Total rank = 42 + 24 = 66

Wait, let me re-examine. The Strassen decomposition for 2x2*2x2:
  M1 = (a11+a22)(b11+b22)
  M2 = (a21+a22)b11
  M3 = a11(b12-b22)
  M4 = a22(b21-b11)
  M5 = (a11+a12)b22
  M6 = (a21-a11)(b11+b12)
  M7 = (a12-a22)(b21+b22)
  
  c11 = M1 + M4 - M5 + M7
  c12 = M3 + M5
  c21 = M2 + M4
  c22 = M1 - M2 + M3 + M6

This gives rank 7 for 2x2*2x2.

For 2x2 * 2x6, we split B into B[:, 0:2], B[:, 2:4], B[:, 4:6], each 2x2.
Each requires rank 7, so total rank 21 for this sub-problem.

For 1x2 * 2x6, it's just dot products. The tensor is:
  C[i,k] = sum_j A[i,j] * B[j,k] for i in {0}, j in {0,1}, k in {0..5}
  This is rank 1*2*6 = 12 (each term A[i,j]*B[j,k] is rank 1).

So total rank = 21 + 21 + 12 + 12 = 66.

Let me implement this.
"""

import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def _strassen_2x2x2():
    """Return U, V, W for 2x2 * 2x2 using Strassen (rank 7).
    
    A is 2x2, B is 2x2, C = A*B is 2x2.
    U: shape (2*2, 7) = (4, 7), indexed by [i*2+j, r]
    V: shape (2*2, 7) = (4, 7), indexed by [j*2+k, r]  
    W: shape (2*2, 7) = (4, 7), indexed by [k*2+i, r]
    
    Strassen:
      M1 = (a00+a11)(b00+b11)
      M2 = (a10+a11)b00
      M3 = a00(b01-b11)
      M4 = a11(b10-b00)
      M5 = (a00+a01)b11
      M6 = (a10-a00)(b00+b01)
      M7 = (a01-a11)(b10+b11)
      
      c00 = M1 + M4 - M5 + M7
      c01 = M3 + M5
      c10 = M2 + M4
      c11 = M1 - M2 + M3 + M6
    """
    # U: for A (2x2), index [i*2+j, r]
    # V: for B (2x2), index [j*2+k, r]
    # W: for C (2x2), index [k*2+i, r]
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # r=0: M1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1  # a00, a11
    V[0, 0] = 1; V[3, 0] = 1  # b00, b11
    W[0, 0] = 1; W[3, 0] = 1  # c00 (k=0,i=0), c11 (k=1,i=1) -> W[k*2+i]
    # c00: k=0, i=0 -> W[0*2+0]=W[0]
    # c11: k=1, i=1 -> W[1*2+1]=W[3]
    
    # r=1: M2 = (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1  # a10, a11
    V[0, 1] = 1  # b00
    W[2, 1] = 1  # c10: k=0, i=1 -> W[0*2+1]=W[1]? 
    # Wait, c10 means row 1, col 0. In C[i,k], i=1, k=0. W[k*n+i] = W[0*2+1] = W[1]
    W[1, 1] = 1
    
    # r=2: M3 = a00(b01-b11)
    U[0, 2] = 1  # a00
    V[1, 2] = 1; V[3, 2] = -1  # b01, b11
    W[1, 2] = 1  # c01: k=1, i=0 -> W[1*2+0]=W[2]
    # c01: row 0, col 1. i=0, k=1. W[k*2+i] = W[1*2+0] = W[2]
    W[2, 2] = 1
    
    # r=3: M4 = a11(b10-b00)
    U[3, 3] = 1  # a11
    V[2, 3] = 1; V[0, 3] = -1  # b10, b00
    W[0, 3] = 1  # c00: W[0]
    W[2, 3] = 1  # c10: W[1]
    
    # r=4: M5 = (a00+a01)b11
    U[0, 4] = 1; U[1, 4] = 1  # a00, a01
    V[3, 4] = 1  # b11
    W[2, 4] = -1  # c01: W[2], with -1
    W[0, 4] = -1  # c00: W[0], with -1
    
    # r=5: M6 = (a10-a00)(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1  # a10, a00
    V[0, 5] = 1; V[1, 5] = 1  # b00, b01
    W[3, 5] = 1  # c11: W[3]
    
    # r=6: M7 = (a01-a11)(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1  # a01, a11
    V[2, 6] = 1; V[3, 6] = 1  # b10, b11
    W[0, 6] = 1  # c00: W[0]
    W[3, 6] = 1  # c11: W[3]
    
    return U, V, W

def _build_2x2_x_2x6(offset_i, offset_j, offset_k, U_list, V_list, W_list):
    """Add a 2x2 * 2x6 decomposition using Strassen on three 2x2 blocks of B.
    
    offset_i: starting row index of A (0 or 1 for global)
    offset_j: starting col index of A / row index of B  
    offset_k: starting col index of B / col index of C
    
    A_local is 2x2, B_local is 2x6 (split into three 2x2), C_local is 2x6.
    """
    n_local = 2
    m_local = 2
    p_local = 6
    
    # For each of the 3 2x2 blocks in B
    for k_block in range(3):
        U_s, V_s, W_s = _strassen_2x2x2()
        # U_s: (4, 7), V_s: (4, 7), W_s: (4, 7)
        # Map to global indices
        for r in range(7):
            # U: A indices [i*2+j, r] -> global [(offset_i+i)*4 + (offset_j+j), r]
            for idx in range(4):
                if U_s[idx, r] != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    g_i = offset_i + i_local
                    g_j = offset_j + j_local
                    g_u = g_i * 4 + g_j
                    U_list.append([0] * 7)  # placeholder
                    V_list.append([0] * 7)
                    W_list.append([0] * 7)
                    col = len(U_list) - 1
                    U_list[col][r] = U_s[idx, r]
                    # V: B indices [j*2+k, r] -> global [(offset_j+j)*6 + (offset_k + k_block*2 + kk), r]
                    for idx_v in range(4):
                        if V_s[idx_v, r] != 0:
                            j_local_v = idx_v // 2
                            k_local_v = idx_v % 2
                            g_j_v = offset_j + j_local_v
                            g_k_v = offset_k + k_block * 2 + k_local_v
                            g_v = g_j_v * 6 + g_k_v
                            V_list[col][r] += V_s[idx_v, r]  # wait, this is wrong
                            
    # This approach is getting complicated. Let me use a different strategy.
    pass

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # We'll build the decomposition as a list of (U_col, V_col, W_col) tuples
    # where each is a sparse vector. Then combine.
    
    # Total rank will be 66: 21 + 21 + 12 + 12
    
    # Let's build U, V, W as lists of columns.
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_strassen_2x2x2(off_i, off_j, off_k):
        """Add Strassen 2x2*2x2 at offset (off_i, off_j, off_k).
        off_i: row offset in A (and C)
        off_j: col offset in A (and row in B)
        off_k: col offset in B (and C)
        """
        U_s, V_s, W_s = _strassen_2x2x2()
        for r in range(7):
            u_col = [0.0] * (n * m)
            v_col = [0.0] * (m * p)
            w_col = [0.0] * (p * n)
            
            for idx in range(4):
                if U_s[idx, r] != 0:
                    i_l = idx // 2
                    j_l = idx % 2
                    gi = off_i + i_l
                    gj = off_j + j_l
                    u_col[gi * m + gj] = U_s[idx, r]
            
            for idx in range(4):
                if V_s[idx, r] != 0:
                    j_l = idx // 2
                    k_l = idx % 2
                    gj = off_j + j_l
                    gk = off_k + k_l
                    v_col[gj * p + gk] = V_s[idx, r]
            
            for idx in range(4):
                if W_s[idx, r] != 0:
                    k_l = idx // 2
                    i_l = idx % 2
                    gk = off_k + k_l
                    gi = off_i + i_l
                    w_col[gk * n + gi] = W_s[idx, r]
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
    
    def add_direct_1x2x2(off_i, off_j, off_k):
        """Add direct 1x2 * 2x2 at offset.
        A is 1x2, B is 2x2, C is 1x2.
        """
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # i is fixed (off_i), j in {off_j, off_j+1}, k in {off_k, off_k+1}
        for j_l in range(2):
            for k_l in range(2):
                u_col = [0.0] * (n * m)
                v_col = [0.0] * (m * p)
                w_col = [0.0] * (p * n)
                
                gi = off_i
                gj = off_j + j_l
                gk = off_k + k_l
                
                u_col[gi * m + gj] = 1.0
                v_col[gj * p + gk] = 1.0
                w_col[gk * n + gi] = 1.0
                
                U_cols.append(u_col)
                V_cols.append(v_col)
                W_cols.append(w_col)
    
    def add_direct_1x2x6(off_i, off_j, off_k):
        """Add direct 1x2 * 2x6 at offset.
        A is 1x2, B is 2x6, C is 1x6.
        """
        for j_l in range(2):
            for k_l in range(6):
                u_col = [0.0] * (n * m)
                v_col = [0.0] * (m * p)
                w_col = [0.0] * (p * n)
                
                gi = off_i
                gj = off_j + j_l
                gk = off_k + k_l
                
                u_col[gi * m + gj] = 1.0
                v_col[gj * p + gk] = 1.0
                w_col[gk * n + gi] = 1.0
                
                U_cols.append(u_col)
                V_cols.append(v_col)
                W_cols.append(w_col)
    
    # Block 1: A11 (rows 0-1, cols 0-1) * Btop (rows 0-1, cols 0-5)
    # 2x2 * 2x6: split Btop into three 2x2 blocks
    for kb in range(3):
        add_strassen_2x2x2(off_i=0, off_j=0, off_k=kb*2)
    
    # Block 2: A12 (rows 0-1, cols 2-3) * Bbot (rows 2-3, cols 0-5)
    # 2x2 * 2x6: split Bbot into three 2x2 blocks
    for kb in range(3):
        add_strassen_2x2x2(off_i=0, off_j=2, off_k=kb*2)
    
    # Block 3: A21 (row 2, cols 0-1) * Btop (rows 0-1, cols 0-5)
    # 1x2 * 2x6
    add_direct_1x2x6(off_i=2, off_j=0, off_k=0)
    
    # Block 4: A22 (row 2, cols 2-3) * Bbot (rows 2-3, cols 0-5)
    # 1x2 * 2x6
    add_direct_1x2x6(off_i=2, off_j=2, off_k=0)
    
    rank = len(U_cols)
    
    U = U_cols
    V = V_cols
    W = W_cols
    
    return {"U": U, "V": V, "W": W}

# Verify
if __name__ == "__main__":
    import time
    t0 = time.time()
    result = construct()
    t1 = time.time()
    print(f"Construct time: {t1-t0:.3f}s")
    print(f"Rank: {len(result['U'])}")
    
    # Verify with random test
    np.random.seed(42)
    A = np.random.rand(3, 4)
    B = np.random.rand(4, 6)
    C_true = A @ B
    
    U = np.array(result['U']).T  # (rank, n*m)
    V = np.array(result['V']).T  # (rank, m*p)
    W = np.array(result['W']).T  # (rank, p*n)
    
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This is a tensor contraction. Let's verify.
    C_test = np.zeros((3, 6))
    for r in range(len(result['U'])):
        u = np.array(result['U'][r])  # (12,)
        v = np.array(result['V'][r])  # (24,)
        w = np.array(result['W'][r])  # (18,)
        
        # u is indexed by [i*4+j], v by [j*6+k], w by [k*3+i]
        # C[i,k] += sum_j u[i*4+j] * v[j*6+k] * w[k*3+i]
        for i in range(3):
            for k in range(6):
                s = 0.0
                for j in range(4):
                    s += u[i*4+j] * v[j*6+k] * w[k*3+i]
                C_test[i,k] += s
    
    max_err = np.max(np.abs(C_test - C_true))
    print(f"Max error: {max_err:.2e}")
    assert max_err < 1e-10, f"Decomposition is incorrect! Max error = {max_err}"
    print("Verification PASSED")
