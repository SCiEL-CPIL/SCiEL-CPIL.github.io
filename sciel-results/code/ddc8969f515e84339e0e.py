"""Matrix multiplication 3x5x5 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]
n, m, p = DIMENSIONS
nm, mp, pn = n * m, m * p, p * n


def _verify(U, V, W, n, m, p, tol=1e-7):
    """Check: for all i,k,j1,j2: sum_r W[k*n+i,r] U[i*m+j1,r] V[j2*p+k,r] == delta_{j1,j2}"""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    s = np.dot(W[k * n + i, :], U[i * m + j1, :] * V[j2 * p + k, :])
                    target = 1.0 if j1 == j2 else 0.0
                    if abs(s - target) > tol:
                        return False
    return True


def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _build_strassen_2x2x2(n, m, p):
    """
    Use Strassen's 7-term algorithm for 2x2x2 blocks.
    Partition the 3x5x5 multiplication into blocks.
    A is 3x5, B is 5x5, C is 3x5.
    
    We can partition as:
    - Rows of A: 3 -> blocks of size [2, 1]
    - Cols of A / Rows of B: 5 -> blocks of size [2, 2, 1]
    - Cols of B / Cols of C: 5 -> blocks of size [2, 2, 1]
    
    For each (row_block, col_block_pair) in A*B, we get a sub-multiplication.
    The 2x2x2 blocks can use Strassen (rank 7 instead of 8).
    
    Let's be more precise:
    A = [[A11, A12, A13],  [A21, A22, A23]]   (2x2 + 1x1 row blocks, 2+2+1 col blocks)
    B = [[B11, B12, B13],  [B21, B22, B23],   (3x3 block matrix with 2+2+1 row/col blocks)
         [B31, B32, B33]]
    
    C = A*B is 3x5:
    C = [[C11, C12, C13],  [C21, C22, C23]]
         [C31, C32, C33]
    
    where C_ij = sum_k A_ik B_kj
    
    The 2x2x2 sub-blocks are where both the row-block of A is size 2, 
    the col-block of A (row-block of B) is size 2, and the col-block of B is size 2.
    
    Specifically, C11 = A11*B11 + A12*B21 + A13*B31
    Here A11 is 2x2, B11 is 2x2 -> this is a 2x2x2 multiplication (rank 8 schoolbook, 7 Strassen)
    A12 is 2x2, B21 is 2x2 -> another 2x2x2
    A13 is 2x1, B31 is 1x2 -> 2x1x2 (rank 4 schoolbook)
    
    Let me count all sub-blocks:
    Row blocks of A: sizes [2, 1] (2 blocks)
    Col blocks of A = Row blocks of B: sizes [2, 2, 1] (3 blocks)
    Col blocks of B: sizes [2, 2, 1] (3 blocks)
    
    Sub-multiplications for C[rb, cb]:
    For rb in {0,1} (sizes 2,1), cb in {0,1,2} (sizes 2,2,1):
      C[rb,cb] = sum_kb A[rb,kb] * B[kb,cb]
      where kb in {0,1,2} (sizes 2,2,1)
    
    Total sub-multiplications: 2 * 3 * 3 = 18
    Each sub-mult (rb,kb,cb) has dimensions (size_rb x size_kb) * (size_kb x size_cb)
    
    Let's list them:
    rb=0 (size 2):
      kb=0 (size 2), cb=0 (size 2): 2x2 * 2x2 = 2x2x2 -> Strassen saves 1
      kb=0 (size 2), cb=1 (size 2): 2x2 * 2x2 = 2x2x2 -> Strassen saves 1
      kb=0 (size 2), cb=2 (size 1): 2x2 * 2x1 = 2x2x1 -> rank 4
      kb=1 (size 2), cb=0 (size 2): 2x2 * 2x2 = 2x2x2 -> Strassen saves 1
      kb=1 (size 2), cb=1 (size 2): 2x2 * 2x2 = 2x2x2 -> Strassen saves 1
      kb=1 (size 2), cb=2 (size 1): 2x2 * 2x1 = 2x2x1 -> rank 4
      kb=2 (size 1), cb=0 (size 2): 2x1 * 1x2 = 2x1x2 -> rank 2
      kb=2 (size 1), cb=1 (size 2): 2x1 * 1x2 = 2x1x2 -> rank 2
      kb=2 (size 1), cb=2 (size 1): 2x1 * 1x1 = 2x1x1 -> rank 2
    
    rb=1 (size 1):
      kb=0 (size 2), cb=0 (size 2): 1x2 * 2x2 = 1x2x2 -> rank 4
      kb=0 (size 2), cb=1 (size 2): 1x2 * 2x2 = 1x2x2 -> rank 4
      kb=0 (size 2), cb=2 (size 1): 1x2 * 2x1 = 1x2x1 -> rank 2
      kb=1 (size 2), cb=0 (size 2): 1x2 * 2x2 = 1x2x2 -> rank 4
      kb=1 (size 2), cb=1 (size 2): 1x2 * 2x2 = 1x2x2 -> rank 4
      kb=1 (size 2), cb=2 (size 1): 1x2 * 2x1 = 1x2x1 -> rank 2
      kb=2 (size 1), cb=0 (size 2): 1x1 * 1x2 = 1x1x2 -> rank 1
      kb=2 (size 1), cb=1 (size 2): 1x1 * 1x2 = 1x1x2 -> rank 1
      kb=2 (size 1), cb=2 (size 1): 1x1 * 1x1 = 1x1x1 -> rank 1
    
    Schoolbook total: 
    2x2x2: 4 blocks * 8 = 32
    2x2x1: 2 blocks * 4 = 8
    2x1x2: 2 blocks * 2 = 4
    2x1x1: 1 block * 2 = 2
    1x2x2: 4 blocks * 4 = 16
    1x2x1: 2 blocks * 2 = 4
    1x1x2: 2 blocks * 1 = 2
    1x1x1: 1 block * 1 = 1
    Total: 32+8+4+2+16+4+2+1 = 69... wait that's already less than 75?
    
    Hmm, that doesn't seem right. Let me recount.
    Actually the schoolbook rank for the full 3x5x5 is 3*5*5=75.
    The block decomposition should give exactly 75 if each sub-block uses schoolbook.
    
    Let me recount:
    2x2x2: 4 * 8 = 32
    2x2x1: 2 * 4 = 8
    2x1x2: 2 * 2 = 4
    2x1x1: 1 * 2 = 2
    1x2x2: 4 * 4 = 16
    1x2x1: 2 * 2 = 4
    1x1x2: 2 * 1 = 2
    1x1x1: 1 * 1 = 1
    Sum = 32+8+4+2+16+4+2+1 = 69
    
    That's 69, not 75. That can't be right for schoolbook. Let me re-examine.
    
    Oh wait, I think I miscounted the block sizes. Let me redo:
    n=3 rows, m=5 cols/rows, p=5 cols.
    Row blocks of A: [2, 1] -> 2 blocks
    Col blocks of A (= row blocks of B): [2, 2, 1] -> 3 blocks  
    Col blocks of B: [2, 2, 1] -> 3 blocks
    
    Sub-mult (rb, kb, cb) has size (n_rb, m_kb, p_cb) where:
    n_rb = size of row block rb in A
    m_kb = size of col block kb in A (= row block kb in B)
    p_cb = size of col block cb in B
    
    For the schoolbook, rank of sub-mult (a,b,c) = a*b*c.
    
    rb=0 (n=2):
      kb=0 (m=2), cb=0 (p=2): 2*2*2=8
      kb=0 (m=2), cb=1 (p=2): 2*2*2=8
      kb=0 (m=2), cb=2 (p=1): 2*2*1=4
      kb=1 (m=2), cb=0 (p=2): 2*2*2=8
      kb=1 (m=2), cb=1 (p=2): 2*2*2=8
      kb=1 (m=2), cb=2 (p=1): 2*2*1=4
      kb=2 (m=1), cb=0 (p=2): 2*1*2=4
      kb=2 (m=1), cb=1 (p=2): 2*1*2=4
      kb=2 (m=1), cb=2 (p=1): 2*1*1=2
    Subtotal rb=0: 8+8+4+8+8+4+4+4+2 = 50
    
    rb=1 (n=1):
      kb=0 (m=2), cb=0 (p=2): 1*2*2=4
      kb=0 (m=2), cb=1 (p=2): 1*2*2=4
      kb=0 (m=2), cb=2 (p=1): 1*2*1=2
      kb=1 (m=2), cb=0 (p=2): 1*2*2=4
      kb=1 (m=2), cb=1 (p=2): 1*2*2=4
      kb=1 (m=2), cb=2 (p=1): 1*2*1=2
      kb=2 (m=1), cb=0 (p=2): 1*1*2=2
      kb=2 (m=1), cb=1 (p=2): 1*1*2=2
      kb=2 (m=1), cb=2 (p=1): 1*1*1=1
    Subtotal rb=1: 4+4+2+4+4+2+2+2+1 = 25
    
    Total: 50+25 = 75. 
    
    Now with Strassen for the 2x2x2 blocks (4 of them, each saves 1):
    Total = 75 - 4 = 71.
    
    So we can achieve rank 71 by using Strassen's algorithm for the four 2x2x2 sub-blocks.
    """
    # Strassen's 7-term algorithm for 2x2x2:
    # Given A (2x2), B (2x2), compute C = A*B
    # m1 = (a11+a22)(b12-b22)
    # m2 = (a11+a21)b22
    # m3 = a11(b11-b21)
    # m4 = a22(b22-b11)
    # m5 = (a11+a12)b11
    # m6 = (a21+a22)(b11+b12)
    # m7 = (a11-a21)(b21+b22)
    # c11 = m1+m4-m5+m7
    # c12 = m3+m5
    # c21 = m2+m4
    # c22 = m1-m2+m3+m6
    
    # We need to build the full decomposition.
    # The approach: for each sub-block, add its contribution to U, V, W.
    
    # Let's define the block structure:
    # A rows: block 0 = rows 0,1; block 1 = row 2
    # A cols / B rows: block 0 = cols/rows 0,1; block 1 = cols/rows 2,3; block 2 = col/row 4
    # B cols: block 0 = cols 0,1; block 1 = cols 2,3; block 2 = col 4
    
    # For each (rb, kb, cb), we add a sub-decomposition.
    # The sub-decomposition for (a,b,c) dimensions contributes to:
    #   U: rows corresponding to (i, j) where i is in row block rb, j is in col block kb
    #   V: rows corresponding to (j, k) where j is in row block kb, k is in col block cb
    #   W: rows corresponding to (k, i) where k is in col block cb, i is in row block rb
    
    # For a general (a,b,c) sub-block at position (rb, kb, cb):
    # The local indices: i' in [0,a), j' in [0,b), k' in [0,c)
    # Global: i = offset_rb + i', j = offset_kb + j', k = offset_cb + k'
    
    row_offsets = [0, 2]  # offsets for row blocks [2,1]
    col_offsets = [0, 2, 4]  # offsets for col blocks [2,2,1]
    row_sizes = [2, 1]
    col_sizes = [2, 2, 1]
    
    # For each sub-block, determine if it's 2x2x2 (use Strassen) or schoolbook
    # Build the decomposition term by term.
    
    U = np.zeros((nm, 0))
    V = np.zeros((mp, 0))
    W = np.zeros((pn, 0))
    
    def add_terms(U_list, V_list, W_list, u_entries, v_entries, w_entries):
        """Add a set of rank-1 terms. Each term is (u_row, v_row, w_row, value)."""
        # We'll build column by column
        pass
    
    # Simpler approach: build arrays directly.
    # First, determine total rank needed.
    total_rank = 0
    sub_blocks = []
    for rb in range(2):
        for kb in range(3):
            for cb in range(3):
                a = row_sizes[rb]
                b = col_sizes[kb]
                c = col_sizes[cb]
                if a == 2 and b == 2 and c == 2:
                    rank_sub = 7  # Strassen
                else:
                    rank_sub = a * b * c  # schoolbook
                total_rank += rank_sub
                sub_blocks.append((rb, kb, cb, a, b, c, rank_sub))
    
    U = np.zeros((nm, total_rank))
    V = np.zeros((mp, total_rank))
    W = np.zeros((pn, total_rank))
    
    col = 0
    for (rb, kb, cb, a, b, c, rank_sub) in sub_blocks:
        i_off = row_offsets[rb]
        j_off = col_offsets[kb]
        k_off = col_offsets[cb]
        
        if a == 2 and b == 2 and c == 2:
            # Strassen's 7 terms
            # m1 = (a00+a11)(b01-b11)
            # m2 = (a00+a10)b11
            # m3 = a00(b00-b10)
            # m4 = a11(b11-b00)
            # m5 = (a00+a01)b00
            # m6 = (a10+a11)(b00+b01)
            # m7 = (a00-a10)(b10+b11)
            
            # m1: U has (a00+a11) at positions (0,0) and (1,1); V has (b01-b11) at (0,1) and (1,1)
            # Actually, let me think about this differently.
            # In Strassen, we compute 7 products m1..m7, each is a scalar product of two "linear combinations".
            # m1 = M1 * N1 where M1 = a00+a11, N1 = b01-b11
            # m2 = M2 * N2 where M2 = a00+a10, N2 = b11
            # m3 = M3 * N3 where M3 = a00, N3 = b00-b10
            # m4 = M4 * N4 where M4 = a11, N4 = b11-b00
            # m5 = M5 * N5 where M5 = a00+a01, N5 = b00
            # m6 = M6 * N6 where M6 = a10+a11, N6 = b00+b01
            # m7 = M7 * N7 where M7 = a00-a10, N7 = b10+b11
            
            # Each mr contributes to the output C via linear combinations.
            # c00 = m1+m4-m5+m7
            # c01 = m3+m5
            # c10 = m2+m4
            # c11 = m1-m2+m3+m6
            
            # For the tensor decomposition, each mr is a rank-1 term in the sub-block.
            # The U factor for mr: coefficients of the linear combination in A
            # The V factor for mr: coefficients of the linear combination in B
            # The W factor for mr: coefficients for how mr contributes to C
            
            # m1: U: (1,0,0,1) [a00+a11], V: (0,1,1,-1) [b01-b11], W: (1,0,0,1) [c00+=m1, c11+=m1]
            # Wait, W should encode how m1 contributes to each output element.
            # m1 contributes +1 to c00 and +1 to c11.
            # So W: c00 coeff=1, c01 coeff=0, c10 coeff=0, c11 coeff=1
            
            # Let me define for each mr: (U_coeffs, V_coeffs, W_coeffs)
            # U_coeffs: 4 values for (a00,a01,a10,a11)
            # V_coeffs: 4 values for (b00,b01,b10,b11)
            # W_coeffs: 4 values for (c00,c01,c10,c11)
            
            strassen_terms = [
                # (U: [a00,a01,a10,a11], V: [b00,b01,b10,b11], W: [c00,c01,c10,c11])
                ([1, 0, 0, 1], [0, 1, 0, -1], [1, 0, 0, 1]),   # m1
                ([1, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 0]),     # m2
                ([1, 0, 0, 0], [1, 0, -1, 0], [0, 1, 0, 0]),    # m3
                ([0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 1, 0]),     # m4 -- wait
            ]
            
            # Let me redo this more carefully.
            # m1 = (a00+a11)(b01-b11)
            #   U: a00 coeff=1, a01 coeff=0, a10 coeff=0, a11 coeff=1
            #   V: b00 coeff=0, b01 coeff=1, b10 coeff=0, b11 coeff=-1
            #   W: m1 contributes to c00 (+1) and c11 (+1)
            #   So W: c00=1, c01=0, c10=0, c11=1
            
            # m2 = (a00+a10)b11
            #   U: a00=1, a01=0, a10=1, a11=0
            #   V: b00=0, b01=0, b10=0, b11=1
            #   W: m2 contributes to c10 (+1) and c11 (-1)
            #   So W: c00=0, c01=0, c10=1, c11=-1
            
            # m3 = a00(b00-b10)
            #   U: a00=1, a01=0, a10=0, a11=0
            #   V: b00=1, b01=0, b10=-1, b11=0
            #   W: m3 contributes to c01 (+1) and c11 (+1)
            #   So W: c00=0, c01=1, c10=0, c11=1
            
            # m4 = a11(b11-b00)
            #   U: a00=0, a01=0, a10=0, a11=1
            #   V: b00=-1, b01=0, b10=0, b11=1
            #   W: m4 contributes to c00 (+1) and c10 (+1)
            #   So W: c00=1, c01=0, c10=1, c11=0
            
            # m5 = (a00+a01)b00
            #   U: a00=1, a01=1, a10=0, a11=0
            #   V: b00=1, b01=0, b10=0, b11=0
            #   W: m5 contributes to c00 (-1) and c01 (+1)
            #   So W: c00=-1, c01=1, c10=0, c11=0
            
            # m6 = (a10+a11)(b00+b01)
            #   U: a00=0, a01=0, a10=1, a11=1
            #   V: b00=1, b01=1, b10=0, b11=0
            #   W: m6 contributes to c11 (+1)
            #   So W: c00=0, c01=0, c10=0, c11=1
            
            # m7 = (a00-a10)(b10+b11)
            #   U: a00=1, a01=0, a10=-1, a11=0
            #   V: b00=0, b01=0, b10=1, b11=1
            #   W: m7 contributes to c00 (+1)
            #   So W: c00=1, c01=0, c10=0, c11=0
            
            strassen_terms = [
                ([1, 0, 0, 1], [0, 1, 0, -1], [1, 0, 0, 1]),   # m1
                ([1, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, -1]),   # m2
                ([1, 0, 0, 0], [1, 0, -1, 0], [0, 1, 0, 1]),   # m3
                ([0, 0, 0, 1], [-1, 0, 0, 1], [1, 0, 1, 0]),   # m4
                ([1, 1, 0, 0], [1, 0, 0, 0], [-1, 1, 0, 0]),   # m5
                ([0, 0, 1, 1], [1, 1, 0, 0], [0, 0, 0, 1]),   # m6
                ([1, 0, -1, 0], [0, 0, 1, 1], [1, 0, 0, 0]),   # m7
            ]
            
            for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
                # U: 4 values for (i'=0,j'=0), (i'=0,j'=1), (i'=1,j'=0), (i'=1,j'=1)
                for li in range(2):
                    for lj in range(2):
                        idx = li * 2 + lj
                        U[(i_off + li) * m + (j_off + lj), col] = u_coeffs[idx]
                # V: 4 values for (j'=0,k'=0), (j'=0,k'=1), (j'=1,k'=0), (j'=1,k'=1)
                for lj in range(2):
                    for lk in range(2):
                        idx = lj * 2 + lk
                        V[(j_off + lj) * p + (k_off + lk), col] = v_coeffs[idx]
                # W: 4 values for (k'=0,i'=0), (k'=0,i'=1), (k'=1,i'=0), (k'=1,i'=1)
                # W index is k*n+i, so for k'=0,i'=0: (k_off+0)*n+(i_off+0)
                # For k'=0,i'=1: (k_off+0)*n+(i_off+1)
                # For k'=1,i'=0: (k_off+1)*n+(i_off+0)
                # For k'=1,i'=1: (k_off+1)*n+(i_off+1)
                for lk in range(2):
                    for li in range(2):
                        idx = lk * 2 + li
                        W[(k_off + lk) * n + (i_off + li), col] = w_coeffs[idx]
                col += 1
        else:
            # Schoolbook for this sub-block
            for li in range(a):
                for lj in range(b):
                    for lk in range(c):
                        U[(i_off + li) * m + (j_off + lj), col] = 1
                        V[(j_off + lj) * p + (k_off + lk), col] = 1
                        W[(k_off + lk) * n + (i_off + li), col] = 1
                        col += 1
    
    return U, V, W, total_rank


def _als_optimize(U, V, W, n, m, p, max_iters=100, tol=1e-8):
    """ALS-style optimization for the tensor decomposition."""
    for it in range(max_iters):
        # Update W
        for k in range(p):
            for i in range(n):
                A_W = np.zeros((m * m, W.shape[1]))
                b = np.zeros(m * m)
                for j1 in range(m):
                    for j2 in range(m):
                        A_W[j1 * m + j2, :] = U[i * m + j1, :] * V[j2 * p + k, :]
                        b[j1 * m + j2] = 1.0 if j1 == j2 else 0.0
                w, *_ = np.linalg.lstsq(A_W, b, rcond=None)
                W[k * n + i, :] = w
        
        # Update U
        for i in range(n):
            for j1 in range(m):
                A_U = np.zeros((p * m, U.shape[1]))
                b = np.zeros(p * m)
                for k in range(p):
                    for j2 in range(m):
                        A_U[k * m + j2, :] = W[k * n + i, :] * V[j2 * p + k, :]
                        b[k * m + j2] = 1.0 if j1 == j2 else 0.0
                u, *_ = np.linalg.lstsq(A_U, b, rcond=None)
                U[i * m + j1, :] = u
        
        # Update V
        for j2 in range(m):
            for k in range(p):
                A_V = np.zeros((n * m, V.shape[1]))
                b = np.zeros(n * m)
                for i in range(n):
                    for j1 in range(m):
                        A_V[i * m + j1, :] = W[k * n + i, :] * U[i * m + j1, :]
                        b[i * m + j1] = 1.0 if j1 == j2 else 0.0
                v, *_ = np.linalg.lstsq(A_V, b, rcond=None)
                V[j2 * p + k, :] = v
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    n, m, p = 3, 5, 5
    
    # First, try the Strassen-based decomposition (rank 71)
    U_str, V_str, W_str, rank_str = _build_strassen_2x2x2(n, m, p)
    
    # Verify the Strassen decomposition
    if _verify(U_str, V_str, W_str, n, m, p):
        best_rank = rank_str
        best_U = U_str.copy()
        best_V = V_str.copy()
        best_W = W_str.copy()
    else:
        # Fallback to schoolbook
        U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
        best_rank = n * m * p
        best_U = U_sb
        best_V = V_sb
        best_W = W_sb
    
    # Try to improve further with ALS optimization at lower ranks
    # Try ranks from best_rank-1 down to a reasonable lower bound
    lower_bound = max(n * m, m * p, p * n)  # 25
    
    np.random.seed(42)
    
    for r in range(best_rank - 1, max(lower_bound, best_rank - 10)):
        if time.time() > deadline - 10:
            break
        
        success = False
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            
            U = np.random.randn(n * m, r) * 0.3
            V = np.random.randn(m * p, r) * 0.3
            W = np.random.randn(p * n, r) * 0.3
            
            # ALS optimization
            U, V, W = _als_optimize(U, V, W, n, m, p, max_iters=150)
            
            # Round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            if _verify(U_r, V_r, W_r, n, m, p):
                best_rank = r
                best_U = U_r.copy()
                best_V = V_r.copy()
                best_W = W_r.copy()
                success = True
                break
        
        if success:
            continue  # Try even lower
        else:
            # If we failed at rank r, try one more restart at the same rank
            if time.time() < deadline - 10:
                for restart in range(2):
                    if time.time() > deadline - 10:
                        break
                    U = np.random.randn(n * m, r) * 0.5
                    V = np.random.randn(m * p, r) * 0.5
                    W = np.random.randn(p * n, r) * 0.5
                    U, V, W = _als_optimize(U, V, W, n, m, p, max_iters=200)
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    if _verify(U_r, V_r, W_r, n, m, p):
                        best_rank = r
                        best_U = U_r.copy()
                        best_V = V_r.copy()
                        best_W = W_r.copy()
                        break
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
