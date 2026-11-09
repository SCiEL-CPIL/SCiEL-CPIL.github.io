"""Matrix multiplication 4x5x5 with rank minimization via structured decomposition."""

import time
import math

DIMENSIONS = [4, 5, 5]

def _schoolbook(n, m, p):
    """Return U, V, W for the schoolbook decomposition of n x m x p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _strassen_2x2():
    """Return U, V, W for rank-7 Strassen decomposition of 2x2x2."""
    # Strassen's 7 products for 2x2 matrix multiplication
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # M1: (A11+A22)(B11+B22)
    # U: A11=U[0,0]=1, A22=U[3,0]=1
    # V: B11=V[0,0]=1, B22=V[3,0]=1
    # W: C11 gets +1, C22 gets +1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # C11 (k=0,i=0) and C22 (k=1,i=1)
    
    # M2: (A21+A22)B11
    # U: A21=U[2,1]=1, A22=U[3,1]=1
    # V: B11=V[0,1]=1
    # W: C21 gets +1 (k=0,i=1)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[4][1] = 1  # k=0, i=1 -> 0*2+1=1? Wait, W[k*n+i]. k=0,i=1 -> 1. Let me re-check.
    # W index: k*n+i. For C21: row=1, col=0. k=0, i=1. W[0*2+1]=W[1].
    W[1][1] = 1
    
    # M3: A11(B12-B22)
    # U: A11=U[0,2]=1
    # V: B12=V[1,2]=1, B22=V[3,2]=-1
    # W: C12 gets +1 (k=1,i=0). W[1*2+0]=W[2].
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1
    
    # M4: A22(B21-B11)
    # U: A22=U[3,3]=1
    # V: B21=V[2,3]=1, B11=V[0,3]=-1
    # W: C11 gets +1 (k=0,i=0), C21 gets +1 (k=0,i=1)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5: (A11+A12)B22
    # U: A11=U[0,4]=1, A12=U[1,4]=1
    # V: B22=V[3,4]=1
    # W: C11 gets -1 (k=0,i=0)
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1
    
    # M6: (A21-A11)(B11+B12)
    # U: A21=U[2,5]=1, A11=U[0,5]=-1
    # V: B11=V[0,5]=1, B12=V[1,5]=1
    # W: C22 gets +1 (k=1,i=1)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (A12-A22)(B21+B22)
    # U: A12=U[1,6]=1, A22=U[3,6]=-1
    # V: B21=V[2,6]=1, B22=V[3,6]=1
    # W: C11 gets +1 (k=0,i=0)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _combine_blocks(block_specs):
    """
    Combine block decompositions.
    block_specs: list of (U_block, V_block, W_block, row_offset, col_offset)
    where U_block is for the full U space but only non-zero in the relevant rows,
    etc. Actually, let's just concatenate the rank dimensions.
    
    Better approach: each block decomposition already has U, V, W of the right
    full size (n*m, m*p, p*n) with rank r_block. We just stack them.
    """
    U_parts = []
    V_parts = []
    W_parts = []
    for U_b, V_b, W_b in block_specs:
        U_parts.append(U_b)
        V_parts.append(V_b)
        W_parts.append(W_b)
    
    total_rank = sum(len(u[0]) for u in U_parts)
    n, m, p = DIMENSIONS
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for U_b, V_b, W_b in block_specs:
        r = len(U_b[0])
        for i in range(n * m):
            for j in range(r):
                U[i][col_offset + j] = U_b[i][j]
        for i in range(m * p):
            for j in range(r):
                V[i][col_offset + j] = V_b[i][j]
        for i in range(p * n):
            for j in range(r):
                W[i][col_offset + j] = W_b[i][j]
        col_offset += r
    
    return U, V, W

def _embed_block(U_block, V_block, W_block, r, n_full, m_full, p_full, i0, j0, k0, n_b, m_b, p_b):
    """
    Embed a block decomposition into the full-sized matrices.
    U_block: (n_b*m_b, r), V_block: (m_b*p_b, r), W_block: (p_b*n_b, r)
    i0, j0, k0: offsets in the full matrix
    """
    U_full = [[0] * r for _ in range(n_full * m_full)]
    V_full = [[0] * r for _ in range(m_full * p_full)]
    W_full = [[0] * r for _ in range(p_full * n_full)]
    
    for a in range(n_b):
        for b in range(m_b):
            ui = a * m_b + b
            ufull_i = (i0 + a) * m_full + (j0 + b)
            for c in range(r):
                U_full[ufull_i][c] = U_block[ui][c]
    
    for b in range(m_b):
        for c in range(p_b):
            vi = b * p_b + c
            vfull_i = (j0 + b) * p_full + (k0 + c)
            for d in range(r):
                V_full[vfull_i][d] = V_block[vi][d]
    
    for c in range(p_b):
        for a in range(n_b):
            wi = c * n_b + a
            wfull_i = (k0 + c) * n_full + (i0 + a)
            for e in range(r):
                W_full[wfull_i][e] = W_block[wi][e]
    
    return U_full, V_full, W_full

def construct():
    """
    Construct a rank-minimal decomposition for 4x5x5 matrix multiplication.
    
    Strategy: Decompose 4x5x5 into blocks:
    - A (4x5) split as A_left (4x4) and A_right (4x1)
    - B (5x5) split as 4x4, 4x1, 1x4, 1x1 blocks
    
    C = A*B where:
    C[:4,:4] = A_left * B_top_left (4x4 * 4x4) + A_right * B_bottom_left (4x1 * 1x4)
    C[:4,4]  = A_left * B_top_right (4x4 * 4x1) + A_right * B_bottom_right (4x1 * 1x1)
    
    Blocks:
    1. 4x4x4: Use Strassen recursively. 4=2*2, so split into 2x2x2 blocks.
       Each 2x2x2 uses rank 7. Total for 4x4x4: we can use a more clever approach.
       Actually, for 4x4x4, we can use the fact that it's 2x2 blocks of 2x2.
       The rank of 4x4x4 is at most 4*4*4=64 by schoolbook, but can be reduced.
       Let's use: split 4x4x4 into 8 blocks of 2x2x2 (since 4=2*2 in each dim).
       That gives 8 * 7 = 56. But we can do better with Strassen on the 2x2 block level.
       
       Actually, let me think differently. For 4x4x4, use the following:
       Split A (4x4) into 4 blocks of 2x2, B (4x4) into 4 blocks of 2x2.
       C is 4x4, composed of 4 blocks of 2x2.
       C_ij = sum_k A_ik B_kj
       Using Strassen at the block level: 7 products of 2x2x2 matrices, each rank 7.
       Total: 7 * 7 = 49.
       
    2. 4x1x4: Schoolbook rank = 4*1*4 = 16
    3. 4x4x1: Schoolbook rank = 4*4*1 = 16
    4. 4x1x1: Schoolbook rank = 4*1*1 = 4
    
    Total: 49 + 16 + 16 + 4 = 85
    
    Let's implement this.
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Block 1: 4x4x4 using Strassen at block level (7 products of 2x2x2, each rank 7)
    # We need to build a 4x4x4 decomposition of rank 49.
    # U shape: (4*4, 49) = (16, 49)
    # V shape: (4*4, 49) = (16, 49)
    # W shape: (4*4, 49) = (16, 49)
    
    # Strassen for 2x2 blocks of 2x2 matrices:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mk is a 2x2x2 multiplication, rank 7.
    # So total rank = 7 * 7 = 49.
    
    U_444 = [[0] * 49 for _ in range(16)]
    V_444 = [[0] * 49 for _ in range(16)]
    W_444 = [[0] * 49 for _ in range(16)]
    
    # Strassen 2x2x2 factors
    U_s, V_s, W_s = _strassen_2x2()
    
    # For each Strassen product Mk, we need to build the U, V, W contributions.
    # The 4x4 matrices are composed of 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]] where each is 2x2
    
    # Block indices in the 4x4 space:
    # A11: rows 0-1, cols 0-1 -> U indices: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
    # A12: rows 0-1, cols 2-3 -> U indices: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
    # A21: rows 2-3, cols 0-1 -> U indices: 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13
    # A22: rows 2-3, cols 2-3 -> U indices: 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15
    
    # Similarly for V (B) and W (C).
    
    # For each Strassen term, we define which A-blocks and B-blocks are involved,
    # with what signs, and which C-block gets the result.
    
    strassen_terms = [
        # (A_block_coeff, B_block_coeff, C_block)
        # M1: (A11+A22)(B11+B22) -> C11 += 1, C22 += 1
        # M2: (A21+A22)B11 -> C21 += 1
        # M3: A11(B12-B22) -> C12 += 1
        # M4: A22(B21-B11) -> C11 += 1, C21 += 1
        # M5: (A11+A12)B22 -> C11 -= 1
        # M6: (A21-A11)(B11+B12) -> C22 += 1
        # M7: (A12-A22)(B21+B22) -> C11 += 1
    ]
    
    # Define block index ranges for 2x2 blocks in 4x4 space
    # Block (br, bc) where br, bc in {0,1}: rows br*2 to br*2+1, cols bc*2 to bc*2+1
    def block_indices(br, bc):
        """Return (u_rows, v_cols, w_rows) for block (br, bc) in a 4x4 matrix."""
        rows = list(range(br * 2, br * 2 + 2))
        cols = list(range(bc * 2, bc * 2 + 2))
        # U indices: row * 4 + col
        u_idx = [r * 4 + c for r in rows for c in cols]
        # V indices: same as U for the B matrix
        v_idx = [r * 4 + c for r in rows for c in cols]
        # W indices: col * 4 + row (since W[k*n+i], k=col, i=row)
        w_idx = [c * 4 + r for r in rows for c in cols]
        return u_idx, v_idx, w_idx
    
    # Block (br, bc) -> indices
    blocks = {}
    for br in range(2):
        for bc in range(2):
            blocks[(br, bc)] = block_indices(br, bc)
    
    # A blocks: A11=(0,0), A12=(0,1), A21=(1,0), A22=(1,1)
    A_blocks = {
        'A11': (0, 0),
        'A12': (0, 1),
        'A21': (1, 0),
        'A22': (1, 1),
    }
    B_blocks = {
        'B11': (0, 0),
        'B12': (0, 1),
        'B21': (1, 0),
        'B22': (1, 1),
    }
    C_blocks = {
        'C11': (0, 0),
        'C12': (0, 1),
        'C21': (1, 0),
        'C22': (1, 1),
    }
    
    # Strassen products:
    # M1: (A11+A22)(B11+B22), contributes to C11 (+1) and C22 (+1)
    # M2: (A21+A22)B11, contributes to C21 (+1)
    # M3: A11(B12-B22), contributes to C12 (+1)
    # M4: A22(B21-B11), contributes to C11 (+1) and C21 (+1)
    # M5: (A11+A12)B22, contributes to C11 (-1)
    # M6: (A21-A11)(B11+B12), contributes to C22 (+1)
    # M7: (A12-A22)(B21+B22), contributes to C11 (+1)
    
    strassen_data = [
        # (a_coeffs, b_coeffs, c_targets)
        # a_coeffs: dict from block name to coefficient
        # b_coeffs: dict from block name to coefficient
        # c_targets: dict from block name to coefficient
        ({'A11': 1, 'A22': 1}, {'B11': 1, 'B22': 1}, {'C11': 1, 'C22': 1}),
        ({'A21': 1, 'A22': 1}, {'B11': 1}, {'C21': 1}),
        ({'A11': 1}, {'B12': 1, 'B22': -1}, {'C12': 1}),
        ({'A22': 1}, {'B21': 1, 'B11': -1}, {'C11': 1, 'C21': 1}),
        ({'A11': 1, 'A12': 1}, {'B22': 1}, {'C11': -1}),
        ({'A21': 1, 'A11': -1}, {'B11': 1, 'B12': 1}, {'C22': 1}),
        ({'A12': 1, 'A22': -1}, {'B21': 1, 'B22': 1}, {'C11': 1}),
    ]
    
    for s, (a_coeffs, b_coeffs, c_targets) in enumerate(strassen_data):
        rank_offset = s * 7
        # For each non-zero A coefficient
        for a_name, a_coeff in a_coeffs.items():
            if a_coeff == 0:
                continue
            a_br, a_bc = A_blocks[a_name]
            u_rows = blocks[(a_br, a_bc)][0]
            for r in range(7):
                u_val = U_s[0][r]  # Strassen U factor for the first row
                # Actually, we need to be more careful. The Strassen 2x2x2
                # has U[i*m+j, c] for i,j in 0,1 and c in 0,6.
                # For each 2x2 block, we map the local (i,j) to global.
                # U_s[i*2+j, c] gives the coefficient for A_ij in product c.
                # But we're embedding, so for each local (li, lj) in the 2x2 block,
                # the global U index is (a_br*2+li)*4 + (a_bc*2+lj).
                # And the Strassen factor is U_s[li*2+lj, r].
                
                # Let me redo this more carefully.
                pass
        
        # Let me restructure: for each Strassen product s, and for each rank column c in 0..6,
        # we need to set U, V, W entries.
        # The local 2x2 block (a_br, a_bc) maps to global rows a_br*2..a_br*2+1 and cols a_bc*2..a_bc*2+1.
        # For local (li, lj), global U index = (a_br*2+li)*4 + (a_bc*2+lj).
        # Strassen U factor: U_s[li*2+lj, c].
        
        for c_local in range(7):
            c_global = rank_offset + c_local
            # A part
            for a_name, a_coeff in a_coeffs.items():
                if a_coeff == 0:
                    continue
                a_br, a_bc = A_blocks[a_name]
                for li in range(2):
                    for lj in range(2):
                        global_u_idx = (a_br * 2 + li) * 4 + (a_bc * 2 + lj)
                        local_u_idx = li * 2 + lj
                        U_444[global_u_idx][c_global] += a_coeff * U_s[local_u_idx][c_local]
            
            # B part
            for b_name, b_coeff in b_coeffs.items():
                if b_coeff == 0:
                    continue
                b_br, b_bc = B_blocks[b_name]
                for li in range(2):
                    for lj in range(2):
                        global_v_idx = (b_br * 2 + li) * 4 + (b_bc * 2 + lj)
                        local_v_idx = li * 2 + lj
                        V_444[global_v_idx][c_global] += b_coeff * V_s[local_v_idx][c_local]
            
            # C part
            for c_name, c_coeff in c_targets.items():
                if c_coeff == 0:
                    continue
                c_br, c_bc = C_blocks[c_name]
                for li in range(2):
                    for lj in range(2):
                        # W index: k*n+i where k=col, i=row in the C matrix
                        # Local (li, lj) means row=li, col=lj in the 2x2 block
                        # Global row = c_br*2+li, global col = c_bc*2+lj
                        # W index = col*4 + row = (c_bc*2+lj)*4 + (c_br*2+li)
                        global_w_idx = (c_bc * 2 + lj) * 4 + (c_br * 2 + li)
                        # Strassen W index: k*n+i where k=lj, i=li, n=2
                        # W_s[lj*2+li, c_local]
                        local_w_idx = lj * 2 + li
                        W_444[global_w_idx][c_global] += c_coeff * W_s[local_w_idx][c_local]
    
    # Now embed U_444, V_444, W_444 into full 4x5x5 space
    # The 4x4x4 block corresponds to rows 0-3 of A, cols 0-3 of B, rows 0-3, cols 0-3 of C
    # U_444 is (16, 49) for the 4x4 part. Full U is (4*5=20, rank).
    # U index in full: i*5 + j where i in 0..3, j in 0..3 for this block.
    # U_444 index: i*4 + j where i in 0..3, j in 0..3.
    
    # Block 2: 4x1x4 (A_right * B_bottom_left)
    # A: 4x1 (rows 0-3, col 4), B: 1x4 (row 4, cols 0-3), C: 4x4 (rows 0-3, cols 0-3)
    # Rank = 4*1*4 = 16
    # C[i,k] = A[i,0] * B[0,k] for i in 0..3, k in 0..3
    # For each (i,k): U[i*5+4, r] = 1, V[4*5+k, r] = 1, W[k*4+i, r] = 1
    # Wait, U index: i*m+j = i*5+j. j=4. So U[i*5+4, r].
    # V index: j*p+k. j=4, p=5. V[4*5+k, r] = V[20+k, r].
    # W index: k*n+i. k in 0..3, i in 0..3. W[k*4+i, r].
    
    U_414 = [[0] * 16 for _ in range(20)]
    V_414 = [[0] * 16 for _ in range(25)]
    W_414 = [[0] * 16 for _ in range(20)]
    
    col = 0
    for i in range(4):
        for k in range(4):
            U_414[i * 5 + 4][col] = 1
            V_414[4 * 5 + k][col] = 1
            W_414[k * 4 + i][col] = 1
            col += 1
    
    # Block 3: 4x4x1 (A_left * B_top_right)
    # A: 4x4 (rows 0-3, cols 0-3), B: 4x1 (rows 0-3, col 4), C: 4x1 (rows 0-3, col 4)
    # Rank = 4*4*1 = 16
    # C[i,0] = sum_j A[i,j] * B[j,0]
    # For each (i,j): U[i*5+j, r] = 1, V[j*5+4, r] = 1, W[4*4+i, r] = 1
    # W index: k*n+i. k=4, i in 0..3. W[4*4+i, r] = W[16+i, r].
    
    U_441 = [[0] * 16 for _ in range(20)]
    V_441 = [[0] * 16 for _ in range(25)]
    W_441 = [[0] * 16 for _ in range(20)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            U_441[i * 5 + j][col] = 1
            V_441[j * 5 + 4][col] = 1
            W_441[4 * 4 + i][col] = 1
            col += 1
    
    # Block 4: 4x1x1 (A_right * B_bottom_right)
    # A: 4x1 (rows 0-3, col 4), B: 1x1 (row 4, col 4), C: 4x1 (rows 0-3, col 4)
    # Rank = 4*1*1 = 4
    # C[i,0] = A[i,0] * B[0,0]
    # For each i: U[i*5+4, r] = 1, V[4*5+4, r] = 1, W[4*4+i, r] = 1
    
    U_411 = [[0] * 4 for _ in range(20)]
    V_411 = [[0] * 4 for _ in range(25)]
    W_411 = [[0] * 4 for _ in range(20)]
    
    col = 0
    for i in range(4):
        U_411[i * 5 + 4][col] = 1
        V_411[4 * 5 + 4][col] = 1
        W_411[4 * 4 + i][col] = 1
        col += 1
    
    # Now combine all blocks.
    # Block 1 (4x4x4): U_444 is (16, 49), needs to be mapped to full (20, rank)
    # U_444[i*4+j, c] -> full U[i*5+j, c] for i in 0..3, j in 0..3
    # V_444[j*4+k, c] -> full V[j*5+k, c] for j in 0..3, k in 0..3
    # W_444[k*4+i, c] -> full W[k*4+i, c] for k in 0..3, i in 0..3 (already correct since n=4)
    
    total_rank = 49 + 16 + 16 + 4  # 85
    U = [[0] * total_rank for _ in range(20)]
    V = [[0] * total_rank for _ in range(25)]
    W = [[0] * total_rank for _ in range(20)]
    
    # Block 1: 4x4x4, rank 49
    for i in range(4):
        for j in range(4):
            u444_idx = i * 4 + j
            ufull_idx = i * 5 + j
            for c in range(49):
                U[ufull_idx][c] = U_444[u444_idx][c]
    
    for j in range(4):
        for k in range(4):
            v444_idx = j * 4 + k
            vfull_idx = j * 5 + k
            for c in range(49):
                V[vfull_idx][c] = V_444[v444_idx][c]
    
    for k in range(4):
        for i in range(4):
            w444_idx = k * 4 + i
            wfull_idx = k * 4 + i  # Same since n=4
            for c in range(49):
                W[wfull_idx][c] = W_444[w444_idx][c]
    
    # Block 2: 4x1x4, rank 16, offset 49
    offset = 49
    for i in range(20):
        for c in range(16):
            U[i][offset + c] = U_414[i][c]
    for i in range(25):
        for c in range(16):
            V[i][offset + c] = V_414[i][c]
    for i in range(20):
        for c in range(16):
            W[i][offset + c] = W_414[i][c]
    
    # Block 3: 4x4x1, rank 16, offset 65
    offset = 65
    for i in range(20):
        for c in range(16):
            U[i][offset + c] = U_441[i][c]
    for i in range(25):
        for c in range(16):
            V[i][offset + c] = V_441[i][c]
    for i in range(20):
        for c in range(16):
            W[i][offset + c] = W_441[i][c]
    
    # Block 4: 4x1x1, rank 4, offset 81
    offset = 81
    for i in range(20):
        for c in range(4):
            U[i][offset + c] = U_411[i][c]
    for i in range(25):
        for c in range(4):
            V[i][offset + c] = V_411[i][c]
    for i in range(20):
        for c in range(4):
            W[i][offset + c] = W_411[i][c]
    
    return {"U": U, "V": V, "W": W}
