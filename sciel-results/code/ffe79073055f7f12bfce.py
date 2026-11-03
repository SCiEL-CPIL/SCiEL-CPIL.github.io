"""5x5 matrix multiplication with rank-reduced decomposition."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Strassen decomposition for 2x2 * 2x2 -> 2x2, rank 7.
    Returns U (4x7), V (4x7), W (4x7) as numpy arrays.
    
    Convention:
    A is 2x2, B is 2x2, C = A*B
    U indexed by i*m+j (i,j in {0,1}): rows 0,1,2,3 for (0,0),(0,1),(1,0),(1,1)
    V indexed by j*p+k (j,k in {0,1}): rows 0,1,2,3 for (0,0),(0,1),(1,0),(1,1)
    W indexed by k*n+i (k,i in {0,1}): rows 0,1,2,3 for (0,0),(0,1),(1,0),(1,1)
    
    C[i,k] = sum_r (sum_{i',j'} U[i'*2+j', r] * A[i',j']) * (sum_{j',k'} V[j'*2+k', r] * B[j',k']) * W[k*2+i, r]
    """
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # Term 0: P1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 1: P2 = (a10+a11)*b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # Term 2: P3 = a00*(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # Term 3: P4 = a11*(b10-b00)
    U[3, 3] = 1
    V[0, 3] = -1; V[2, 3] = 1
    W[0, 3] = 1; W[2, 3] = 1
    
    # Term 4: P5 = (a00+a01)*b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Term 5: P6 = (a10-a00)*(b00+b01)
    U[0, 5] = -1; U[2, 5] = 1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 6: P7 = (a01-a11)*(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def _strassen_4x4():
    """
    Strassen decomposition for 4x4 * 4x4 -> 4x4, rank 49.
    Composes two levels of Strassen.
    """
    U2, V2, W2 = _strassen_2x2()
    
    # Level 1: 4x4 Strassen has 7 big terms
    # Each big term involves 2x2 block combinations
    # Big term r: (A_block1 + A_block2) * (B_block1 + B_block2) etc.
    
    # The 7 big terms for 4x4 Strassen:
    # P1 = (A00+A11)(B00+B11)  -> blocks (0,0)+(1,1) times (0,0)+(1,1)
    # P2 = (A10+A11)B00        -> blocks (1,0)+(1,1) times (0,0)
    # P3 = A00(B01-B11)        -> block (0,0) times (0,1)-(1,1)
    # P4 = A11(B10-B00)        -> block (1,1) times (1,0)-(0,0)
    # P5 = (A00+A01)B11        -> blocks (0,0)+(0,1) times (1,1)
    # P6 = (A10-A00)(B00+B01)  -> blocks (1,0)-(0,0) times (0,0)+(0,1)
    # P7 = (A01-A11)(B10+B11)  -> blocks (0,1)-(1,1) times (1,0)+(1,1)
    
    # Block positions in 4x4 (2x2 blocks):
    # A00 = rows 0-1, cols 0-1
    # A01 = rows 0-1, cols 2-3
    # A10 = rows 2-3, cols 0-1
    # A11 = rows 2-3, cols 2-3
    
    # For each big term, we need to map the 2x2 Strassen to the appropriate blocks.
    
    # Block row ranges: block (br, bc) covers rows br*2 to br*2+1, cols bc*2 to bc*2+1
    # U index for A[i,j] = i*4+j
    # V index for B[i,j] = i*4+j
    # W index for C[i,j] = j*4+i  (k*n+i convention)
    
    def block_u_indices(br, bc):
        """Return U row indices for block (br, bc) of A."""
        return [(br*2+di)*4 + (bc*2+dj) for di in range(2) for dj in range(2)]
    
    def block_v_indices(br, bc):
        """Return V row indices for block (br, bc) of B."""
        return [(br*2+di)*4 + (bc*2+dj) for di in range(2) for dj in range(2)]
    
    def block_w_indices(br, bc):
        """Return W row indices for block (br, bc) of C.
        C[br*2+di, bc*2+dj] -> W index = (bc*2+dj)*4 + (br*2+di)
        """
        return [(bc*2+dj)*4 + (br*2+di) for di in range(2) for dj in range(2)]
    
    # Define the 7 big terms as (u_blocks, v_blocks) where each block has a coefficient
    # Each entry is (block_row, block_col, coeff)
    big_terms = [
        # P1 = (A00+A11)(B00+B11)
        (([ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ])),
        # P2 = (A10+A11)B00
        (([ (1,0,1), (1,1,1) ], [ (0,0,1) ])),
        # P3 = A00(B01-B11)
        (([ (0,0,1) ], [ (0,1,1), (1,1,-1) ])),
        # P4 = A11(B10-B00)
        (([ (1,1,1) ], [ (1,0,1), (0,0,-1) ])),
        # P5 = (A00+A01)B11
        (([ (0,0,1), (0,1,1) ], [ (1,1,1) ])),
        # P6 = (A10-A00)(B00+B01)
        (([ (1,0,1), (0,0,-1) ], [ (0,0,1), (0,1,1) ])),
        # P7 = (A01-A11)(B10+B11)
        (([ (0,1,1), (1,1,-1) ], [ (1,0,1), (1,1,1) ])),
    ]
    
    U4 = np.zeros((16, 49), dtype=np.int64)
    V4 = np.zeros((16, 49), dtype=np.int64)
    W4 = np.zeros((16, 49), dtype=np.int64)
    
    for big_r, (u_blocks, v_blocks) in enumerate(big_terms):
        for small_r in range(7):
            r = big_r * 7 + small_r
            
            # For each block in u_blocks, add the small_r column of U2
            for (br, bc, coeff) in u_blocks:
                u_indices = block_u_indices(br, bc)
                for idx, ui in enumerate(u_indices):
                    U4[ui, r] = coeff * U2[idx, small_r]
            
            for (br, bc, coeff) in v_blocks:
                v_indices = block_v_indices(br, bc)
                for idx, vi in enumerate(v_indices):
                    V4[vi, r] = coeff * V2[idx, small_r]
            
            # W: the small_r column of W2 applies to the output block
            # The output of big term big_r goes to specific C blocks
            # P1 -> C00 and C11 (from C00=P1+P4-P5+P7, C11=P1-P2+P3+P6)
            # P2 -> C10
            # P3 -> C01 and C11
            # P4 -> C00 and C10
            # P5 -> C00 (neg) and C01
            # P6 -> C11
            # P7 -> C00
            
            # W assignments for each big term:
            w_map = {
                0: [(0,0,1), (1,1,1)],    # P1 -> C00(+), C11(+)
                1: [(1,0,1)],             # P2 -> C10(+)
                2: [(0,1,1), (1,1,1)],    # P3 -> C01(+), C11(+)
                3: [(0,0,1), (1,0,1)],    # P4 -> C00(+), C10(+)
                4: [(0,0,-1), (0,1,1)],   # P5 -> C00(-), C01(+)
                5: [(1,1,1)],             # P6 -> C11(+)
                6: [(0,0,1)],             # P7 -> C00(+)
            }
            
            for (br, bc, coeff) in w_map[big_r]:
                w_indices = block_w_indices(br, bc)
                for idx, wi in enumerate(w_indices):
                    W4[wi, r] = coeff * W2[idx, small_r]
    
    return U4, V4, W4


def _schoolbook(n, m, p):
    """Schoolbook decomposition for n x m * m x p -> n x p, rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    # Split 5x5 into 4x4 + 1x1 blocks
    # A = [[A11, A12], [A21, A22]] where A11 is 4x4, A12 is 4x1, A21 is 1x4, A22 is 1x1
    # B = [[B11, B12], [B21, B22]] similarly
    # C = A*B:
    # C11 = A11*B11 + A12*B21  (4x4)
    # C12 = A11*B12 + A12*B22  (4x1)
    # C21 = A21*B11 + A22*B21  (1x4)
    # C22 = A21*B12 + A22*B22  (1x1)
    
    # Sub-problems:
    # 1. A11*B11 (4x4 * 4x4 -> 4x4): Strassen rank 49
    # 2. A12*B21 (4x1 * 1x4 -> 4x4): Schoolbook rank 4
    # 3. A11*B12 (4x4 * 4x1 -> 4x4... wait, 4x4 * 4x1 -> 4x1): Schoolbook rank 16
    # 4. A12*B22 (4x1 * 1x1 -> 4x1): Schoolbook rank 4
    # 5. A21*B11 (1x4 * 4x4 -> 1x4): Schoolbook rank 16
    # 6. A22*B21 (1x1 * 1x4 -> 1x4): Schoolbook rank 4
    # 7. A21*B12 (1x4 * 4x1 -> 1x1): Schoolbook rank 4
    # 8. A22*B22 (1x1 * 1x1 -> 1x1): Schoolbook rank 1
    
    # Total rank: 49 + 4 + 16 + 4 + 16 + 4 + 4 + 1 = 98
    
    # Build the full 5x5 decomposition by placing each sub-problem in the correct block.
    
    U5 = np.zeros((25, 98), dtype=np.int64)
    V5 = np.zeros((25, 98), dtype=np.int64)
    W5 = np.zeros((25, 98), dtype=np.int64)
    
    col_offset = 0
    
    # Helper to place a sub-decomposition
    def place(U_sub, V_sub, W_sub, u_row_offset, u_col_offset, v_row_offset, v_col_offset, w_row_offset, w_col_offset, rank_sub):
        nonlocal col_offset
        U5[u_row_offset:u_row_offset+U_sub.shape[0], col_offset:col_offset+rank_sub] = U_sub
        V5[v_row_offset:v_row_offset+V_sub.shape[0], col_offset:col_offset+rank_sub] = V_sub
        W5[w_row_offset:w_row_offset+W_sub.shape[0], col_offset:col_offset+rank_sub] = W_sub
        col_offset += rank_sub
    
    # Block positions in 5x5:
    # A11: rows 0-3, cols 0-3 -> U indices: i*5+j for i,j in 0-3
    # A12: rows 0-3, col 4 -> U indices: i*5+4 for i in 0-3
    # A21: row 4, cols 0-3 -> U indices: 4*5+j = 20+j for j in 0-3
    # A22: row 4, col 4 -> U index: 4*5+4 = 24
    
    # B11: rows 0-3, cols 0-3 -> V indices: j*5+k for j,k in 0-3
    # B12: rows 0-3, col 4 -> V indices: j*5+4 for j in 0-3
    # B21: row 4, cols 0-3 -> V indices: 4*5+k = 20+k for k in 0-3
    # B22: row 4, col 4 -> V index: 4*5+4 = 24
    
    # C11: rows 0-3, cols 0-3 -> W indices: k*5+i for k,i in 0-3
    # C12: rows 0-3, col 4 -> W indices: 4*5+i = 20+i for i in 0-3
    # C21: row 4, cols 0-3 -> W indices: k*5+4 for k in 0-3
    # C22: row 4, col 4 -> W index: 4*5+4 = 24
    
    # 1. A11*B11 (4x4 * 4x4 -> 4x4)
    U1, V1, W1 = _strassen_4x4()
    # U indices for A11: i*5+j for i,j in 0-3
    # We need to remap: U1 has 16 rows indexed i*4+j, we need i*5+j
    U1_remapped = np.zeros((16, 49), dtype=np.int64)
    for i in range(4):
        for j in range(4):
            U1_remapped[i*4+j, :] = U1[i*4+j, :]  # same values, just relabel
    # Actually U1 is already in the right format for 4x4, we just need to place it
    # in the correct rows of U5
    for i in range(4):
        for j in range(4):
            U5[i*5+j, col_offset:col_offset+49] = U1[i*4+j, :]
    
    for j in range(4):
        for k in range(4):
            V5[j*5+k, col_offset:col_offset+49] = V1[j*4+k, :]
    
    for k in range(4):
        for i in range(4):
            W5[k*5+i, col_offset:col_offset+49] = W1[k*4+i, :]
    
    col_offset += 49
    
    # 2. A12*B21 (4x1 * 1x4 -> 4x4)
    # A12: 4x1, B21: 1x4, C: 4x4
    # C[i,k] = A12[i] * B21[k] for i,k in 0-3
    # Rank 4: for each j (which is 0 only since A12 is 4x1, B21 is 1x4)
    # Actually A12 has 4 rows, 1 col. B21 has 1 row, 4 cols.
    # C[i,k] = sum_j A12[i,j] * B21[j,k] = A12[i,0] * B21[0,k]
    # This is rank 4 (one for each (i,k) pair? No, it's an outer product, rank 1)
    # Wait, A12 is 4x1, B21 is 1x4, so C = A12 * B21 is 4x4, rank 1 (outer product)
    # But in the bilinear decomposition, we need:
    # C[i,k] = sum_r U[i*1+0, r] * A12[i,0] * V[0*4+k, r] * B21[0,k] * W[k*4+i, r]
    # For rank 1: U[i,0]=1 for all i, V[k,0]=1 for all k, W[k*4+i,0]=1 for all i,k
    # Wait no, that would give C[i,k] = (sum_i A12[i,0]) * (sum_k B21[0,k]) which is wrong.
    
    # Let me reconsider. For A12 (4x1) * B21 (1x4) -> C (4x4):
    # C[i,k] = A12[i,0] * B21[0,k]
    # In bilinear form: C[i,k] = sum_r U[i*1+0, r] * A12[i,0] * V[0*4+k, r] * B21[0,k] * W[k*4+i, r]
    # We need this to equal A12[i,0] * B21[0,k] for all i,k.
    # So U[i, r] * V[k, r] * W[k*4+i, r] should sum to 1 for the correct (i,k) and 0 otherwise.
    # Actually it's: sum_r U[i*1+0, r] * V[0*4+k, r] * W[k*4+i, r] * A12[i,0] * B21[0,k]
    # = A12[i,0] * B21[0,k] * sum_r U[i, r] * V[k, r] * W[k*4+i, r]
    # We need sum_r U[i, r] * V[k, r] * W[k*4+i, r] = 1 for all i,k.
    # This is the all-ones 4x4x4 tensor, which has rank... well, we can use rank 4:
    # For each pair (i,k), one term. But that's 16 terms.
    # Actually, since A12[i,0] and B21[0,k] are separate, we can factor:
    # C[i,k] = A12[i,0] * B21[0,k]
    # This is rank 1 in the bilinear sense? No, because U depends on i, V depends on k, W depends on (k,i).
    # We need: sum_r u_r[i] * v_r[k] * w_r[k,i] = 1 for all i,k.
    # With rank 4: let r = i (0-3), u_r[i'] = delta(i,i'), v_r[k'] = 1, w_r[k,i'] = delta(k,i')? No...
    # Let's just use schoolbook rank 16 for this sub-problem.
    
    U_sub = np.zeros((4, 16), dtype=np.int64)
    V_sub = np.zeros((4, 16), dtype=np.int64)
    W_sub = np.zeros((16, 16), dtype=np.int64)
    c = 0
    for i in range(4):
        for k in range(4):
            U_sub[i, c] = 1
            V_sub[k, c] = 1
            W_sub[k*4+i, c] = 1
            c += 1
    # Place: A12 rows are i*5+4 for i in 0-3, B21 is at row 20 (j=4), cols 0-3
    # U: A12[i,0] -> index i*5+4
    # V: B21[0,k] -> index 4*5+k = 20+k
    # W: C[i,k] -> index k*5+i
    for i in range(4):
        U5[i*5+4, col_offset:col_offset+16] = U_sub[i, :]
    for k in range(4):
        V5[20+k, col_offset:col_offset+16] = V_sub[k, :]
    for k in range(4):
        for i in range(4):
            W5[k*5+i, col_offset:col_offset+16] = W_sub[k*4+i, :]
    col_offset += 16
    
    # 3. A11*B12 (4x4 * 4x1 -> 4x1)
    # A11: 4x4, B12: 4x1, C: 4x1
    # C[i,0] = sum_j A11[i,j] * B12[j,0]
    # Schoolbook rank 16
    U_sub = np.zeros((16, 16), dtype=np.int64)
    V_sub = np.zeros((4, 16), dtype=np.int64)
    W_sub = np.zeros((4, 16), dtype=np.int64)
    c = 0
    for i in range(4):
        for j in range(4):
            U_sub[i*4+j, c] = 1
            V_sub[j, c] = 1
            W_sub[i, c] = 1
            c += 1
    # Place: A11 rows i*5+j for i,j in 0-3, B12 rows j*5+4 for j in 0-3, C rows 20+i for i in 0-3
    for i in range(4):
        for j in range(4):
            U5[i*5+j, col_offset:col_offset+16] = U_sub[i*4+j, :]
    for j in range(4):
        V5[j*5+4, col_offset:col_offset+16] = V_sub[j, :]
    for i in range(4):
        W5[20+i, col_offset:col_offset+16] = W_sub[i, :]
    col_offset += 16
    
    # 4. A12*B22 (4x1 * 1x1 -> 4x1)
    # A12: 4x1, B22: 1x1, C: 4x1
    # C[i,0] = A12[i,0] * B22[0,0]
    # Rank 4
    U_sub = np.zeros((4, 4), dtype=np.int64)
    V_sub = np.zeros((1, 4), dtype=np.int64)
    W_sub = np.zeros((4, 4), dtype=np.int64)
    for i in range(4):
        U_sub[i, i] = 1
        V_sub[0, i] = 1
        W_sub[i, i] = 1
    # Place: A12 rows i*5+4, B22 row 24, C rows 20+i
    for i in range(4):
        U5[i*5+4, col_offset+i] = 1
        V5[24, col_offset+i] = 1
        W5[20+i, col_offset+i] = 1
    col_offset += 4
    
    # 5. A21*B11 (1x4 * 4x4 -> 1x4)
    # A21: 1x4, B11: 4x4, C: 1x4
    # C[0,k] = sum_j A21[0,j] * B11[j,k]
    # Schoolbook rank 16
    U_sub = np.zeros((4, 16), dtype=np.int64)
    V_sub = np.zeros((16, 16), dtype=np.int64)
    W_sub = np.zeros((4, 16), dtype=np.int64)
    c = 0
    for j in range(4):
        for k in range(4):
            U_sub[j, c] = 1
            V_sub[j*4+k, c] = 1
            W_sub[k, c] = 1
            c += 1
    # Place: A21 row 20+j for j in 0-3, B11 rows j*5+k for j,k in 0-3, C rows k*5+4 for k in 0-3
    for j in range(4):
        for k in range(4):
            U5[20+j, col_offset + j*4+k] = 1
            V5[j*5+k, col_offset + j*4+k] = 1
            W5[k*5+4, col_offset + j*4+k] = 1
    col_offset += 16
    
    # 6. A22*B21 (1x1 * 1x4 -> 1x4)
    # A22: 1x1, B21: 1x4, C: 1x4
    # C[0,k] = A22[0,0] * B21[0,k]
    # Rank 4
    for k in range(4):
        U5[24, col_offset+k] = 1
        V5[20+k, col_offset+k] = 1
        W5[k*5+4, col_offset+k] = 1
    col_offset += 4
    
    # 7. A21*B12 (1x4 * 4x1 -> 1x1)
    # A21: 1x4, B12: 4x1, C: 1x1
    # C[0,0] = sum_j A21[0,j] * B12[j,0]
    # Rank 4
    for j in range(4):
        U5[20+j, col_offset+j] = 1
        V5[j*5+4, col_offset+j] = 1
        W5[24, col_offset+j] = 1
    col_offset += 4
    
    # 8. A22*B22 (1x1 * 1x1 -> 1x1)
    # Rank 1
    U5[24, col_offset] = 1
    V5[24, col_offset] = 1
    W5[24, col_offset] = 1
    col_offset += 1
    
    assert col_offset == 98, f"Expected 98, got {col_offset}"
    
    return {"U": U5.tolist(), "V": V5.tolist(), "W": W5.tolist()}

# EVOLVE-BLOCK-END
