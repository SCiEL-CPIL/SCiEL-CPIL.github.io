"""Matrix multiplication 4x4x6 rank-minimal decomposition using Strassen block decomposition."""
import time

DIMENSIONS = [4, 4, 6]

# Strassen rank-7 decomposition for 2x2x2
# U, V, W are 4x7 matrices
def strassen_2x2x2():
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # m1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1  # a11+a22
    V[0][0] = 1; V[3][0] = 1  # b11+b22
    W[0][0] = 1; W[3][0] = 1  # c11+=m1, c22+=m1
    # m2 = a22*(b21-b11)
    U[3][1] = 1
    V[2][1] = 1; V[0][1] = -1
    W[2][1] = 1; W[3][1] = 1  # c21+=m2, c22+=m2
    # m3 = a11*(b22-b12)
    U[0][2] = 1
    V[3][2] = 1; V[1][2] = -1
    W[0][2] = 1; W[1][2] = 1  # c11+=m3, c12+=m3
    # m4 = (a11+a12)*b22
    U[0][3] = 1; U[1][3] = 1
    V[3][3] = 1
    W[1][3] = 1; W[3][3] = 1  # c12+=m4, c22+=m4
    # m5 = (a21+a22)*b11
    U[2][4] = 1; U[3][4] = 1
    V[0][4] = 1
    W[0][4] = 1; W[2][4] = 1  # c11+=m5, c21+=m5
    # m6 = (a11-a21)*(b12+b22)
    U[0][5] = 1; U[2][5] = -1
    V[1][5] = 1; V[3][5] = 1
    W[1][5] = 1; W[2][5] = 1  # c12+=m6, c21+=m6
    # m7 = (a12-a22)*(b11+b21)
    U[1][6] = 1; U[3][6] = -1
    V[0][6] = 1; V[2][6] = 1
    W[1][6] = 1; W[3][6] = 1  # c12+=m7, c22+=m7
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Use 2x2 block decomposition with Strassen
    # A is 4x4, B is 4x6, C is 4x6
    # Split into 2x2 blocks of size 2x2
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12, B13], [B21, B22, B23]]  (each 2x2 block)
    # C = [[C11, C12, C13], [C21, C22, C23]]
    
    # Total rank = 6 * 7 = 42 (for each column group of B, we do 2x2x2 multiplication)
    # Wait, that's not quite right. Let me think again.
    
    # Actually, for 4x4 * 4x6, we can view it as:
    # C = A * B where A is 4x4 and B is 4x6
    # Split A into 2x2 blocks of 2x2
    # Split B into 2x2 blocks (4 rows = 2 blocks of 2, 6 cols = 3 blocks of 2)
    # C will have 2x2 blocks (2 rows = 2 blocks of 2, 6 cols = 3 blocks of 2)
    
    # For each pair of block rows/cols, we compute C_ij = A_ik * B_kj
    # But with Strassen, we can share computations across the block structure
    
    # Simpler approach: just do 6 independent 4x4x2 multiplications using Strassen
    # Each 4x4x2 uses rank 28 (2*2x2x2 blocks * 7 each = 28? No, that's for 4x4x4)
    
    # Let me reconsider. For 4x4 * 4x2:
    # Split into 2x2 blocks: A has 4 blocks of 2x2, B has 2 blocks of 2x2
    # C has 2 blocks of 2x2
    # C1 = A11*B1 + A12*B2
    # C2 = A21*B1 + A22*B2
    # This is just matrix multiplication, no Strassen benefit unless we share
    
    # Actually, the cleanest approach: treat 4x4x6 as 3 copies of 4x4x2
    # And use the known result that R(4,4,2) can be achieved with rank 28 via Strassen-like methods
    
    # But wait - the knowledge says rank 84 was achieved with 12 sub-problems of 2x2x2
    # That's 12 * 7 = 84
    # 12 sub-problems: 2x2 blocks in A (4 blocks) * 3 column blocks in B = 12
    
    # So the approach is:
    # Split A into 2x2 blocks (4 blocks: A11, A12, A21, A22)
    # Split B into 2x2 column blocks (3 blocks: B1, B2, B3, each 4x2)
    # For each (A_block, B_block) pair, compute a 2x2x2 multiplication
    # Total: 4 * 3 = 12 sub-problems, each rank 7, total rank 84
    
    # But this is actually just schoolbook with Strassen applied to each 2x2x2!
    # The benefit comes from using rank 7 instead of rank 8 for each 2x2x2
    
    # Let me implement this properly.
    
    # A is 4x4: rows 0-3, cols 0-3
    # B is 4x6: rows 0-3, cols 0-5
    # C is 4x6: rows 0-3, cols 0-5
    
    # Block structure:
    # A blocks (2x2 each):
    #   A11: rows 0-1, cols 0-1
    #   A12: rows 0-1, cols 2-3
    #   A21: rows 2-3, cols 0-1
    #   A22: rows 2-3, cols 2-3
    
    # B blocks (2x2 each, but B is 4x6, so 3 column blocks):
    #   B1: rows 0-1, cols 0-1
    #   B2: rows 0-1, cols 2-3
    #   B3: rows 0-1, cols 4-5
    #   B4: rows 2-3, cols 0-1
    #   B5: rows 2-3, cols 2-3
    #   B6: rows 2-3, cols 4-5
    
    # C blocks (2x2 each):
    #   C1: rows 0-1, cols 0-1
    #   C2: rows 0-1, cols 2-3
    #   C3: rows 0-1, cols 4-5
    #   C4: rows 2-3, cols 0-1
    #   C5: rows 2-3, cols 2-3
    #   C6: rows 2-3, cols 4-5
    
    # For each C block, we need products of A and B blocks
    # C1 = A11*B1 + A12*B4
    # C2 = A11*B2 + A12*B5
    # C3 = A11*B3 + A12*B6
    # C4 = A21*B1 + A22*B4
    # C5 = A21*B2 + A22*B5
    # C6 = A21*B3 + A22*B6
    
    # Each term like A11*B1 is a 2x2 * 2x2 = 2x2 multiplication
    # We can use Strassen for each, but they're independent
    
    # Total number of 2x2x2 multiplications: 6 C blocks * 2 terms each = 12
    # Each has rank 7, so total rank = 84
    
    # Now let's build the factor matrices.
    
    U_22, V_22, W_22 = strassen_2x2x2()
    
    rank_total = 12 * 7
    U = [[0] * rank_total for _ in range(n * m)]  # 16 x 84
    V = [[0] * rank_total for _ in range(m * p)]  # 24 x 84
    W = [[0] * rank_total for _ in range(p * n)]  # 24 x 84
    
    # Helper to map local 2x2 indices to global indices
    # For A (4x4), block (bi, bj) where bi, bj in {0,1}
    #   local (i, j) -> global row bi*2+i, col bj*2+j
    # For B (4x6), block (bi, bj) where bi in {0,1}, bj in {0,1,2}
    #   local (i, j) -> global row bi*2+i, col bj*2+j
    # For C (4x6), block (ci, cj) where ci in {0,1}, cj in {0,1,2}
    #   local (i, j) -> global row ci*2+i, col cj*2+j
    
    def a_global(bi, bj, i, j):
        """A block (bi,bj), local (i,j) -> global row index in U"""
        return (bi*2 + i) * m + (bj*2 + j)
    
    def b_global(bi, bj, i, j):
        """B block (bi,bj), local (i,j) -> global row index in V"""
        return (bi*2 + i) * p + (bj*2 + j)
    
    def c_global(ci, cj, i, j):
        """C block (ci,cj), local (i,j) -> global col index in W (k*n+i format)"""
        # W[k*n + i, c] where k is col, i is row of C
        return (cj*2 + j) * n + (ci*2 + i)
    
    # Define the 12 sub-problems:
    # Each is (a_block, b_block, c_block)
    # a_block: (bi, bj) for A
    # b_block: (bi, bj) for B  
    # c_block: (ci, cj) for C
    
    subproblems = [
        # C1 = A11*B1 + A12*B4
        ((0, 0), (0, 0), (0, 0)),  # A11 * B1 -> C1
        ((0, 1), (1, 0), (0, 0)),  # A12 * B4 -> C1
        # C2 = A11*B2 + A12*B5
        ((0, 0), (0, 1), (0, 1)),  # A11 * B2 -> C2
        ((0, 1), (1, 1), (0, 1)),  # A12 * B5 -> C2
        # C3 = A11*B3 + A12*B6
        ((0, 0), (0, 2), (0, 2)),  # A11 * B3 -> C3
        ((0, 1), (1, 2), (0, 2)),  # A12 * B6 -> C3
        # C4 = A21*B1 + A22*B4
        ((1, 0), (0, 0), (1, 0)),  # A21 * B1 -> C4
        ((1, 1), (1, 0), (1, 0)),  # A22 * B4 -> C4
        # C5 = A21*B2 + A22*B5
        ((1, 0), (0, 1), (1, 1)),  # A21 * B2 -> C5
        ((1, 1), (1, 1), (1, 1)),  # A22 * B5 -> C5
        # C6 = A21*B3 + A22*B6
        ((1, 0), (0, 2), (1, 2)),  # A21 * B3 -> C6
        ((1, 1), (1, 2), (1, 2)),  # A22 * B6 -> C6
    ]
    
    for sp_idx, (ab, bb, cb) in enumerate(subproblems):
        base_col = sp_idx * 7
        for c_local in range(7):
            col = base_col + c_local
            for i_local in range(2):
                for j_local in range(2):
                    # U contribution
                    u_val = U_22[i_local * 2 + j_local][c_local]
                    if u_val != 0:
                        gi = ab[0] * 2 + i_local
                        gj = ab[1] * 2 + j_local
                        U[gi * m + gj][col] = u_val
                    
                    # V contribution
                    v_val = V_22[i_local * 2 + j_local][c_local]
                    if v_val != 0:
                        gi = bb[0] * 2 + i_local
                        gj = bb[1] * 2 + j_local
                        V[gi * p + gj][col] = v_val
                    
                    # W contribution
                    w_val = W_22[i_local * 2 + j_local][c_local]
                    if w_val != 0:
                        gi = cb[0] * 2 + i_local
                        gj = cb[1] * 2 + j_local
                        W[(gj) * n + gi][col] = w_val
    
    # Check deadline
    if time.time() > deadline:
        # Shouldn't happen for this fast construction, but safety guard
        pass
    
    return {"U": U, "V": V, "W": W}
