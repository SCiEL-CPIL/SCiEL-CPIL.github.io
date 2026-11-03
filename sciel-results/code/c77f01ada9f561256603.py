"""Improved program: block-Strassen decomposition for 4x5x5 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 5]
_DEADLINE_SECONDS = 115  # safety margin before 120s kill

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return (U, V, W) as lists of lists, rank n*m*p."""
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
    """Strassen's 7-product algorithm for 2x2x2.
    Returns U (4x7), V (4x7), W (4x7) as lists of lists (integers).
    A is 2x2 flattened row-major: (a,b,c,d) -> indices 0,1,2,3
    B is 2x2 flattened row-major: (e,f,g,h) -> indices 0,1,2,3
    C = A*B, W index = k*2 + i (n=2, p=2).
    W[0] -> i=0,k=0: C[0][0]
    W[1] -> i=0,k=1: C[0][1]
    W[2] -> i=1,k=0: C[1][0]
    W[3] -> i=1,k=1: C[1][1]
    """
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, -1, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V = [
        [1, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 1]
    ]
    W = [
        [1, 0, 0, -1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, -1, 0, 0, 1, 0]
    ]
    return U, V, W


def _schoolbook_block(n, m, p):
    """Schoolbook decomposition for n x m times m x p = n x p.
    Returns U (n*m x n*m*p), V (m*p x n*m*p), W (p*n x n*m*p) as lists of lists.
    U row index: i*m + j
    V row index: j*p + k
    W row index: k*n + i
    """
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


def _embed_block_U(block_U, n_block, m_block, i_offset, j_offset, n_total, m_total, rank_total):
    """Embed a block U (n_block*m_block x rank_block) into full U (n_total*m_total x rank_total).
    block_U rows correspond to A[i_offset:r, j_offset:c] where r=i_offset+n_block, c=j_offset+m_block.
    In full U, row index = i*m_total + j for i in [i_offset, i_offset+n_block), j in [j_offset, j_offset+m_block).
    """
    rank_block = len(block_U[0])
    full_U = [[0] * rank_total for _ in range(n_total * m_total)]
    for bi in range(n_block):
        for bj in range(m_block):
            full_row = (i_offset + bi) * m_total + (j_offset + bj)
            block_row = bi * m_block + bj
            for c in range(rank_block):
                full_U[full_row][c] = block_U[block_row][c]
    return full_U


def _embed_block_V(block_V, m_block, p_block, j_offset, k_offset, m_total, p_total, rank_total):
    """Embed a block V (m_block*p_block x rank_block) into full V (m_total*p_total x rank_total).
    block_V rows correspond to B[j_offset:r, k_offset:c] where r=j_offset+m_block, c=k_offset+p_block.
    In full V, row index = j*p_total + k.
    """
    rank_block = len(block_V[0])
    full_V = [[0] * rank_total for _ in range(m_total * p_total)]
    for bj in range(m_block):
        for bk in range(p_block):
            full_row = (j_offset + bj) * p_total + (k_offset + bk)
            block_row = bj * p_block + bk
            for c in range(rank_block):
                full_V[full_row][c] = block_V[block_row][c]
    return full_V


def _embed_block_W(block_W, p_block, n_block, k_offset, i_offset, p_total, n_total, rank_total):
    """Embed a block W (p_block*n_block x rank_block) into full W (p_total*n_total x rank_total).
    block_W rows correspond to C[i_offset:r, k_offset:c] where r=i_offset+n_block, c=k_offset+p_block.
    In full W, row index = k*n_total + i.
    """
    rank_block = len(block_W[0])
    full_W = [[0] * rank_total for _ in range(p_total * n_total)]
    for bk in range(p_block):
        for bi in range(n_block):
            full_row = (k_offset + bk) * n_total + (i_offset + bi)
            block_row = bk * n_block + bi
            for c in range(rank_block):
                full_W[full_row][c] = block_W[block_row][c]
    return full_W


def _add_matrices(A, B):
    """Add two matrices (lists of lists) of same shape."""
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def _strassen_2x2x5():
    """Strassen-based decomposition for 2x2 * 2x5 = 2x5.
    Split B (2x5) into B1 (2x2) and B2 (2x3).
    A*B = A*B1 + A*B2.
    A*B1: 2x2*2x2, Strassen rank 7.
    A*B2: 2x2*2x3, schoolbook rank 12.
    Total rank: 19.
    
    A: 2x2, indices (0,1,2,3) row-major.
    B: 2x5, indices (0..9) row-major. B1 = cols 0-1, B2 = cols 2-4.
    C: 2x5, W index = k*2 + i (n=2, p=5).
    W[0]=C[0,0], W[1]=C[0,1], W[2]=C[0,2], W[3]=C[0,3], W[4]=C[0,4],
    W[5]=C[1,0], W[6]=C[1,1], W[7]=C[1,2], W[8]=C[1,3], W[9]=C[1,4].
    """
    sU, sV, sW = _strassen_2x2()
    
    # A*B1 part (cols 0-1 of B, which are indices 0,1,4,5 in B's 2x5 flattening)
    # In B (2x5), B1 = rows 0-1, cols 0-1: indices 0,1 (row0) and 4,5 (row1)
    # B1 flattened 2x2 row-major: B[0,0]=idx0, B[0,1]=idx1, B[1,0]=idx4, B[1,1]=idx5
    # Strassen V expects 4 entries in order (e,f,g,h) = (B[0,0],B[0,1],B[1,0],B[1,1])
    # So V row for B1: maps sV rows 0,1,2,3 to B indices 0,1,4,5
    
    # A*B2 part (cols 2-4 of B, indices 2,3,4,7,8,9 in B's 2x5 flattening)
    # B2 = rows 0-1, cols 2-4: 
    #   row 0: B[0,2]=idx2, B[0,3]=idx3, B[0,4]=idx4... wait B is 2x5 so indices 0..9
    #   B row 0: cols 0,1,2,3,4 -> indices 0,1,2,3,4
    #   B row 1: cols 0,1,2,3,4 -> indices 5,6,7,8,9
    # B2 = cols 2,3,4: row0: idx 2,3,4; row1: idx 7,8,9
    # B2 flattened 2x3 row-major: (B[0,2],B[0,3],B[0,4],B[1,2],B[1,3],B[1,4]) = (2,3,4,7,8,9)
    
    rank_total = 19
    U = [[0] * rank_total for _ in range(4)]  # 2*2
    V = [[0] * rank_total for _ in range(10)]  # 2*5
    W = [[0] * rank_total for _ in range(10)]  # 5*2
    
    # Part 1: A*B1 (Strassen, cols 0-1)
    for i in range(4):
        for c in range(7):
            U[i][c] = sU[i][c]
    # Map sV rows to B indices: sV row 0 -> B idx 0, row 1 -> B idx 1, row 2 -> B idx 4, row 3 -> B idx 5
    b1_map = [0, 1, 4, 5]
    for si in range(4):
        for c in range(7):
            V[b1_map[si]][c] = sV[si][c]
    # sW rows: 0->C[0,0], 1->C[0,1], 2->C[1,0], 3->C[1,1]
    # W indices: C[0,0]=0, C[0,1]=1, C[1,0]=5, C[1,1]=6
    w1_map = [0, 1, 5, 6]
    for swi in range(4):
        for c in range(7):
            W[w1_map[swi]][c] = sW[swi][c]
    
    # Part 2: A*B2 (schoolbook, cols 2-4), offset by 7
    # A rows: 0,1 (A[0,0]=U row 0, A[0,1]=U row 1, A[1,0]=U row 2, A[1,1]=U row 3)
    # B2: 2x3, flattened indices in B: 2,3,4 (row0), 7,8,9 (row1)
    # B2 flattened 2x3: (B[0,2],B[0,3],B[0,4],B[1,2],B[1,3],B[1,4])
    # V row for B2[j,k]: j*p_block+k where p_block=3, so:
    #   B2[0,0]->B idx 2, B2[0,1]->B idx 3, B2[0,2]->B idx 4
    #   B2[1,0]->B idx 7, B2[1,1]->B idx 8, B2[1,2]->B idx 9
    # C output: 2x3, C[i,k] for i=0,1 and k=2,3,4
    # W indices: k*2+i: k=2,i=0->4; k=3,i=0->7; k=4,i=0->10... 
    # Wait, p=5 for the full 2x5 problem, so W index = k*2+i where k is column index in the 2x5 output.
    # C has 5 columns (0-4). W index for C[i,k] = k*2+i.
    # C[0,2]=4, C[0,3]=7, C[0,4]=10... but W only has 10 rows (5*2=10), indices 0-9.
    # k=4, i=0: 4*2+0=8; k=4,i=1: 4*2+1=9. OK.
    # C[1,2]=4*1+... wait let me recompute. W index = k*n + i = k*2 + i.
    # C[0,0]=0, C[0,1]=1, C[0,2]=2, C[0,3]=3, C[0,4]=4
    # C[1,0]=5, C[1,1]=6, C[1,2]=7, C[1,3]=8, C[1,4]=9
    # So C[0,2]=2, C[0,3]=3, C[0,4]=4; C[1,2]=7, C[1,3]=8, C[1,4]=9
    
    col = 7
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i * 2 + j][col] = 1  # A[i,j]
                # B2[j, k] corresponds to B row j, col (2+k)
                V[j * 5 + (2 + k)][col] = 1  # B[j, 2+k]
                W[(2 + k) * 2 + i][col] = 1  # C[i, 2+k]
                col += 1
    
    return U, V, W


def _strassen_2x3x5():
    """Decomposition for 2x3 * 3x5 = 2x5.
    Schoolbook rank: 2*3*5 = 30.
    Split B (3x5) into B1 (3x2) and B2 (3x3).
    A*B1: 2x3 * 3x2, schoolbook rank 12.
    A*B2: 2x3 * 3x3, schoolbook rank 18.
    Total: 30. No improvement from this split.
    
    Alternative: Split A (2x3) into A1 (2x2) and A2 (2x1).
    B (3x5) into B1 (2x5) and B2 (1x5).
    A1*B1: 2x2*2x5, Strassen-based rank 19 (from _strassen_2x2x5).
    A2*B2: 2x1*1x5, rank 10.
    Cross terms: A1*B2 (2x2*1x5, rank 10) and A2*B1 (2x1*2x5, rank 10).
    Total: 19+10+10+10 = 49. Worse.
    
    Just use schoolbook rank 30.
    """
    return _schoolbook_block(2, 3, 5)


def _construct_4x5x5():
    """Build 4x5x5 decomposition using block Strassen.
    
    A: 4x5, B: 5x5, C: 4x5.
    Split A rows into two 2x5 blocks: A_top (rows 0-1), A_bot (rows 2-3).
    Split B rows into B1 (rows 0-1) and B2 (rows 2-4).
    
    C_top (2x5) = A_top * B = A_top*B1 + A_top*B2
    C_bot (2x5) = A_bot * B = A_bot*B1 + A_bot*B2
    
    A_top*B1: 2x5 * 2x5. Split B1 cols into 2+3: A_top*B1_left (2x5*2x2) + A_top*B1_right (2x5*2x3).
      A_top*B1_left: 2x5*2x2. Split A_top cols into 2+3: 
        A_top_left*B1_left: 2x2*2x2, Strassen rank 7.
        A_top_right*B1_left: 2x3*2x2, schoolbook rank 12.
      A_top*B1_right: 2x5*2x3. Split A_top cols into 2+3:
        A_top_left*B1_right: 2x2*2x3, schoolbook rank 12.
        A_top_right*B1_right: 2x3*2x3, schoolbook rank 18.
      Total for A_top*B1: 7+12+12+18 = 49.
    
    A_top*B2: 2x5 * 3x5. Schoolbook rank 30.
      Split A_top cols into 2+3:
        A_top_left*B2: 2x2*3x5, schoolbook rank 30.
        A_top_right*B2: 2x3*3x5, schoolbook rank 30.
      Total: 60. Worse than 30.
    
    This is getting complex. Let me use a simpler approach:
    Split the summation index j (0-4) into j=0,1 and j=2,3,4.
    
    C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{4} A[i,j]*B[j,k]
    
    Part 1: A_left (4x2) * B_top (2x5) -> 4x5
      Split A_left rows into 2+2: A_top_left (2x2), A_bot_left (2x2)
      Split B_top cols into 2+3: B_top_left (2x2), B_top_right (2x3)
      A_top_left*B_top_left: 2x2*2x2, Strassen rank 7
      A_top_left*B_top_right: 2x2*2x3, schoolbook rank 12
      A_bot_left*B_top_left: 2x2*2x2, Strassen rank 7
      A_bot_left*B_top_right: 2x2*2x3, schoolbook rank 12
      Total Part 1: 38
    
    Part 2: A_right (4x3) * B_bot (3x5) -> 4x5
      Schoolbook rank: 4*3*5 = 60.
      Split A_right rows into 2+2:
        A_top_right (2x3) * B_bot (3x5): schoolbook rank 30
        A_bot_right (2x3) * B_bot (3x5): schoolbook rank 30
      Total Part 2: 60
    
    Grand total: 38 + 60 = 98.
    
    Let me build this explicitly.
    """
    n, m, p = 4, 5, 5
    total_rank = 98
    
    U = [[0] * total_rank for _ in range(n * m)]  # 20 x 98
    V = [[0] * total_rank for _ in range(m * p)]  # 25 x 98
    W = [[0] * total_rank for _ in range(p * n)]  # 20 x 98
    
    col = 0
    
    # Part 1: A_left (4x2) * B_top (2x5)
    # A_left: A[i,j] for i=0..3, j=0..1 -> U rows i*5+j
    # B_top: B[j,k] for j=0..1, k=0..4 -> V rows j*5+k
    # Output: C[i,k] for i=0..3, k=0..4 -> W rows k*4+i
    
    # Sub-part 1a: A_top_left (2x2) * B_top_left (2x2) - Strassen
    # A_top_left: A[0,0],A[0,1],A[1,0],A[1,1] -> U rows 0,1,5,6
    # B_top_left: B[0,0],B[0,1],B[1,0],B[1,1] -> V rows 0,1,5,6
    # Output: C[0,0],C[0,1],C[1,0],C[1,1] -> W rows 0,1,4,5
    
    sU, sV, sW = _strassen_2x2()
    a_rows = [0, 1, 5, 6]  # U rows for A_top_left
    b_rows = [0, 1, 5, 6]  # V rows for B_top_left
    c_rows = [0, 1, 4, 5]  # W rows for C[0,0],C[0,1],C[1,0],C[1,1]
    for i in range(4):
        for c in range(7):
            U[a_rows[i]][col + c] = sU[i][c]
            V[b_rows[i]][col + c] = sV[i][c]
            W[c_rows[i]][col + c] = sW[i][c]
    col += 7
    
    # Sub-part 1b: A_top_left (2x2) * B_top_right (2x3) - schoolbook
    # A_top_left: U rows 0,1,5,6
    # B_top_right: B[0,2],B[0,3],B[0,4],B[1,2],B[1,3],B[1,4] -> V rows 2,3,4,7,8,9
    # Output: C[0,2],C[0,3],C[0,4],C[1,2],C[1,3],C[1,4] -> W rows 2,3,4... 
    # W index = k*4+i: C[0,2]=2*4+0=8, C[0,3]=3*4+0=12, C[0,4]=4*4+0=16
    # C[1,2]=2*4+1=9, C[1,3]=3*4+1=13, C[1,4]=4*4+1=17
    a_rows = [0, 1, 5, 6]  # A_top_left: (0,0),(0,1),(1,0),(1,1)
    b_rows = [2, 3, 4, 7, 8, 9]  # B_top_right: (0,2),(0,3),(0,4),(1,2),(1,3),(1,4)
    # C output rows: for i in {0,1}, k in {2,3,4}: W[k*4+i]
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[a_rows[i * 2 + j]][col] = 1
                V[b_rows[j * 3 + k]][col] = 1
                W[(2 + k) * 4 + i][col] = 1
                col += 1
    # col now at 7 + 12 = 19
    
    # Sub-part 1c: A_bot_left (2x2) * B_top_left (2x2) - Strassen
    # A_bot_left: A[2,0],A[2,1],A[3,0],A[3,1] -> U rows 10,11,15,16
    # B_top_left: V rows 0,1,5,6
    # Output: C[2,0],C[2,1],C[3,0],C[3,1] -> W rows 0*4+2=8, 1*4+2=10, 0*4+3=12, 1*4+3=14
    
    sU, sV, sW = _strassen_2x2()
    a_rows = [10, 11, 15, 16]
    b_rows = [0, 1, 5, 6]
    c_rows = [8, 10, 12, 14]  # C[2,0]=8, C[2,1]=10, C[3,0]=12, C[3,1]=14
    for i in range(4):
        for c in range(7):
            U[a_rows[i]][col + c] = sU[i][c]
            V[b_rows[i]][col + c] = sV[i][c]
            W[c_rows[i]][col + c] = sW[i][c]
    col += 7
    # col now at 19 + 7 = 26
    
    # Sub-part 1d: A_bot_left (2x2) * B_top_right (2x3) - schoolbook
    # A_bot_left: U rows 10,11,15,16
    # B_top_right: V rows 2,3,4,7,8,9
    # Output: C[2,2],C[2,3],C[2,4],C[3,2],C[3,3],C[3,4]
    # W: C[2,2]=2*4+2=10, C[2,3]=3*4+2=14, C[2,4]=4*4+2=18
    # C[3,2]=2*4+3=11, C[3,3]=3*4+3=15, C[3,4]=4*4+3=19
    a_rows = [10, 11, 15, 16]
    b_rows = [2, 3, 4, 7, 8, 9]
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[a_rows[i * 2 + j]][col] = 1
                V[b_rows[j * 3 + k]][col] = 1
                W[(2 + k) * 4 + (2 + i)][col] = 1
                col += 1
    # col now at 26 + 12 = 38
    
    # Part 2: A_right (4x3) * B_bot (3x5) - schoolbook
    # A_right: A[i,j] for i=0..3, j=2..4 -> U rows i*5+j
    # B_bot: B[j,k] for j=2..4, k=0..4 -> V rows j*5+k
    # Output: C[i,k] for i=0..3, k=0..4 -> W rows k*4+i
    
    for i in range(4):
        for j in range(3):
            for k in range(5):
                U[i * 5 + (2 + j)][col] = 1
                V[(2 + j) * 5 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1
    # col now at 38 + 60 = 98
    
    assert col == total_rank, f"Expected {total_rank}, got {col}"
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, num_trials=10):
    """Verify that the decomposition correctly computes matrix multiplication."""
    np.random.seed(42)
    rank = len(W[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for trial in range(num_trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute C from decomposition
        # C[i,k] = sum_r U[i*m+k... no
        # U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank
        # C[i,k] = sum_r (sum_j U[i*m+j] * V[j*p+k]... no
        # The decomposition is: T = sum_r U_r ⊗ V_r ⊗ W_r
        # where (U_r ⊗ V_r ⊗ W_r)[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # So C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for r in range(rank):
                        C_computed[i, k] += U_np[i * m + j, r] * V_np[j * p + k, r] * W_np[k * n + i, r]
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True


def construct():
    """Construct the rank-minimal exact tensor decomposition for 4x5x5 matrix multiplication."""
    t0 = time.time()
    
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Build the block-Strassen decomposition (rank 98)
    U, V, W = _construct_4x5x5()
    rank = len(W[0])
    
    # Verify correctness
    if time.time() - t0 < _DEADLINE_SECONDS - 10:
        if not _verify_decomposition(U, V, W, n, m, p, num_trials=5):
            # Fallback to schoolbook if verification fails
            U, V, W = _schoolbook(n, m, p)
    
    # Check deadline
    if time.time() - t0 > _DEADLINE_SECONDS:
        U, V, W = _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
