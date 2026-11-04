"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    deadline = time.time() + 140  # safety margin before 150s kill
    
    def _strassen_222():
        """Return U,V,W for 2x2x2 Strassen (rank 7)."""
        # Strassen 7 products for (A11 A12; A21 A22) * (B11 B12; B21 B22)
        # A is 2x2, B is 2x2, C = A*B is 2x2
        # Indexing: A[i,j], B[j,k], C[i,k]
        # U: maps A entries -> products, shape (4,7)
        # V: maps B entries -> products, shape (4,7)
        # W: maps C entries -> products, shape (4,7)
        # Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        # For n=2,m=2,p=2: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # Strassen products:
        # P1 = (A11 + A22)(B11 + B22)
        # P2 = (A21 + A22) B11
        # P3 = A11 (B12 - B22)
        # P4 = A22 (B21 - B11)
        # P5 = (A11 + A12) B22
        # P6 = (A21 - A11) (B11 + B12)
        # P7 = (A12 - A22) (B21 + B22)
        
        # U rows: A11(0), A12(1), A21(2), A22(3)
        # V rows: B11(0), B12(1), B21(2), B22(3)
        # W rows: C11(0), C12(1), C21(2), C22(3)
        
        # P1
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[3,0] = 1
        # P2
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[2,1] = 1; W[3,1] = -1
        # P3
        U[0,2] = 1
        V[1,2] = 1; V[3,2] = -1
        W[1,2] = 1
        # P4
        U[3,3] = 1
        V[2,3] = 1; V[0,3] = -1
        W[2,3] = 1
        # P5
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        W[1,4] = 1; W[3,4] = 1
        # P6
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[0,5] = 1; W[2,5] = 1
        # P7
        U[1,6] = 1; U[3,6] = -1
        V[2,6] = 1; V[3,6] = 1
        W[0,6] = 1; W[1,6] = 1
        
        return U.tolist(), V.tolist(), W.tolist()
    
    def _schoolbook(n_s, m_s, p_s):
        """Return U,V,W for schoolbook multiplication of n_s x m_s by m_s x p_s."""
        rank = n_s * m_s * p_s
        U = [[0] * rank for _ in range(n_s * m_s)]
        V = [[0] * rank for _ in range(m_s * p_s)]
        W = [[0] * rank for _ in range(p_s * n_s)]
        col = 0
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    U[i*m_s+j][col] = 1
                    V[j*p_s+k][col] = 1
                    W[k*n_s+i][col] = 1
                    col += 1
        return U, V, W
    
    # Strategy: Use Strassen on 2x2x2 blocks where possible.
    # For 2x3x10:
    # Partition m=3 into 2+1, p=10 into 2+2+2+2+2
    # For each (n_block, m_block, p_block):
    #   n_block is always 2 (since n=2)
    #   m_block can be 2 or 1
    #   p_block is 2
    
    # Block structure:
    # m=3 = 2 + 1
    # p=10 = 2+2+2+2+2
    # 
    # For m_block=2, p_block=2: use Strassen (rank 7)
    # For m_block=1, p_block=2: use schoolbook (rank 2*1*2 = 4)
    #
    # Total blocks: 2 (for m=2) * 5 (for p=2 each) + 1 (for m=1) * 5 (for p=2 each)
    # = 10 Strassen blocks + 5 schoolbook blocks
    # Rank = 10*7 + 5*4 = 70 + 20 = 90. That's worse than 60!
    
    # Better approach: Use Strassen on 2x2x2 sub-blocks of the 2x2x10 part.
    # n=2, m=2, p=10: Use Strassen for each 2x2x2 block of p.
    # p=10 = 5 blocks of size 2.
    # Rank for 2x2x10 with Strassen: 5 * 7 = 35
    # Plus the remaining m=1 part: 2x1x10 = schoolbook rank 20
    # Total: 35 + 20 = 55
    
    # Let's implement this:
    # U has shape (n*m, rank) = (6, 55)
    # V has shape (m*p, rank) = (30, 55)
    # W has shape (p*n, rank) = (20, 55)
    
    rank_total = 55
    U = [[0] * rank_total for _ in range(n*m)]  # 6 x 55
    V = [[0] * rank_total for _ in range(m*p)]  # 30 x 55
    W = [[0] * rank_total for _ in range(p*n)]  # 20 x 55
    
    col = 0
    # Part 1: n=2, m=2, p=10 using Strassen for each 2x2x2 block
    # m indices: 0,1 (first two columns of the 3x10 matrix)
    # p indices: 0..9 (all columns)
    for p_block in range(5):
        p_start = p_block * 2
        S_U, S_V, S_W = _strassen_222()
        for r in range(7):
            for i in range(4):
                if S_U[i][r] != 0:
                    U[i][col] += S_U[i][r]
            for j in range(4):
                if S_V[j][r] != 0:
                    # j corresponds to row in B (m x p block)
                    # m index = j // 2 (0 or 1), p index = p_start + j % 2
                    m_idx = j // 2
                    p_idx = p_start + j % 2
                    V[m_idx * p + p_idx][col] += S_V[j][r]
            for k in range(4):
                if S_W[k][r] != 0:
                    # k corresponds to C entry
                    # C[i,k] where i is row of A (n=2), k is col of B (p block)
                    # W index: k*n + i -> but convention is W[k*n+i, c]
                    # k here is the p-index within the block (0 or 1)
                    # Actual p index = p_start + k%2? No.
                    # Let me re-examine. In the 2x2x2 case:
                    # C[i,j] where i=0,1 (n dim), j=0,1 (p dim)
                    # W[j*n + i] = W[j*2 + i]
                    # So for block p_start, W[(p_start + j%2)*n + i]
                    # But k in S_W is indexed 0..3 as k*n+i where k=0,1 (p dim within block)
                    # So k = j*2 + i where j is p-dim index within block, i is n-dim index
                    n_idx = k % 2
                    p_idx_within = k // 2
                    p_idx = p_start + p_idx_within
                    W[p_idx * n + n_idx][col] += S_W[k][r]
            col += 1
    
    # Part 2: n=2, m=1, p=10 using schoolbook
    # m index = 2 (third column)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1  # i*m + 2 since m=3, row for (i, m_idx=2)
            V[2 * p + k][col] = 1  # m_idx=2, p_idx=k
            W[k * n + i][col] = 1
            col += 1
    
    # Verify col == rank_total
    assert col == rank_total, f"Expected {rank_total}, got {col}"
    
    # Convert to list of lists (already are)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
