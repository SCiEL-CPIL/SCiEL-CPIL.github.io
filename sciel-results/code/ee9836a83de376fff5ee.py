"""Find a rank-minimal exact tensor decomposition for 2x3 * 3x7 matrix multiplication."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]
SEED = 42

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit
    
    def make_strassen_2x2x2():
        """Return (U, V, W) for 2x2 * 2x2 using Strassen's 7 multiplications.
        Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        """
        # Strassen's algorithm for 2x2:
        # M1 = (A11+A22)*(B11+B22)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B12-B22)
        # M4 = A22*(B21-B11)
        # M5 = (A11+A12)*B22
        # M6 = (A21-A11)*(B11+B12)
        # M7 = (A12-A22)*(B21+B22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        rank = 7
        U = [[0]*rank for _ in range(4)]  # n*m = 4
        V = [[0]*rank for _ in range(4)]  # m*p = 4
        W = [[0]*rank for _ in range(4)]  # p*n = 4
        
        # For each multiplication c, we need:
        # U[i*2+j, c] = coefficient of A[i,j]
        # V[j*2+k, c] = coefficient of B[j,k]
        # W[k*2+i, c] = coefficient in C[i,k]
        
        # M1: (A11+A22)*(B11+B22)
        # A factors: A11=1, A22=1 -> U[0,0]=1, U[3,0]=1 (i=0,j=0; i=1,j=1)
        # B factors: B11=1, B22=1 -> V[0,0]=1, V[3,0]=1
        # C: C11+=1, C22+=1 -> W[0,0]=1, W[3,0]=1
        U[0][0] += 1; U[3][0] += 1
        V[0][0] += 1; V[3][0] += 1
        W[0][0] += 1; W[3][0] += 1
        
        # M2: (A21+A22)*B11
        # A: A21=1, A22=1 -> U[2,1]=1, U[3,1]=1
        # B: B11=1 -> V[0,1]=1
        # C: C21+=1 -> W[2,1]=1 (k=1,i=1 => index k*2+i = 3? Wait)
        # W[k*n+i, c]: for C21, i=1, k=1 => W[1*2+1]=W[3]
        # Hmm wait. C[i,k] where i is row of result, k is column.
        # C21 means i=1, k=1 (0-indexed). W[k*n+i] = W[1*2+1] = W[3]
        U[2][1] += 1; U[3][1] += 1
        V[0][1] += 1
        W[3][1] += 1  # C21: k=1, i=1
        
        # M3: A11*(B12-B22)
        # A: A11=1 -> U[0,2]=1
        # B: B12=1, B22=-1 -> V[1,2]=1, V[3,2]=-1
        # C: C12+=1 -> i=0, k=1 => W[1*2+0]=W[2]
        U[0][2] += 1
        V[1][2] += 1; V[3][2] += -1
        W[2][2] += 1
        
        # M4: A22*(B21-B11)
        # A: A22=1 -> U[3,3]=1
        # B: B21=1, B11=-1 -> V[2,3]=1, V[0,3]=-1
        # C: C11+=1, C21+=1 -> W[0,3]=1 (k=0,i=0), W[3,3]=1 (k=1,i=1)
        U[3][3] += 1
        V[2][3] += 1; V[0][3] += -1
        W[0][3] += 1; W[3][3] += 1
        
        # M5: (A11+A12)*B22
        # A: A11=1, A12=1 -> U[0,4]=1, U[1,4]=1
        # B: B22=1 -> V[3,4]=1
        # C: C11-=1, C12+=1 -> W[0,4]=-1 (k=0,i=0), W[2,4]=1 (k=1,i=0)
        U[0][4] += 1; U[1][4] += 1
        V[3][4] += 1
        W[0][4] += -1; W[2][4] += 1
        
        # M6: (A21-A11)*(B11+B12)
        # A: A21=1, A11=-1 -> U[2,5]=1, U[0,5]=-1
        # B: B11=1, B12=1 -> V[0,5]=1, V[1,5]=1
        # C: C22+=1 -> W[3,6]=1 (k=1,i=1)
        U[2][5] += 1; U[0][5] += -1
        V[0][5] += 1; V[1][5] += 1
        W[3][5] += 1
        
        # M7: (A12-A22)*(B21+B22)
        # A: A12=1, A22=-1 -> U[1,6]=1, U[3,6]=-1
        # B: B21=1, B22=1 -> V[2,6]=1, V[3,6]=1
        # C: C11+=1 -> W[0,6]=1 (k=0,i=0)
        U[1][6] += 1; U[3][6] += -1
        V[2][6] += 1; V[3][6] += 1
        W[0][6] += 1
        
        return U, V, W
    
    def schoolbook_block(n1, m1, p1):
        """Return (U, V, W) for schoolbook n1 x m1 * m1 x p1."""
        rank = n1 * m1 * p1
        U = [[0]*rank for _ in range(n1*m1)]
        V = [[0]*rank for _ in range(m1*p1)]
        W = [[0]*rank for _ in range(p1*n1)]
        col = 0
        for i in range(n1):
            for j in range(m1):
                for k in range(p1):
                    U[i*m1+j][col] = 1
                    V[j*p1+k][col] = 1
                    W[k*n1+i][col] = 1
                    col += 1
        return U, V, W
    
    def combine_decompositions(blocks):
        """Combine list of (U, V, W) with row/col offsets into one decomposition.
        blocks: list of dicts with keys 'U','V','W','row_offset_U','col_offset_U',
                'row_offset_V','col_offset_V','row_offset_W','col_offset_W'
        """
        total_rank = sum(len(b['W']) for b in blocks)  # rank = number of columns in W
        # Actually rank = len(b['U'][0]) for each block, but they should be consistent
        # Let me reconsider: each block has its own rank. Total rank = sum of ranks.
        
        n_total = n  # 2
        m_total = m  # 3  
        p_total = p  # 7
        
        U_full = [[0]*total_rank for _ in range(n_total*m_total)]
        V_full = [[0]*total_rank for _ in range(m_total*p_total)]
        W_full = [[0]*total_rank for _ in range(p_total*n_total)]
        
        col_offset = 0
        for b in blocks:
            rank_b = len(b['U'][0])
            U_b = b['U']
            V_b = b['V']
            W_b = b['W']
            ro_U = b['row_offset_U']  # offset in U rows (i*m+j indexing)
            co_U = b['col_offset_U']  # offset in U cols (j index within block)
            ro_V = b['row_offset_V']  # offset in V rows (j*p+k indexing)
            co_V = b['col_offset_V']  # offset in V cols (k index within block)
            ro_W = b['row_offset_W']  # offset in W rows (k*n+i indexing)
            co_W = b['col_offset_W']  # offset in W cols (i index within block)
            
            for ri in range(len(U_b)):
                for c in range(rank_b):
                    U_full[ro_U + ri][col_offset + c] = U_b[ri][c]
            for ri in range(len(V_b)):
                for c in range(rank_b):
                    V_full[ro_V + ri][col_offset + c] = V_b[ri][c]
            for ri in range(len(W_b)):
                for c in range(rank_b):
                    W_full[ro_W + ri][col_offset + c] = W_b[ri][c]
            col_offset += rank_b
        
        return U_full, V_full, W_full
    
    # Strategy: Split the 3x7 into blocks and use Strassen where possible.
    # 
    # Option 1: Split columns of B into 2+2+3, rows of A into 2.
    # For each 2-row block of A and 2-column block of B, we have a 2x2 * 2x2 subproblem.
    # But A is 2x3, so we can't directly do 2x2 blocks unless we split A's 3 columns.
    #
    # Better approach: Think of it as computing C = A * B where A is 2x3, B is 3x7.
    # We can split B into B1 (3x2), B2 (3x2), B3 (3x3).
    # C1 = A * B1 (2x2), C2 = A * B2 (2x2), C3 = A * B3 (2x3)
    #
    # For C1 = A * B1: A is 2x3, B1 is 3x2. This is a 2x3 * 3x2 problem.
    # Similarly for C2. For C3: 2x3 * 3x3.
    #
    # Alternative: Split A's columns. Let A = [A1 | A2] where A1 is 2x1, A2 is 2x2.
    # Let B = [B1; B2] where B1 is 1x7, B2 is 2x7.
    # C = A1*B1 + A2*B2
    # A1*B1: 2x1 * 1x7 = rank 2*1*7 = 14 (schoolbook)
    # A2*B2: 2x2 * 2x7. Split B2 into B21 (2x2) and B22 (2x5).
    #   A2*B21: 2x2 * 2x2 = rank 7 (Strassen)
    #   A2*B22: 2x2 * 2x5. Split into 2x2*2x2 + 2x2*2x3
    #     2x2*2x2: rank 7 (Strassen)
    #     2x2*2x3: Split B into 2x1 + 2x2
    #       2x2*2x1: rank 2*2*1 = 4 (schoolbook)
    #       2x2*2x2: rank 7 (Strassen)
    #
    # Total: 14 + 7 + 7 + 4 + 7 = 39
    #
    # Let me trace this more carefully:
    # A is 2x3. Split column-wise: A = [A_left | A_right] where A_left is 2x1, A_right is 2x2.
    # B is 3x7. Split row-wise: B = [B_top; B_bottom] where B_top is 1x7, B_bottom is 2x7.
    #
    # C = A_left * B_top + A_right * B_bottom
    #
    # Term 1: A_left (2x1) * B_top (1x7): schoolbook, rank = 2*1*7 = 14
    # Term 2: A_right (2x2) * B_bottom (2x7): split B_bottom into B_b1 (2x2) and B_b2 (2x5)
    #   Term 2a: A_right (2x2) * B_b1 (2x2): Strassen, rank = 7
    #   Term 2b: A_right (2x2) * B_b2 (2x5): split B_b2 into B_b2a (2x2) and B_b2b (2x3)
    #     Term 2b1: A_right (2x2) * B_b2a (2x2): Strassen, rank = 7
    #     Term 2b2: A_right (2x2) * B_b2b (2x3): split B_b2b into B_b2b1 (2x1) and B_b2b2 (2x2)
    #       Term 2b2i: A_right (2x2) * B_b2b1 (2x1): schoolbook, rank = 2*2*1 = 4
    #       Term 2b2ii: A_right (2x2) * B_b2b2 (2x2): Strassen, rank = 7
    #
    # Total rank = 14 + 7 + 7 + 4 + 7 = 39
    #
    # Now I need to carefully set up the offsets for each block.
    #
    # Let me define the blocks with their positions in the full matrices:
    #
    # Full A: 2x3, indexed as A[i,j] where i in {0,1}, j in {0,1,2}
    # Full B: 3x7, indexed as B[j,k] where j in {0,1,2}, k in {0,...,6}
    # Full C: 2x7, indexed as C[i,k] where i in {0,1}, k in {0,...,6}
    #
    # U: shape (2*3, rank) = (6, rank), U[i*3+j, c]
    # V: shape (3*7, rank) = (21, rank), V[j*7+k, c]
    # W: shape (7*2, rank) = (14, rank), W[k*2+i, c]
    #
    # Block 1: A_left * B_top
    #   A_left: rows 0-1, col 0 of A. So in U, rows are i*3+0 for i in {0,1} -> rows 0, 3
    #   B_top: row 0, cols 0-6 of B. So in V, rows are 0*7+k for k in 0..6 -> rows 0,1,2,3,4,5,6
    #   This is a 2x1 * 1x7 schoolbook.
    #   Local U: 2x1=2 rows, 1x1=1 col? No wait.
    #   
    #   For a block with local dimensions n1 x m1 * m1 x p1:
    #   U_local: n1*m1 rows, rank cols. U_local[i1*m1+j1, c]
    #   V_local: m1*p1 rows, rank cols. V_local[j1*p1+k1, c]
    #   W_local: p1*n1 rows, rank cols. W_local[k1*n1+i1, c]
    #
    #   Mapping to global:
    #   U_global_row = global_i_offset * m + (local_i * m1 + local_j)
    #   Actually, A is stored as A[i,j], U row index = i*m + j.
    #   For A_left: i ranges over {0,1} (global i), j=0 (global j).
    #   So local i1 in {0,1} maps to global i = i1, local j1 in {0} maps to global j = 0.
    #   U_global_row = i1 * 3 + 0
    #
    #   For B_top: j=0 (global j), k ranges over 0..6.
    #   V row index = j*7 + k. Local j1 in {0} maps to global j = 0.
    #   Local k1 in {0..6} maps to global k = k1.
    #   V_global_row = 0 * 7 + k1 = k1
    #
    #   For W: C[i,k], W row index = k*2 + i.
    #   Local i1 in {0,1} maps to global i = i1.
    #   Local k1 in {0..6} maps to global k = k1.
    #   W_global_row = k1 * 2 + i1
    
    # Let me build this step by step.
    
    # I'll define a helper to create a schoolbook block and a Strassen block,
    # then assemble them with proper offsets.
    
    def build_hybrid():
        """Build the hybrid Strassen/schoolbook decomposition of rank 39."""
        total_rank = 39
        U = [[0]*total_rank for _ in range(6)]   # 2*3
        V = [[0]*total_rank for _ in range(21)]  # 3*7
        W = [[0]*total_rank for _ in range(14)]  # 7*2
        
        col = 0
        
        # Block 1: A_left (2x1) * B_top (1x7), schoolbook
        # n1=2, m1=1, p1=7, rank1=14
        U1, V1, W1 = schoolbook_block(2, 1, 7)
        # U1: 2*1=2 rows. U1[i1*1+j1, c] where i1 in {0,1}, j1 in {0}
        # Map to global U: row = i1*3 + 0
        for ri in range(2):
            for c in range(14):
                i1 = ri // 1
                j1 = ri % 1
                U[i1*3 + 0][col + c] = U1[ri][c]
        # V1: 1*7=7 rows. V1[j1*7+k1, c] where j1 in {0}, k1 in {0..6}
        # Map to global V: row = 0*7 + k1 = k1
        for ri in range(7):
            for c in range(14):
                j1 = ri // 7
                k1 = ri % 7
                V[0*7 + k1][col + c] = V1[ri][c]
        # W1: 7*2=14 rows. W1[k1*2+i1, c] where k1 in {0..6}, i1 in {0,1}
        # Map to global W: row = k1*2 + i1
        for ri in range(14):
            for c in range(14):
                k1 = ri // 2
                i1 = ri % 2
                W[k1*2 + i1][col + c] = W1[ri][c]
        col += 14
        
        # Block 2a: A_right (2x2) * B_b1 (2x2), Strassen
        # A_right: rows 0-1, cols 1-2 of A. Global j in {1,2}
        # B_b1: rows 1-2, cols 0-1 of B. Global j in {1,2}, k in {0,1}
        U2, V2, W2 = make_strassen_2x2x2()
        # U2: 4 rows. U2[i1*2+j1, c] where i1 in {0,1}, j1 in {0,1}
        # Map to global U: row = i1*3 + (1+j1)
        for ri in range(4):
            for c in range(7):
                i1 = ri // 2
                j1 = ri % 2
                U[i1*3 + (1+j1)][col + c] = U2[ri][c]
        # V2: 4 rows. V2[j1*2+k1, c] where j1 in {0,1}, k1 in {0,1}
        # Map to global V: row = (1+j1)*7 + (0+k1)
        for ri in range(4):
            for c in range(7):
                j1 = ri // 2
                k1 = ri % 2
                V[(1+j1)*7 + (0+k1)][col + c] = V2[ri][c]
        # W2: 4 rows. W2[k1*2+i1, c] where k1 in {0,1}, i1 in {0,1}
        # Map to global W: row = (0+k1)*2 + i1
        for ri in range(4):
            for c in range(7):
                k1 = ri // 2
                i1 = ri % 2
                W[(0+k1)*2 + i1][col + c] = W2[ri][c]
        col += 7
        
        # Block 2b1: A_right (2x2) * B_b2a (2x2), Strassen
        # B_b2a: rows 1-2, cols 2-3 of B. Global j in {1,2}, k in {2,3}
        U3, V3, W3 = make_strassen_2x2x2()
        for ri in range(4):
            for c in range(7):
                i1 = ri // 2
                j1 = ri % 2
                U[i1*3 + (1+j1)][col + c] = U3[ri][c]
        for ri in range(4):
            for c in range(7):
                j1 = ri // 2
                k1 = ri % 2
                V[(1+j1)*7 + (2+k1)][col + c] = V3[ri][c]
        for ri in range(4):
            for c in range(7):
                k1 = ri // 2
                i1 = ri % 2
                W[(2+k1)*2 + i1][col + c] = W3[ri][c]
        col += 7
        
        # Block 2b2i: A_right (2x2) * B_b2b1 (2x1), schoolbook
        # B_b2b1: rows 1-2, col 4 of B. Global j in {1,2}, k in {4}
        U4, V4, W4 = schoolbook_block(2, 2, 1)
        # U4: 4 rows. U4[i1*2+j1, c]
        for ri in range(4):
            for c in range(4):
                i1 = ri // 2
                j1 = ri % 2
                U[i1*3 + (1+j1)][col + c] = U4[ri][c]
        # V4: 2 rows. V4[j1*1+k1, c] where j1 in {0,1}, k1 in {0}
        for ri in range(2):
            for c in range(4):
                j1 = ri // 1
                k1 = ri % 1
                V[(1+j1)*7 + (4+k1)][col + c] = V4[ri][c]
        # W4: 2 rows. W4[k1*2+i1, c] where k1 in {0}, i1 in {0,1}
        for ri in range(2):
            for c in range(4):
                k1 = ri // 2
                i1 = ri % 2
                W[(4+k1)*2 + i1][col + c] = W4[ri][c]
        col += 4
        
        # Block 2b2ii: A_right (2x2) * B_b2b2 (2x2), Strassen
        # B_b2b2: rows 1-2, cols 5-6 of B. Global j in {1,2}, k in {5,6}
        U5, V5, W5 = make_strassen_2x2x2()
        for ri in range(4):
            for c in range(7):
                i1 = ri // 2
                j1 = ri % 2
                U[i1*3 + (1+j1)][col + c] = U5[ri][c]
        for ri in range(4):
            for c in range(7):
                j1 = ri // 2
                k1 = ri % 2
                V[(1+j1)*7 + (5+k1)][col + c] = V5[ri][c]
        for ri in range(4):
            for c in range(7):
                k1 = ri // 2
                i1 = ri % 2
                W[(5+k1)*2 + i1][col + c] = W5[ri][c]
        col += 7
        
        assert col == total_rank, f"Column mismatch: {col} != {total_rank}"
        return U, V, W
    
    # Try the hybrid decomposition
    try:
        U, V, W = build_hybrid()
        return {"U": U, "V": V, "W": W}
    except Exception as e:
        # Fallback to schoolbook if hybrid fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
