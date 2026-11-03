"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use a recursive block decomposition with Strassen for 2x2x2 blocks
    # to reduce rank below schoolbook (n*m*p = 60)
    
    # Strategy: Split the 3x4x5 multiplication into blocks
    # A is 3x4, B is 4x5, C is 3x5
    # 
    # Split A into A1 (1x4) and A2 (2x4)
    # Split B into B1 (4x2) and B2 (4x3) -- no, let's be more careful
    # 
    # Let's use a different approach:
    # Split A into rows: A_top (1x4), A_bot (2x4)
    # Split B into cols: B_left (4x2), B_right (4x3)
    # C_top = A_top * B (1x5)
    # C_bot = A_bot * B (2x5)
    #
    # For C_top: 1x4 * 4x5 = 1x5, schoolbook rank = 1*4*5 = 20
    # For C_bot: 2x4 * 4x5 = 2x5
    #   Split A_bot into A_bot1 (2x2) and A_bot2 (2x2)
    #   Split B into B1 (4x2) and B2 (4x3) -- but we need consistent splits
    #
    # Better: Split the inner dimension m=4 into two parts of size 2 each
    # A = [A_left | A_right] where A_left is 3x2, A_right is 3x2
    # B = [B_top; B_bot] where B_top is 2x5, B_bot is 2x5
    # C = A_left * B_top + A_right * B_bot
    #
    # Each term is 3x2 * 2x5 = 3x5, schoolbook rank = 3*2*5 = 30 each
    # Total = 60, no improvement.
    #
    # Use Strassen for 2x2x2 blocks:
    # Split n=3 into 1+2, m=4 into 2+2, p=5 into 2+3
    #
    # Block (1,2,2): A[0:1,0:2] * B[0:2,0:2] -> C[0:1,0:2], rank = 1*2*2 = 4
    # Block (1,2,3): A[0:1,0:2] * B[0:2,2:5] -> C[0:1,2:5], rank = 1*2*3 = 6
    # Block (1,2,2): A[0:1,2:4] * B[2:4,0:2] -> C[0:1,0:2], rank = 1*2*2 = 4
    # Block (1,2,3): A[0:1,2:4] * B[2:4,2:5] -> C[0:1,2:5], rank = 1*2*3 = 6
    # Block (2,2,2): A[1:3,0:2] * B[0:2,0:2] -> C[1:3,0:2], rank = 2*2*2 = 8 -> Strassen: 7
    # Block (2,2,3): A[1:3,0:2] * B[0:2,2:5] -> C[1:3,2:5], rank = 2*2*3 = 12
    # Block (2,2,2): A[1:3,2:4] * B[2:4,0:2] -> C[1:3,0:2], rank = 2*2*2 = 8 -> Strassen: 7
    # Block (2,2,3): A[1:3,2:4] * B[2:4,2:5] -> C[1:3,2:5], rank = 2*2*3 = 12
    #
    # Total = 4+6+4+6+7+12+7+12 = 58
    
    # Let's implement this block decomposition
    
    rank = 0
    U = []
    V = []
    W = []
    
    def add_block(i_start, i_end, j_start, j_end, k_start, k_end, use_strassen=False):
        nonlocal rank, U, V, W
        ni = i_end - i_start
        nj = j_end - j_start
        nk = k_end - k_start
        
        if use_strassen and ni == 2 and nj == 2 and nk == 2:
            # Strassen 2x2x2
            # M1 = (A11+A22)(B11+B22)
            # M2 = (A21+A22)B11
            # M3 = A11(B21-B22)
            # M4 = A22(B12-B11)
            # M5 = (A11+A12)B22
            # M6 = (A21-A11)(B11+B12)
            # M7 = (A12-A22)(B21+B22)
            # C11 = M1+M4-M5+M7
            # C12 = M3+M5
            # C21 = M2+M4
            # C22 = M1-M2+M3+M6
            for s in range(7):
                col = rank
                rank += 1
                U.append([0] * (rank - col))  # placeholder, we'll build properly
                V.append([0] * (rank - col))
                W.append([0] * (rank - col))
            # Actually, let's build the full matrices properly
            # We need to construct U, V, W as lists of lists
            # Let me restructure
            
        else:
            # Schoolbook for this block
            for ii in range(ni):
                for jj in range(nj):
                    for kk in range(nk):
                        col = rank
                        rank += 1
                        U.append([0] * 1)
                        V.append([0] * 1)
                        W.append([0] * 1)
    
    # This approach is getting complicated. Let me just build the full U, V, W directly.
    
    # Start with schoolbook rank 60, then apply Strassen to the 2x2x2 blocks
    # to reduce by 1 each (from 8 to 7)
    
    # There are two 2x2x2 blocks:
    # Block A: i in [1,3), j in [0,2), k in [0,2)
    # Block B: i in [1,3), j in [2,4), k in [0,2)
    
    # For each 2x2x2 block, replace 8 schoolbook terms with 7 Strassen terms
    
    # Let's build the decomposition explicitly
    
    # First, create the schoolbook decomposition
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
    
    # Now, for the two 2x2x2 blocks, replace 8 columns with 7 Strassen columns
    # Block 1: i in {1,2}, j in {0,1}, k in {0,1}
    # Block 2: i in {1,2}, j in {2,3}, k in {0,1}
    
    # For Block 1:
    # The 8 schoolbook terms are at columns:
    # (i=1,j=0,k=0): col 0*15 + 0*5 + 0 = 0... wait, let me recalculate
    # col = i * (m*p) + j * p + k? No, the original indexing is:
    # for i in range(n): for j in range(m): for k in range(p): col
    # So col = i * m * p + j * p + k
    
    # Block 1: i in {1,2}, j in {0,1}, k in {0,1}
    # Cols: 
    # (1,0,0): 1*20 + 0*5 + 0 = 20
    # (1,0,1): 1*20 + 0*5 + 1 = 21
    # (1,1,0): 1*20 + 1*5 + 0 = 25
    # (1,1,1): 1*20 + 1*5 + 1 = 26
    # (2,0,0): 2*20 + 0*5 + 0 = 40
    # (2,0,1): 2*20 + 0*5 + 1 = 41
    # (2,1,0): 2*20 + 1*5 + 0 = 45
    # (2,1,1): 2*20 + 1*5 + 1 = 46
    
    # Strassen for 2x2x2:
    # Let A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For Block 1, the local indices are:
    # i_local 0 -> i_global 1, i_local 1 -> i_global 2
    # j_local 0 -> j_global 0, j_local 1 -> j_global 1
    # k_local 0 -> k_global 0, k_local 1 -> k_global 1
    
    # The 8 schoolbook columns to replace:
    block1_cols = [20, 21, 25, 26, 40, 41, 45, 46]
    
    # We'll replace these 8 columns with 7 new columns
    # But we need to be careful: we can't just remove columns, we need to
    # restructure the entire decomposition.
    
    # Let me take a different approach: build the decomposition from scratch
    # using the block structure with Strassen for the 2x2x2 blocks.
    
    # Reset
    rank = 0
    U_rows = {}  # (i, j) -> list of (coeff, col)
    V_rows = {}  # (j, k) -> list of (coeff, col)
    W_rows = {}  # (k, i) -> list of (coeff, col)
    
    def add_term(i, j, k, coeff=1):
        nonlocal rank
        col = rank
        rank += 1
        U_rows.setdefault((i, j), []).append((coeff, col))
        V_rows.setdefault((j, k), []).append((coeff, col))
        W_rows.setdefault((k, i), []).append((coeff, col))
    
    # Process each block
    # Block (i_start:i_end, j_start:j_end, k_start:k_end)
    def process_block(i_start, i_end, j_start, j_end, k_start, k_end):
        ni = i_end - i_start
        nj = j_end - j_start
        nk = k_end - k_start
        
        if ni == 2 and nj == 2 and nk == 2:
            # Strassen
            # Local indices: i0=i_start, i1=i_start+1
            # j0=j_start, j1=j_start+1
            # k0=k_start, k1=k_start+1
            i0, i1 = i_start, i_start + 1
            j0, j1 = j_start, j_start + 1
            k0, k1 = k_start, k_start + 1
            
            # M1 = (a00+a11)(b00+b11)
            c = rank; rank += 1
            U_rows.setdefault((i0, j0), []).append((1, c))
            U_rows.setdefault((i1, j1), []).append((1, c))
            V_rows.setdefault((j0, k0), []).append((1, c))
            V_rows.setdefault((j1, k1), []).append((1, c))
            W_rows.setdefault((k0, i0), []).append((1, c))
            W_rows.setdefault((k0, i1), []).append((1, c))
            W_rows.setdefault((k1, i0), []).append((1, c))
            W_rows.setdefault((k1, i1), []).append((1, c))
            
            # M2 = (a10+a11)b00
            c = rank; rank += 1
            U_rows.setdefault((i1, j0), []).append((1, c))
            U_rows.setdefault((i1, j1), []).append((1, c))
            V_rows.setdefault((j0, k0), []).append((1, c))
            W_rows.setdefault((k0, i1), []).append((1, c))
            
            # M3 = a00(b10-b11)
            c = rank; rank += 1
            U_rows.setdefault((i0, j0), []).append((1, c))
            V_rows.setdefault((j1, k0), []).append((1, c))
            V_rows.setdefault((j1, k1), []).append((-1, c))
            W_rows.setdefault((k0, i0), []).append((1, c))
            W_rows.setdefault((k1, i0), []).append((-1, c))
            
            # M4 = a11(b01-b00)
            c = rank; rank += 1
            U_rows.setdefault((i1, j1), []).append((1, c))
            V_rows.setdefault((j0, k1), []).append((1, c))
            V_rows.setdefault((j0, k0), []).append((-1, c))
            W_rows.setdefault((k1, i1), []).append((1, c))
            W_rows.setdefault((k0, i1), []).append((-1, c))
            
            # M5 = (a00+a01)b11
            c = rank; rank += 1
            U_rows.setdefault((i0, j0), []).append((1, c))
            U_rows.setdefault((i0, j1), []).append((1, c))
            V_rows.setdefault((j1, k1), []).append((1, c))
            W_rows.setdefault((k1, i0), []).append((1, c))
            
            # M6 = (a10-a00)(b00+b01)
            c = rank; rank += 1
            U_rows.setdefault((i1, j0), []).append((1, c))
            U_rows.setdefault((i0, j0), []).append((-1, c))
            V_rows.setdefault((j0, k0), []).append((1, c))
            V_rows.setdefault((j0, k1), []).append((1, c))
            W_rows.setdefault((k0, i1), []).append((1, c))
            W_rows.setdefault((k1, i1), []).append((1, c))
            W_rows.setdefault((k0, i0), []).append((-1, c))
            W_rows.setdefault((k1, i0), []).append((-1, c))
            
            # M7 = (a01-a11)(b10+b11)
            c = rank; rank += 1
            U_rows.setdefault((i0, j1), []).append((1, c))
            U_rows.setdefault((i1, j1), []).append((-1, c))
            V_rows.setdefault((j1, k0), []).append((1, c))
            V_rows.setdefault((j1, k1), []).append((1, c))
            W_rows.setdefault((k0, i0), []).append((1, c))
            W_rows.setdefault((k1, i0), []).append((1, c))
            W_rows.setdefault((k0, i1), []).append((-1, c))
            W_rows.setdefault((k1, i1), []).append((-1, c))
        else:
            # Schoolbook
            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    for k in range(k_start, k_end):
                        c = rank; rank += 1
                        U_rows.setdefault((i, j), []).append((1, c))
                        V_rows.setdefault((j, k), []).append((1, c))
                        W_rows.setdefault((k, i), []).append((1, c))
    
    # Split: n=3 into 1+2, m=4 into 2+2, p=5 into 2+3
    # Blocks:
    process_block(0, 1, 0, 2, 0, 2)  # 1x2x2
    process_block(0, 1, 0, 2, 2, 5)  # 1x2x3
    process_block(0, 1, 2, 4, 0, 2)  # 1x2x2
    process_block(0, 1, 2, 4, 2, 5)  # 1x2x3
    process_block(1, 3, 0, 2, 0, 2)  # 2x2x2 -> Strassen
    process_block(1, 3, 0, 2, 2, 5)  # 2x2x3
    process_block(1, 3, 2, 4, 0, 2)  # 2x2x2 -> Strassen
    process_block(1, 3, 2, 4, 2, 5)  # 2x2x3
    
    # Now build U, V, W as dense matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for (i, j), terms in U_rows.items():
        for coeff, col in terms:
            U[i * m + j][col] += coeff
    
    for (j, k), terms in V_rows.items():
        for coeff, col in terms:
            V[j * p + k][col] += coeff
    
    for (k, i), terms in W_rows.items():
        for coeff, col in terms:
            W[k * n + i][col] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
