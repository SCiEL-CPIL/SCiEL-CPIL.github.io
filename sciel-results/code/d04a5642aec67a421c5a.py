"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]
import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140  # 140 seconds to be safe

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Schoolbook decomposition as fallback
    rank_school = n * m * p  # 60
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Build rank-56 decomposition using Strassen's algorithm for 2x2 blocks
    # Strategy: Split 3x4x5 into two 3x2x5 blocks (columns 1-2 and 3-4 of B)
    # For each 3x2x5 block:
    #   - Row 1 (1x2x5): schoolbook, rank 10
    #   - Rows 2-3 (2x2x5): split into 2x2x2 + 2x2x2 + 2x2x1
    #     - Two 2x2x2 blocks use Strassen (rank 7 each)
    #     - One 2x2x1 block uses schoolbook (rank 4)
    #     Total for 2x2x5: 7+7+4 = 18
    # Total per 3x2x5: 10 + 18 = 28
    # Grand total: 2 * 28 = 56

    def strassen_2x2x2(col_offset):
        """Return U, V, W entries for a 2x2x2 Strassen decomposition.
        col_offset is the starting column index in the global arrays."""
        # Strassen's 7 products for 2x2x2:
        # P1 = (A11+A22)(B11+B22)
        # P2 = (A21+A22)B11
        # P3 = A11(B21-B22)
        # P4 = A22(B12-B11)
        # P5 = (A11+A12)B22
        # P6 = (A21-A11)(B11+B12)
        # P7 = (A12-A22)(B21+B22)
        #
        # C11 = P1 + P4 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        #
        # For 2x2x2 tensor T[i,j,k] where i,j,k in {0,1}:
        # U[i*2+j, c] corresponds to A[i,j]
        # V[j*2+k, c] corresponds to B[j,k]
        # W[k*2+i, c] corresponds to C[i,k]
        
        entries = []
        # (U_row, U_val, V_row, V_val, W_row, W_val)
        # P1
        entries.append(((0*2+0, 1), (0*2+0, 1), (0*2+0, 1)))  # A11*B11 -> C11
        entries.append(((1*2+1, 1), (0*2+0, 1), (0*2+0, 1)))  # A22*B11 -> C11
        entries.append(((0*2+0, 1), (1*2+1, 1), (0*2+0, 1)))  # A11*B22 -> C11
        entries.append(((1*2+1, 1), (1*2+1, 1), (0*2+0, 1)))  # A22*B22 -> C11
        # P2
        entries.append(((1*2+0, 1), (0*2+0, 1), (1*2+0, 1)))  # A21*B11 -> C21
        entries.append(((1*2+1, 1), (0*2+0, 1), (1*2+0, 1)))  # A22*B11 -> C21
        # P3
        entries.append(((0*2+0, 1), (1*2+0, 1), (1*2+0, 1)))  # A11*B21 -> C21
        entries.append(((0*2+0, -1), (1*2+1, 1), (1*2+0, 1)))  # A11*B22 -> C21
        # P4
        entries.append(((1*2+1, 1), (0*2+1, 1), (0*2+1, 1)))  # A22*B12 -> C12
        entries.append(((1*2+1, -1), (0*2+0, 1), (0*2+1, 1)))  # A22*B11 -> C12
        # P5
        entries.append(((0*2+0, 1), (1*2+1, 1), (0*2+1, 1)))  # A11*B22 -> C12
        entries.append(((0*2+1, 1), (1*2+1, 1), (0*2+1, 1)))  # A12*B22 -> C12
        # P6
        entries.append(((1*2+0, 1), (0*2+0, 1), (1*2+1, 1)))  # A21*B11 -> C22
        entries.append(((1*2+0, 1), (0*2+1, 1), (1*2+1, 1)))  # A21*B12 -> C22
        entries.append(((0*2+0, -1), (0*2+0, 1), (1*2+1, 1)))  # A11*B11 -> C22
        entries.append(((0*2+0, -1), (0*2+1, 1), (1*2+1, 1)))  # A11*B12 -> C22
        # P7
        entries.append(((0*2+1, 1), (1*2+0, 1), (0*2+0, 1)))  # A12*B21 -> C11
        entries.append(((0*2+1, 1), (1*2+1, 1), (0*2+0, 1)))  # A12*B22 -> C11
        entries.append(((1*2+1, -1), (1*2+0, 1), (0*2+0, 1)))  # A22*B21 -> C11
        entries.append(((1*2+1, -1), (1*2+1, 1), (0*2+0, 1)))  # A22*B22 -> C11
        
        # Wait, I need to be more careful. Let me re-derive Strassen properly.
        # 
        # Standard Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        #
        # Each M is a single product of linear combinations.
        # For the tensor, each M contributes to U, V, W with coefficients.
        #
        # Let me define each M's contribution:
        # M1: U has A11(+1), A22(+1); V has B11(+1), B22(+1); W has C11(+1), C22(+1)
        # M2: U has A21(+1), A22(+1); V has B11(+1); W has C21(+1), C22(-1)
        # M3: U has A11(+1); V has B21(+1), B22(-1); W has C12(+1), C22(+1)
        # M4: U has A22(+1); V has B12(+1), B11(-1); W has C11(+1), C21(+1)
        # M5: U has A11(+1), A12(+1); V has B22(+1); W has C11(-1), C12(+1)
        # M6: U has A21(+1), A11(-1); V has B11(+1), B12(+1); W has C22(+1)
        # M7: U has A12(+1), A22(-1); V has B21(+1), B22(+1); W has C11(+1)
        
        # I'll build U, V, W as lists of (row, value) for each of the 7 columns
        U_cols = []
        V_cols = []
        W_cols = []
        
        # M1
        U_cols.append([(0, 1), (3, 1)])  # A11, A22 (rows 0,3 in 2x2)
        V_cols.append([(0, 1), (3, 1)])  # B11, B22 (rows 0,3 in 2x2)
        W_cols.append([(0, 1), (3, 1)])  # C11, C22 (rows 0,3 in 2x2)
        
        # M2
        U_cols.append([(2, 1), (3, 1)])  # A21, A22 (rows 2,3)
        V_cols.append([(0, 1)])  # B11 (row 0)
        W_cols.append([(2, 1), (3, -1)])  # C21(+1), C22(-1) (rows 2,3)
        
        # M3
        U_cols.append([(0, 1)])  # A11 (row 0)
        V_cols.append([(2, 1), (3, -1)])  # B21(+1), B22(-1) (rows 2,3)
        W_cols.append([(1, 1), (3, 1)])  # C12(+1), C22(+1) (rows 1,3)
        
        # M4
        U_cols.append([(3, 1)])  # A22 (row 3)
        V_cols.append([(1, 1), (0, -1)])  # B12(+1), B11(-1) (rows 1,0)
        W_cols.append([(0, 1), (2, 1)])  # C11(+1), C21(+1) (rows 0,2)
        
        # M5
        U_cols.append([(0, 1), (1, 1)])  # A11(+1), A12(+1) (rows 0,1)
        V_cols.append([(3, 1)])  # B22 (row 3)
        W_cols.append([(0, -1), (1, 1)])  # C11(-1), C12(+1) (rows 0,1)
        
        # M6
        U_cols.append([(2, 1), (0, -1)])  # A21(+1), A11(-1) (rows 2,0)
        V_cols.append([(0, 1), (1, 1)])  # B11(+1), B12(+1) (rows 0,1)
        W_cols.append([(3, 1)])  # C22 (row 3)
        
        # M7
        U_cols.append([(1, 1), (3, -1)])  # A12(+1), A22(-1) (rows 1,3)
        V_cols.append([(2, 1), (3, 1)])  # B21(+1), B22(+1) (rows 2,3)
        W_cols.append([(0, 1)])  # C11 (row 0)
        
        return U_cols, V_cols, W_cols

    def build_rank56():
        """Build the rank-56 decomposition."""
        rank = 56
        U = [[0.0] * rank for _ in range(nm)]
        V = [[0.0] * rank for _ in range(mp)]
        W = [[0.0] * rank for _ in range(pn)]
        
        col = 0
        
        # Two blocks: columns 0-1 (j=0,1) and columns 2-3 (j=2,3) of B
        for block in range(2):
            j_offset = block * 2  # 0 or 2
            
            # Part 1: Row 0 (i=0) of A, columns j_offset, j_offset+1 of B, all k
            # This is 1x2x5: C[0,k] = sum_{j=0}^{1} A[0,j] * B[j,k]
            # Schoolbook: 1*2*5 = 10 terms
            for j in range(2):
                for k in range(5):
                    a_row = 0 * m + (j_offset + j)  # A row 0, col j_offset+j
                    b_row = (j_offset + j) * p + k  # B row j_offset+j, col k
                    c_row = k * n + 0  # C row k, col 0
                    
                    U[a_row][col] = 1
                    V[b_row][col] = 1
                    W[c_row][col] = 1
                    col += 1
            
            # Part 2: Rows 1-2 (i=1,2) of A, columns j_offset, j_offset+1 of B, all k
            # This is 2x2x5: split into three parts
            # - k=0,1: 2x2x2 (Strassen, rank 7)
            # - k=2,3: 2x2x2 (Strassen, rank 7)
            # - k=4: 2x2x1 (schoolbook, rank 4)
            
            # Sub-part 2a: k=0,1 (2x2x2)
            U_cols, V_cols, W_cols = strassen_2x2x2()
            for c in range(7):
                for (row, val) in U_cols[c]:
                    a_row = (1 + row // 2) * m + (j_offset + row % 2)  # A row 1 or 2
                    U[a_row][col] = val
                for (row, val) in V_cols[c]:
                    b_row = (j_offset + row % 2) * p + (row // 2)  # B row j_offset+j, col k
                    V[b_row][col] = val
                for (row, val) in W_cols[c]:
                    c_row = (row % 2) * n + (1 + row // 2)  # C row k, col i
                    W[c_row][col] = val
                col += 1
            
            # Sub-part 2b: k=2,3 (2x2x2)
            U_cols, V_cols, W_cols = strassen_2x2x2()
            for c in range(7):
                for (row, val) in U_cols[c]:
                    a_row = (1 + row // 2) * m + (j_offset + row % 2)
                    U[a_row][col] = val
                for (row, val) in V_cols[c]:
                    b_row = (j_offset + row % 2) * p + (2 + row // 2)  # B row, col 2 or 3
                    V[b_row][col] = val
                for (row, val) in W_cols[c]:
                    c_row = (2 + row % 2) * n + (1 + row // 2)  # C row 2 or 3, col 1 or 2
                    W[c_row][col] = val
                col += 1
            
            # Sub-part 2c: k=4 (2x2x1, schoolbook, rank 4)
            for i in range(2):  # i=0,1 in local 2x2 (global i=1,2)
                for j in range(2):  # j=0,1 in local 2x2 (global j=j_offset+j)
                    a_row = (1 + i) * m + (j_offset + j)
                    b_row = (j_offset + j) * p + 4  # B row, col 4
                    c_row = 4 * n + (1 + i)  # C row 4, col 1 or 2
                    
                    U[a_row][col] = 1
                    V[b_row][col] = 1
                    W[c_row][col] = 1
                    col += 1
        
        return {"U": U, "V": V, "W": W}
    
    # Try to build and verify the rank-56 decomposition
    if time.time() < deadline - 10:
        candidate = build_rank56()
        
        # Verify
        valid = True
        U_arr = np.array(candidate["U"])
        V_arr = np.array(candidate["V"])
        W_arr = np.array(candidate["W"])
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(56):
                        val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                    if abs(val - 1) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            best = candidate
            best_rank = 56
    
    return best
# EVOLVE-BLOCK-END
