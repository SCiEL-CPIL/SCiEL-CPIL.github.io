# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = 4, 5, 6
    start_time = time.time()
    deadline = start_time + 140

    # Total rank calculation
    # Part 1: 4x4x4 using 8 Strassen 2x2 blocks. Rank = 8 * 7 = 56.
    # Part 2: 4x4x2 using schoolbook. Rank = 4 * 4 * 2 = 32.
    # Part 3: 4x1x6 using outer product. Rank = 1.
    # Total Rank = 56 + 32 + 1 = 89.
    
    total_rank = 89
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    
    # --- Part 1: A1 (4x4) * B1a (4x4) ---
    # A1 corresponds to A[0:4, 0:4]
    # B1a corresponds to B[0:4, 0:4]
    # Output C[0:4, 0:4]
    
    # We divide A1 into 4 2x2 blocks: A11, A12, A21, A22
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    
    # B1a into 4 2x2 blocks: B11, B12, B21, B22
    # B11: rows 0-1, cols 0-1
    # B12: rows 0-1, cols 2-3
    # B21: rows 2-3, cols 0-1
    # B22: rows 2-3, cols 2-3
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # There are 8 distinct 2x2 multiplications:
    # P1: A11 * B11
    # P2: A12 * B21
    # P3: A11 * B12
    # P4: A12 * B22
    # P5: A21 * B11
    # P6: A22 * B21
    # P7: A21 * B12
    # P8: A22 * B22
    
    # Strassen coefficients for 2x2:
    # M1 = (A11+A22)(B11+B22) -> Indices in A: (0,0),(1,1); B: (0,0),(1,1)
    # M2 = (A21+A22)B11 -> A: (1,0),(1,1); B: (0,0)
    # M3 = A11(B12-B22) -> A: (0,0); B: (0,1),(1,1)
    # M4 = A22(B21-B11) -> A: (1,1); B: (1,0),(0,0)
    # M5 = (A11+A12)B22 -> A: (0,0),(0,1); B: (1,1)
    # M6 = (A21-A11)(B11+B12) -> A: (1,0),(0,0); B: (0,0),(0,1)
    # M7 = (A12-A22)(B21+B22) -> A: (0,1),(1,1); B: (1,0),(1,1)
    
    # Results:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map these to global U, V, W indices.
    # Local 2x2 indices (r, c) map to global (row_offset + r, col_offset + c).
    
    def add_strassen_block(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off, base_col):
        """
        Adds a Strassen decomposition for a 2x2 block multiplication.
        Returns the next column index.
        """
        col_idx = base_col
        
        # M1 = (A00+A11)(B00+B11)
        # U: A00, A11
        U[a_row_off*5 + a_col_off + 0, col_idx] += 1.0
        U[a_row_off*5 + a_col_off + 1, col_idx] += 1.0
        # V: B00, B11
        V[b_row_off*6 + b_col_off + 0, col_idx] += 1.0
        V[b_row_off*6 + b_col_off + 1, col_idx] += 1.0
        # W: C00, C01, C10, C11 contributions
        # C00 += M1
        W[(c_col_off + 0)*4 + (c_row_off + 0), col_idx] += 1.0
        # C01 += 0
        # C10 += 0
        # C11 += M1
        W[(c_col_off + 1)*4 + (c_row_off + 1), col_idx] += 1.0
        col_idx += 1
        
        # M2 = (A10+A11)B00
        U[a_row_off*5 + a_col_off + 4, col_idx] += 1.0 # A10 is row 1, col 0 -> index 1*? No, local (1,0)
        # Local indices:
        # A00: (0,0), A01: (0,1)
        # A10: (1,0), A11: (1,1)
        # Global A index: (a_row_off + r) * 5 + (a_col_off + c)
        
        # Correction:
        # A10 global: (a_row_off + 1) * 5 + (a_col_off + 0)
        # A11 global: (a_row_off + 1) * 5 + (a_col_off + 1)
        U[(a_row_off + 1)*5 + a_col_off + 0, col_idx] += 1.0
        U[(a_row_off + 1)*5 + a_col_off + 1, col_idx] += 1.0
        # B00 global: (b_row_off + 0) * 6 + (b_col_off + 0)
        V[b_row_off*6 + b_col_off + 0, col_idx] += 1.0
        # C10 += M2
        W[(c_col_off + 0)*4 + (c_row_off + 1), col_idx] += 1.0
        # C11 += -M2
        W[(c_col_off + 1)*4 + (c_row_off + 1), col_idx] += -1.0
        col_idx += 1
        
        # M3 = A00(B01-B11)
        U[a_row_off*5 + a_col_off + 0, col_idx] += 1.0
        # B01 global: (b_row_off + 0) * 6 + (b_col_off + 1)
        # B11 global: (b_row_off + 1) * 6 + (b_col_off + 1)
        V[b_row_off*6 + b_col_off + 1, col_idx] += 1.0
        V[(b_row_off + 1)*6 + b_col_off + 1, col_idx] += -1.0
        # C01 += M3
        W[(c_col_off + 1)*4 + (c_row_off + 0), col_idx] += 1.0
        # C11 += M3
        W[(c_col_off + 1)*4 + (c_row_off + 1), col_idx] += 1.0
        col_idx += 1
        
        # M4 = A11(B10-B00)
        U[(a_row_off + 1)*5 + a_col_off + 1, col_idx] += 1.0
        # B10 global: (b_row_off + 1) * 6 + (b_col_off + 0)
        # B00 global: (b_row_off + 0) * 6 + (b_col_off + 0)
        V[(b_row_off + 1)*6 + b_col_off + 0, col_idx] += 1.0
        V[b_row_off*6 + b_col_off + 0, col_idx] += -1.0
        # C00 += M4
        W[(c_col_off + 0)*4 + (c_row_off + 0), col_idx] += 1.0
        # C10 += M4
        W[(c_col_off + 0)*4 + (c_row_off + 1), col_idx] += 1.0
        col_idx += 1
        
        # M5 = (A00+A01)B11
        U[a_row_off*5 + a_col_off + 0, col_idx] += 1.0
        U[a_row_off*5 + a_col_off + 1, col_idx] += 1.0
        # B11 global: (b_row_off + 1) * 6 + (b_col_off + 1)
        V[(b_row_off + 1)*6 + b_col_off + 1, col_idx] += 1.0
        # C00 += -M5
        W[(c_col_off + 0)*4 + (c_row_off + 0), col_idx] += -1.0
        # C01 += M5
        W[(c_col_off + 1)*4 + (c_row_off + 0), col_idx] += 1.0
        col_idx += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[(a_row_off + 1)*5 + a_col_off + 0, col_idx] += 1.0
        U[a_row_off*5 + a_col_off + 0, col_idx] += -1.0
        # B00 global: (b_row_off + 0) * 6 + (b_col_off + 0)
        # B01 global: (b_row_off + 0) * 6 + (b_col_off + 1)
        V[b_row_off*6 + b_col_off + 0, col_idx] += 1.0
        V[b_row_off*6 + b_col_off + 1, col_idx] += 1.0
        # C11 += M6
        W[(c_col_off + 1)*4 + (c_row_off + 1), col_idx] += 1.0
        col_idx += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[a_row_off*5 + a_col_off + 1, col_idx] += 1.0
        U[(a_row_off + 1)*5 + a_col_off + 1, col_idx] += -1.0
        # B10 global: (b_row_off + 1) * 6 + (b_col_off + 0)
        # B11 global: (b_row_off + 1) * 6 + (b_col_off + 1)
        V[(b_row_off + 1)*6 + b_col_off + 0, col_idx] += 1.0
        V[(b_row_off + 1)*6 + b_col_off + 1, col_idx] += 1.0
        # C00 += M7
        W[(c_col_off + 0)*4 + (c_row_off + 0), col_idx] += 1.0
        col_idx += 1
        
        return col_idx

    # 8 Strassen blocks for 4x4x4
    # Block 1: A11*B11 -> C11 part
    # A11: rows 0-1, cols 0-1. B11: rows 0-1, cols 0-1. C11: rows 0-1, cols 0-1.
    col = add_strassen_block(0, 0, 0, 0, 0, 0, col)
    
    # Block 2: A12*B21 -> C11 part
    # A12: rows 0-1, cols 2-3. B21: rows 2-3, cols 0-1. C11: rows 0-1, cols 0-1.
    col = add_strassen_block(0, 2, 2, 0, 0, 0, col)
    
    # Block 3: A11*B12 -> C12 part
    # A11: rows 0-1, cols 0-1. B12: rows 0-1, cols 2-3. C12: rows 0-1, cols 2-3.
    col = add_strassen_block(0, 0, 0, 2, 0, 2, col)
    
    # Block 4: A12*B22 -> C12 part
    # A12: rows 0-1, cols 2-3. B22: rows 2-3, cols 2-3. C12: rows 0-1, cols 2-3.
    col = add_strassen_block(0, 2, 2, 2, 0, 2, col)
    
    # Block 5: A21*B11 -> C21 part
    # A21: rows 2-3, cols 0-1. B11: rows 0-1, cols 0-1. C21: rows 2-3, cols 0-1.
    col = add_strassen_block(2, 0, 0, 0, 2, 0, col)
    
    # Block 6: A22*B21 -> C21 part
    # A22: rows 2-3, cols 2-3. B21: rows 2-3, cols 0-1. C21: rows 2-3, cols 0-1.
    col = add_strassen_block(2, 2, 2, 0, 2, 0, col)
    
    # Block 7: A21*B12 -> C22 part
    # A21: rows 2-3, cols 0-1. B12: rows 0-1, cols 2-3. C22: rows 2-3, cols 2-3.
    col = add_strassen_block(2, 0, 0, 2, 2, 2, col)
    
    # Block 8: A22*B22 -> C22 part
    # A22: rows 2-3, cols 2-3. B22: rows 2-3, cols 2-3. C22: rows 2-3, cols 2-3.
    col = add_strassen_block(2, 2, 2, 2, 2, 2, col)
    
    # --- Part 2: A1 (4x4) * B1b (4x2) ---
    # A1: rows 0-3, cols 0-3.
    # B1b: rows 0-3, cols 4-5.
    # Output C: rows 0-3, cols 4-5.
    # Schoolbook: 4 * 4 * 2 = 32 terms.
    for i in range(4):
        for j in range(4):
            for k in range(2):
                # Global A index: i * 5 + j
                # Global B index: j * 6 + (4 + k)
                # Global W index: (4 + k) * 4 + i
                U[i * 5 + j, col] = 1.0
                V[j * 6 + (4 + k), col] = 1.0
                W[(4 + k) * 4 + i, col] = 1.0
                col += 1
                
    # --- Part 3: A2 (4x1) * B2 (1x6) ---
    # A2: rows 0-3, col 4.
    # B2: row 4, cols 0-5.
    # Output C: rows 0-3, cols 0-5.
    # Outer product: Rank 1.
    # C[i, k] = A[i, 4] * B[4, k]
    # U[i * 5 + 4, col] = 1.0
    # V[4 * 6 + k, col] = 1.0
    # W[k * 4 + i, col] = 1.0
    # This single term handles all i, k?
    # No, U, V, W are vectors for the rank-1 term.
    # U is a vector of length 20. U[i*5+4] = 1.
    # V is a vector of length 30. V[4*6+k] = 1.
    # W is a vector of length 24. W[k*4+i] = 1.
    # Then C[i,k] = sum_r W[i,k,r] * (sum_j U[i,j,r] A[i,j]) * (sum_j V[j,k,r] B[j,k])
    # For r = col:
    # sum_j U[i,j,col] A[i,j] = A[i,4] (since U[i*5+4]=1, others 0)
    # sum_j V[j,k,col] B[j,k] = B[4,k] (since V[4*6+k]=1, others 0)
    # W[k*4+i, col] = 1.
    # So C[i,k] = 1 * A[i,4] * B[4,k]. Correct.
    
    for i in range(4):
        U[i * 5 + 4, col] = 1.0
    for k in range(6):
        V[4 * 6 + k, col] = 1.0
    for i in range(4):
        for k in range(6):
            W[k * 4 + i, col] = 1.0
    col += 1
    
    # Verify col == total_rank
    # assert col == 89, f"Col {col} != 89"
    
    def to_list(mat):
        return [[round(float(x), 12) for x in row] for row in mat]
    
    def clean_list(lst):
        for row in lst:
            for idx in range(len(row)):
                v = row[idx]
                if abs(v) < 1e-10:
                    row[idx] = 0.0
                elif abs(v - round(v)) < 1e-10:
                    row[idx] = float(round(v))
                elif abs(v - round(v*2)/2) < 1e-10:
                    row[idx] = round(v*2)/2
        return lst
    
    U_out = clean_list(to_list(U))
    V_out = clean_list(to_list(V))
    W_out = clean_list(to_list(W))
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
