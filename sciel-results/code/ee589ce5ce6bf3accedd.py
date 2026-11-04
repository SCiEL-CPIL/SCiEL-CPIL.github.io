import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 20 decomposition
    rank = 20
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add Strassen decomposition for a 2x2x2 block
    # i_offset, j_offset, k_offset are the global starting indices
    # col_start is the starting column index in U, V, W
    def add_strassen(i_off, j_off, k_off, col_start):
        # Strassen coefficients for U (A), V (B), W (C)
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Term 0 (M1)
        U[i_off*3 + (j_off+0)][col_start+0] += 1
        U[i_off*3 + (j_off+1)][col_start+0] += 1
        V[(j_off+0)*4 + (k_off+0)][col_start+0] += 1
        V[(j_off+0)*4 + (k_off+1)][col_start+0] += 1
        V[(j_off+1)*4 + (k_off+0)][col_start+0] += 1
        V[(j_off+1)*4 + (k_off+1)][col_start+0] += 1
        W[(k_off+0)*3 + (i_off+0)][col_start+0] += 1
        W[(k_off+0)*3 + (i_off+1)][col_start+0] += 1
        W[(k_off+1)*3 + (i_off+0)][col_start+0] += 1
        W[(k_off+1)*3 + (i_off+1)][col_start+0] += 1
        
        # Term 1 (M2)
        U[i_off*3 + (j_off+0)][col_start+1] += 1
        U[i_off*3 + (j_off+1)][col_start+1] += 1
        V[(j_off+0)*4 + (k_off+0)][col_start+1] += 1
        W[(k_off+0)*3 + (i_off+1)][col_start+1] += 1
        W[(k_off+1)*3 + (i_off+1)][col_start+1] += 1
        
        # Term 2 (M3)
        U[i_off*3 + (j_off+0)][col_start+2] += 1
        V[(j_off+0)*4 + (k_off+1)][col_start+2] += 1
        V[(j_off+1)*4 + (k_off+1)][col_start+2] -= 1
        W[(k_off+1)*3 + (i_off+0)][col_start+2] += 1
        W[(k_off+1)*3 + (i_off+1)][col_start+2] += 1
        
        # Term 3 (M4)
        U[i_off*3 + (j_off+1)][col_start+3] += 1
        V[(j_off+1)*4 + (k_off+0)][col_start+3] += 1
        V[(j_off+0)*4 + (k_off+0)][col_start+3] -= 1
        W[(k_off+0)*3 + (i_off+0)][col_start+3] += 1
        W[(k_off+1)*3 + (i_off+0)][col_start+3] += 1
        
        # Term 4 (M5)
        U[i_off*3 + (j_off+0)][col_start+4] += 1
        U[i_off*3 + (j_off+1)][col_start+4] += 1
        V[(j_off+1)*4 + (k_off+1)][col_start+4] += 1
        W[(k_off+0)*3 + (i_off+0)][col_start+4] -= 1
        W[(k_off+1)*3 + (i_off+0)][col_start+4] -= 1
        
        # Term 5 (M6)
        U[i_off*3 + (j_off+0)][col_start+5] += 1
        U[i_off*3 + (j_off+1)][col_start+5] -= 1
        V[(j_off+0)*4 + (k_off+0)][col_start+5] += 1
        V[(j_off+0)*4 + (k_off+1)][col_start+5] += 1
        W[(k_off+1)*3 + (i_off+1)][col_start+5] += 1
        W[(k_off+1)*3 + (i_off+0)][col_start+5] -= 1 # Wait, C11 = M1-M2+M3+M6. C10 = M2+M4.
        
        # Correction on Strassen mapping to W
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's re-verify the W updates for Strassen
        # M1: C00, C01, C10, C11 all +1? No.
        # M1 = (A00+A11)(B00+B11).
        # Contribution to C:
        # C00 += M1
        # C01 += M1 (from A11 B01? No. A11 B01 is part of M1? No. M1 is product of sums.)
        # Actually, Strassen works by computing linear combinations.
        # The W matrix entries should reflect the coefficients of M_r in C_{i,k}.
        
        # Let's restart the Strassen W mapping carefully.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Reset W for this block (subtract previous, or just overwrite if we are careful)
        # Since we are adding to a zero-initialized array, we can just set the values.
        # But we are using +=. So we must ensure we don't double count.
        # I will clear the W entries for this block before setting, or just trust the sum.
        # Better: Just define the W contributions directly.
        
        # M1 (col_start+0)
        W[(k_off+0)*3 + (i_off+0)][col_start+0] = 1  # C00
        W[(k_off+0)*3 + (i_off+1)][col_start+0] = 0  # C01 (M1 not in C01)
        W[(k_off+1)*3 + (i_off+0)][col_start+0] = 0  # C10
        W[(k_off+1)*3 + (i_off+1)][col_start+0] = 1  # C11
        
        # M2 (col_start+1)
        W[(k_off+0)*3 + (i_off+0)][col_start+1] = 0
        W[(k_off+0)*3 + (i_off+1)][col_start+1] = 0
        W[(k_off+1)*3 + (i_off+0)][col_start+1] = 1  # C10
        W[(k_off+1)*3 + (i_off+1)][col_start+1] = -1 # C11
        
        # M3 (col_start+2)
        W[(k_off+0)*3 + (i_off+0)][col_start+2] = 0
        W[(k_off+0)*3 + (i_off+1)][col_start+2] = 1  # C01
        W[(k_off+1)*3 + (i_off+0)][col_start+2] = 0
        W[(k_off+1)*3 + (i_off+1)][col_start+2] = 1  # C11
        
        # M4 (col_start+3)
        W[(k_off+0)*3 + (i_off+0)][col_start+3] = 1  # C00
        W[(k_off+0)*3 + (i_off+1)][col_start+3] = 0
        W[(k_off+1)*3 + (i_off+0)][col_start+3] = 1  # C10
        W[(k_off+1)*3 + (i_off+1)][col_start+3] = 0
        
        # M5 (col_start+4)
        W[(k_off+0)*3 + (i_off+0)][col_start+4] = -1 # C00
        W[(k_off+0)*3 + (i_off+1)][col_start+4] = 1  # C01
        W[(k_off+1)*3 + (i_off+0)][col_start+4] = 0
        W[(k_off+1)*3 + (i_off+1)][col_start+4] = 0
        
        # M6 (col_start+5)
        W[(k_off+0)*3 + (i_off+0)][col_start+5] = 0
        W[(k_off+0)*3 + (i_off+1)][col_start+5] = 0
        W[(k_off+1)*3 + (i_off+0)][col_start+5] = 0
        W[(k_off+1)*3 + (i_off+1)][col_start+5] = 1  # C11
        
        # M7 (col_start+6)
        W[(k_off+0)*3 + (i_off+0)][col_start+6] = 1  # C00
        W[(k_off+0)*3 + (i_off+1)][col_start+6] = 0
        W[(k_off+1)*3 + (i_off+0)][col_start+6] = 0
        W[(k_off+1)*3 + (i_off+1)][col_start+6] = 0

    # Call for Block 1 (i=0,1; j=0,1; k=0,1)
    add_strassen(0, 0, 0, 0)
    
    # Call for Block 5 (i=0,1; j=0,1; k=2,3)
    add_strassen(0, 0, 2, 7)
    
    # Outer products
    # Block 2: i in {0,1}, j=2, k in {0,1} -> Col 14
    for i in [0, 1]:
        for k in [0, 1]:
            U[i*3 + 2][14] = 1
            V[2*4 + k][14] = 1
            W[k*3 + i][14] = 1
            
    # Block 3: i=2, j in {0,1}, k in {0,1} -> Col 15
    for j in [0, 1]:
        for k in [0, 1]:
            U[2*3 + j][15] = 1
            V[j*4 + k][15] = 1
            W[k*3 + 2][15] = 1
            
    # Block 4: i=2, j=2, k in {0,1} -> Col 16
    for k in [0, 1]:
        U[2*3 + 2][16] = 1
        V[2*4 + k][16] = 1
        W[k*3 + 2][16] = 1
        
    # Block 6: i in {0,1}, j=2, k in {2,3} -> Col 17
    for i in [0, 1]:
        for k in [2, 3]:
            U[i*3 + 2][17] = 1
            V[2*4 + k][17] = 1
            W[k*3 + i][17] = 1
            
    # Block 7: i=2, j in {0,1}, k in {2,3} -> Col 18
    for j in [0, 1]:
        for k in [2, 3]:
            U[2*3 + j][18] = 1
            V[j*4 + k][18] = 1
            W[k*3 + 2][18] = 1
            
    # Block 8: i=2, j=2, k in {2,3} -> Col 19
    for k in [2, 3]:
        U[2*3 + 2][19] = 1
        V[2*4 + k][19] = 1
        W[k*3 + 2][19] = 1

    return {"U": U, "V": V, "W": W}
