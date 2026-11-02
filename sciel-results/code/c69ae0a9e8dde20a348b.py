import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # We use a rank 28 decomposition based on Strassen's algorithm for 2x2 blocks.
    # C is 2x4. We split C into two 2x2 blocks C1 (cols 0-1) and C2 (cols 2-3).
    # C1 = A1 @ B11 + A2 @ B21
    # C2 = A1 @ B12 + A2 @ B22
    # Where A1 = A[:, 0:2], A2 = A[:, 2:4]
    # B11 = B[0:2, 0:2], B21 = B[2:4, 0:2]
    # B12 = B[0:2, 2:4], B22 = B[2:4, 2:4]
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2 multiplication
    # Local indices i, j, k in {0, 1}
    # U_local[i, j, r], V_local[j, k, r], W_local[k, i, r]
    
    # r=1: M1 = (A00+A11)(B00+B11)
    # U: (0,0)->1, (1,1)->1. V: (0,0)->1, (1,1)->1.
    # W: C00+=1, C11+=1.
    # C00: k=0, i=0. C11: k=1, i=1.
    
    # r=2: M2 = (A10+A11)B00
    # U: (1,0)->1, (1,1)->1. V: (0,0)->1.
    # W: C10+=1. (k=0, i=1)
    
    # r=3: M3 = A00(B01-B11)
    # U: (0,0)->1. V: (0,1)->1, (1,1)->-1.
    # W: C01+=1, C11+=1. (k=1, i=0), (k=1, i=1)
    
    # r=4: M4 = A11(B10-B00)
    # U: (1,1)->1. V: (1,0)->1, (0,0)->-1.
    # W: C00+=1, C10+=1. (k=0, i=0), (k=0, i=1)
    
    # r=5: M5 = (A00+A01)B11
    # U: (0,0)->1, (0,1)->1. V: (1,1)->1.
    # W: C00-=1, C01+=1. (k=0, i=0), (k=1, i=0)
    
    # r=6: M6 = (A10-A00)(B00+B01)
    # U: (1,0)->1, (0,0)->-1. V: (0,0)->1, (0,1)->1.
    # W: C11+=1. (k=1, i=1)
    
    # r=7: M7 = (A01-A11)(B10+B11)
    # U: (0,1)->1, (1,1)->-1. V: (1,0)->1, (1,1)->1.
    # W: C00+=1. (k=0, i=0)

    # Helper to add a Strassen block
    # a_row_offset, a_col_offset: offset for A (2x2 block)
    # b_row_offset, b_col_offset: offset for B (2x2 block)
    # c_col_offset: offset for C (2x2 block)
    # base_rank: starting rank index
    
    def add_strassen_block(a_r_off, a_c_off, b_r_off, b_c_off, c_c_off, base_r):
        # r=1
        U[a_r_off*4 + (a_c_off+0)][base_r+0] += 1
        U[a_r_off*4 + (a_c_off+1)][base_r+0] += 1
        V[(a_c_off+0)*4 + (b_r_off+0)][base_r+0] += 1
        V[(a_c_off+1)*4 + (b_r_off+1)][base_r+0] += 1
        W[(b_c_off+0)*2 + (a_r_off+0)][base_r+0] += 1 # C00
        W[(b_c_off+1)*2 + (a_r_off+1)][base_r+0] += 1 # C11
        
        # r=2
        U[a_r_off*4 + (a_c_off+1)][base_r+1] += 1
        U[a_r_off*4 + (a_c_off+1)][base_r+1] += 1 # Wait, A10 and A11. Row 1, Col 0 and Row 1, Col 1.
        # Correction: M2 = (A10+A11)B00.
        # A10 is row 1, col 0. A11 is row 1, col 1.
        # So U[1,0] and U[1,1] in local coords.
        U[a_r_off*4 + (a_c_off+1)][base_r+1] = 1 # A10
        U[a_r_off*4 + (a_c_off+1)][base_r+1] = 1 # A11 is same row? No.
        # Local A indices: (row, col).
        # A10: row 1, col 0. Global: a_r_off+1, a_c_off+0.
        # A11: row 1, col 1. Global: a_r_off+1, a_c_off+1.
        U[a_r_off*4 + (a_c_off+0)][base_r+1] = 1
        U[a_r_off*4 + (a_c_off+1)][base_r+1] = 1
        V[(a_c_off+0)*4 + (b_r_off+0)][base_r+1] += 1 # B00. Local (0,0). Global row b_r_off+0, col b_c_off+0?
        # V index is j*p + k. j is local col of A, which is local row of B.
        # Local B row 0, col 0.
        # Global B row b_r_off+0, col b_c_off+0.
        # But V is indexed by j (from A) and k (from B).
        # j_local=0, k_local=0.
        # Global j = a_c_off + j_local. Global k = b_c_off + k_local.
        # Wait, V shape is m x p. Index j*p + k.
        # j is the column of A, which is the row of B.
        # In our block, A cols are a_c_off, a_c_off+1.
        # B rows are b_r_off, b_r_off+1.
        # So j_global = a_c_off + j_local.
        # k_global = b_c_off + k_local.
        # V index: j_global * p + k_global.
        V[(a_c_off+0)*4 + (b_c_off+0)][base_r+1] += 1
        W[(b_c_off+0)*2 + (a_r_off+1)][base_r+1] += 1 # C10. k=0, i=1.
        
        # r=3
        # M3 = A00(B01-B11)
        # A00: local (0,0). Global (a_r_off+0, a_c_off+0).
        U[a_r_off*4 + (a_c_off+0)][base_r+2] += 1
        # B01: local (0,1). Global row b_r_off+0, col b_c_off+1.
        # B11: local (1,1). Global row b_r_off+1, col b_c_off+1.
        V[(a_c_off+0)*4 + (b_c_off+1)][base_r+2] += 1
        V[(a_c_off+1)*4 + (b_c_off+1)][base_r+2] += -1
        # W: C01 (k=1, i=0), C11 (k=1, i=1)
        W[(b_c_off+1)*2 + (a_r_off+0)][base_r+2] += 1
        W[(b_c_off+1)*2 + (a_r_off+1)][base_r+2] += 1
        
        # r=4
        # M4 = A11(B10-B00)
        # A11: local (1,1). Global (a_r_off+1, a_c_off+1).
        U[a_r_off*4 + (a_c_off+1)][base_r+3] += 1
        # B10: local (1,0). Global row b_r_off+1, col b_c_off+0.
        # B00: local (0,0). Global row b_r_off+0, col b_c_off+0.
        V[(a_c_off+1)*4 + (b_c_off+0)][base_r+3] += 1
        V[(a_c_off+0)*4 + (b_c_off+0)][base_r+3] += -1
        # W: C00 (k=0, i=0), C10 (k=0, i=1)
        W[(b_c_off+0)*2 + (a_r_off+0)][base_r+3] += 1
        W[(b_c_off+0)*2 + (a_r_off+1)][base_r+3] += 1
        
        # r=5
        # M5 = (A00+A01)B11
        # A00: (0,0), A01: (0,1).
        U[a_r_off*4 + (a_c_off+0)][base_r+4] += 1
        U[a_r_off*4 + (a_c_off+1)][base_r+4] += 1
        # B11: (1,1).
        V[(a_c_off+1)*4 + (b_c_off+1)][base_r+4] += 1
        # W: C00 (k=0, i=0) -=1, C01 (k=1, i=0) +=1
        W[(b_c_off+0)*2 + (a_r_off+0)][base_r+4] += -1
        W[(b_c_off+1)*2 + (a_r_off+0)][base_r+4] += 1
        
        # r=6
        # M6 = (A10-A00)(B00+B01)
        # A10: (1,0), A00: (0,0).
        U[a_r_off*4 + (a_c_off+0)][base_r+5] += 1
        U[a_r_off*4 + (a_c_off+0)][base_r+5] += -1 # A00 is (0,0). Row 0, Col 0.
        # Correction: A10 is row 1, col 0. A00 is row 0, col 0.
        U[a_r_off*4 + (a_c_off+0)][base_r+5] = 1 # A10? No, A10 is row 1.
        # Let's restart r=6 carefully.
        # Local A:
        # A10: row 1, col 0. Global: a_r_off+1, a_c_off+0.
        # A00: row 0, col 0. Global: a_r_off+0, a_c_off+0.
        U[a_r_off*4 + (a_c_off+0)][base_r+5] = 1 # A10
        U[a_r_off*4 + (a_c_off+0)][base_r+5] = -1 # A00 -- Wait, same index?
        # A10 index: (a_r_off+1)*4 + (a_c_off+0).
        # A00 index: (a_r_off+0)*4 + (a_c_off+0).
        U[(a_r_off+1)*4 + (a_c_off+0)][base_r+5] = 1
        U[(a_r_off+0)*4 + (a_c_off+0)][base_r+5] = -1
        
        # B00: (0,0), B01: (0,1).
        # B00: row 0, col 0. Global: b_r_off+0, b_c_off+0.
        # B01: row 0, col 1. Global: b_r_off+0, b_c_off+1.
        # V index: j*p + k.
        # For B00: j=0, k=0. Global j=a_c_off+0, k=b_c_off+0.
        # For B01: j=0, k=1. Global j=a_c_off+0, k=b_c_off+1.
        V[(a_c_off+0)*4 + (b_c_off+0)][base_r+5] += 1
        V[(a_c_off+0)*4 + (b_c_off+1)][base_r+5] += 1
        
        # W: C11 (k=1, i=1) +=1
        W[(b_c_off+1)*2 + (a_r_off+1)][base_r+5] += 1
        
        # r=7
        # M7 = (A01-A11)(B10+B11)
        # A01: (0,1), A11: (1,1).
        # A01: row 0, col 1. Global: a_r_off+0, a_c_off+1.
        # A11: row 1, col 1. Global: a_r_off+1, a_c_off+1.
        U[(a_r_off+0)*4 + (a_c_off+1)][base_r+6] = 1
        U[(a_r_off+1)*4 + (a_c_off+1)][base_r+6] = -1
        
        # B10: (1,0), B11: (1,1).
        # B10: row 1, col 0. Global: b_r_off+1, b_c_off+0.
        # B11: row 1, col 1. Global: b_r_off+1, b_c_off+1.
        # V index: j*p + k.
        # For B10: j=1, k=0. Global j=a_c_off+1, k=b_c_off+0.
        # For B11: j=1, k=1. Global j=a_c_off+1, k=b_c_off+1.
        V[(a_c_off+1)*4 + (b_c_off+0)][base_r+6] += 1
        V[(a_c_off+1)*4 + (b_c_off+1)][base_r+6] += 1
        
        # W: C00 (k=0, i=0) +=1
        W[(b_c_off+0)*2 + (a_r_off+0)][base_r+6] += 1

    # Block 1: C1 = A1 @ B11 + A2 @ B21
    # A1: rows 0-1, cols 0-1. a_r_off=0, a_c_off=0.
    # B11: rows 0-1, cols 0-1. b_r_off=0, b_c_off=0.
    # C1: cols 0-1. c_c_off=0.
    # This computes A1 @ B11.
    add_strassen_block(0, 0, 0, 0, 0, 0)
    
    # A2: rows 0-1, cols 2-3. a_r_off=0, a_c_off=2.
    # B21: rows 2-3, cols 0-1. b_r_off=2, b_c_off=0.
    # C1: cols 0-1. c_c_off=0.
    # This computes A2 @ B21.
    add_strassen_block(0, 2, 2, 0, 0, 7)
    
    # Block 2: C2 = A1 @ B12 + A2 @ B22
    # A1: rows 0-1, cols 0-1. a_r_off=0, a_c_off=0.
    # B12: rows 0-1, cols 2-3. b_r_off=0, b_c_off=2.
    # C2: cols 2-3. c_c_off=2.
    # This computes A1 @ B12.
    add_strassen_block(0, 0, 0, 2, 2, 14)
    
    # A2: rows 0-1, cols 2-3. a_r_off=0, a_c_off=2.
    # B22: rows 2-3, cols 2-3. b_r_off=2, b_c_off=2.
    # C2: cols 2-3. c_c_off=2.
    # This computes A2 @ B22.
    add_strassen_block(0, 2, 2, 2, 2, 21)

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
