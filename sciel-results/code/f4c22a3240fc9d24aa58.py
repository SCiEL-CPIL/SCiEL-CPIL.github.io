"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # We split the multiplication into:
    # 1. 3x3 block (cols 0-2)
    # 2. 3x3 block (cols 3-5)
    # 3. 3x1 block (col 6)
    
    # Rank for 3x3 block using block-strassen (rank 26)
    # Rank for 3x1 block is 9
    # Total rank = 26 + 26 + 9 = 61
    
    rank = 61
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    
    def add_3x3_block_terms(k_offset):
        """
        Adds 26 terms for a 3x3x3 multiplication block where B columns are k_offset to k_offset+2.
        A indices: i*m + j
        B indices: j*p + (k + k_offset)
        C indices: (k + k_offset)*n + i
        """
        nonlocal col
        
        # Helper to add a term
        def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
            U[u_idx][col] += u_val
            V[v_idx][col] += v_val
            W[w_idx][col] += w_val
        
        # 1. Strassen 2x2x2 for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        # A indices: 0(0,0), 1(0,1), 3(1,0), 4(1,1)
        # B indices: 0(0,0), 1(0,1), 3(1,0), 4(1,1) -> Global: j*7 + (k+k_off)
        # B[0,0] -> 0*7 + (0+k_off) = k_off
        # B[0,1] -> 0*7 + (1+k_off) = 1+k_off
        # B[1,0] -> 1*7 + (0+k_off) = 7+k_off
        # B[1,1] -> 1*7 + (1+k_off) = 8+k_off
        # C indices: k*n + i
        # C[0,0] -> (0+k_off)*3 + 0 = 3*k_off
        # C[0,1] -> (1+k_off)*3 + 0 = 3*(1+k_off)
        # C[1,0] -> (0+k_off)*3 + 1 = 3*k_off + 1
        # C[1,1] -> (1+k_off)*3 + 1 = 3*(1+k_off) + 1
        
        b00 = k_off
        b01 = 1 + k_off
        b10 = 7 + k_off
        b11 = 8 + k_off
        
        c00 = 3 * k_off
        c01 = 3 * (1 + k_off)
        c10 = 3 * k_off + 1
        c11 = 3 * (1 + k_off) + 1
        
        # P1 = (A00+A11)(B00+B11) -> C00+1, C11+1
        add_term(0, 1, b00, 1, c00, 1)
        add_term(4, 1, b11, 1, c00, 1)
        add_term(0, 1, b00, 1, c11, 1)
        add_term(4, 1, b11, 1, c11, 1)
        col += 1
        
        # P2 = (A10+A11)B00 -> C10+1
        add_term(3, 1, b00, 1, c10, 1)
        add_term(4, 1, b00, 1, c10, 1)
        col += 1
        
        # P3 = A00(B10-B11) -> C01+1
        add_term(0, 1, b10, 1, c01, 1)
        add_term(0, 1, b11, -1, c01, 1)
        col += 1
        
        # P4 = A11(B01-B00) -> C00-1, C10+1
        add_term(4, 1, b01, 1, c00, -1)
        add_term(4, 1, b00, -1, c00, -1)
        add_term(4, 1, b01, 1, c10, 1)
        add_term(4, 1, b00, -1, c10, 1)
        col += 1
        
        # P5 = (A00+A01)B11 -> C00-1, C01+1
        add_term(0, 1, b11, 1, c00, -1)
        add_term(1, 1, b11, 1, c00, -1)
        add_term(0, 1, b11, 1, c01, 1)
        add_term(1, 1, b11, 1, c01, 1)
        col += 1
        
        # P6 = (A10-A00)(B00+B01) -> C11+1
        add_term(3, 1, b00, 1, c11, 1)
        add_term(0, -1, b00, 1, c11, 1)
        add_term(3, 1, b01, 1, c11, 1)
        add_term(0, -1, b01, 1, c11, 1)
        col += 1
        
        # P7 = (A01-A11)(B10+B11) -> C00+1, C11-1
        add_term(1, 1, b10, 1, c00, 1)
        add_term(4, -1, b10, 1, c00, 1)
        add_term(1, 1, b11, 1, c00, 1)
        add_term(4, -1, b11, 1, c00, 1)
        add_term(1, 1, b10, 1, c11, -1)
        add_term(4, -1, b10, 1, c11, -1)
        add_term(1, 1, b11, 1, c11, -1)
        add_term(4, -1, b11, 1, c11, -1)
        col += 1
        
        # 2. A12 * B21 (2x1 * 1x2 -> 2x2)
        # A12: A[0,2]=2, A[1,2]=5
        # B21: B[2,0]=14+k_off, B[2,1]=15+k_off
        # C[0:2, 0:2]
        
        # T7: A[0,2]*B[2,0] -> C[0,0]
        add_term(2, 1, 14+k_off, 1, c00, 1)
        col += 1
        
        # T8: A[0,2]*B[2,1] -> C[0,1]
        add_term(2, 1, 15+k_off, 1, c01, 1)
        col += 1
        
        # T9: A[1,2]*B[2,0] -> C[1,0]
        add_term(5, 1, 14+k_off, 1, c10, 1)
        col += 1
        
        # T10: A[1,2]*B[2,1] -> C[1,1]
        add_term(5, 1, 15+k_off, 1, c11, 1)
        col += 1
        
        # 3. A11 * B12 (2x2 * 2x1 -> 2x1)
        # B12: B[0,2]=2+k_off, B[1,2]=9+k_off
        # C[0:2, 2] -> C indices: (2+k_off)*3 + i
        
        c02 = (2 + k_off) * 3
        c12 = (2 + k_off) * 3 + 1
        
        # T11: A[0,0]*B[0,2] -> C[0,2]
        add_term(0, 1, 2+k_off, 1, c02, 1)
        col += 1
        
        # T12: A[0,1]*B[1,2] -> C[0,2]
        add_term(1, 1, 9+k_off, 1, c02, 1)
        col += 1
        
        # T13: A[1,0]*B[0,2] -> C[1,2]
        add_term(3, 1, 2+k_off, 1, c12, 1)
        col += 1
        
        # T14: A[1,1]*B[1,2] -> C[1,2]
        add_term(4, 1, 9+k_off, 1, c12, 1)
        col += 1
        
        # 4. A12 * B22 (2x1 * 1x1 -> 2x1)
        # B22: B[2,2]=16+k_off
        
        # T15: A[0,2]*B[2,2] -> C[0,2]
        add_term(2, 1, 16+k_off, 1, c02, 1)
        col += 1
        
        # T16: A[1,2]*B[2,2] -> C[1,2]
        add_term(5, 1, 16+k_off, 1, c12, 1)
        col += 1
        
        # 5. A21 * B11 (1x2 * 2x2 -> 1x2)
        # A21: A[2,0]=6, A[2,1]=7
        # C[2, 0:2] -> C indices: k*3 + 2
        
        c20 = k_off * 3 + 2
        c21 = (1 + k_off) * 3 + 2
        
        # T17: A[2,0]*B[0,0] -> C[2,0]
        add_term(6, 1, b00, 1, c20, 1)
        col += 1
        
        # T18: A[2,1]*B[1,0] -> C[2,0]
        add_term(7, 1, b10, 1, c20, 1)
        col += 1
        
        # T19: A[2,0]*B[0,1] -> C[2,1]
        add_term(6, 1, b01, 1, c21, 1)
        col += 1
        
        # T20: A[2,1]*B[1,1] -> C[2,1]
        add_term(7, 1, b11, 1, c21, 1)
        col += 1
        
        # 6. A22 * B21 (1x1 * 1x2 -> 1x2)
        # A22: A[2,2]=8
        
        # T21: A[2,2]*B[2,0] -> C[2,0]
        add_term(8, 1, 14+k_off, 1, c20, 1)
        col += 1
        
        # T22: A[2,2]*B[2,1] -> C[2,1]
        add_term(8, 1, 15+k_off, 1, c21, 1)
        col += 1
        
        # 7. A21 * B12 (1x2 * 2x1 -> 1x1)
        # C[2,2] -> (2+k_off)*3 + 2
        
        c22 = (2 + k_off) * 3 + 2
        
        # T23: A[2,0]*B[0,2] -> C[2,2]
        add_term(6, 1, 2+k_off, 1, c22, 1)
        col += 1
        
        # T24: A[2,1]*B[1,2] -> C[2,2]
        add_term(7, 1, 9+k_off, 1, c22, 1)
        col += 1
        
        # 8. A22 * B22 (1x1 * 1x1 -> 1x1)
        
        # T25: A[2,2]*B[2,2] -> C[2,2]
        add_term(8, 1, 16+k_off, 1, c22, 1)
        col += 1

    # Block 1: cols 0-2
    add_3x3_block_terms(0)
    
    # Block 2: cols 3-5
    add_3x3_block_terms(3)
    
    # Block 3: col 6 (3x1)
    # Schoolbook: 9 terms
    for i in range(n):
        for j in range(m):
            k = 6
            U[i*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
            
    if time.time() - start_time > 110:
        # Safety margin
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
