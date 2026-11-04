"""Rewritten program for 3x4x5 matrix multiplication using Strassen and schoolbook blocks."""

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Define indices for blocks
    # n: 0,1 (size 2), 2 (size 1)
    # m: 0,1 (size 2), 2,3 (size 2)
    # p: 0,1 (size 2), 2,3 (size 2), 4 (size 1)
    
    n1_start, n1_end = 0, 2
    n2_start, n2_end = 2, 3
    
    m1_start, m1_end = 0, 2
    m2_start, m2_end = 2, 4
    
    p1_start, p1_end = 0, 2
    p2_start, p2_end = 2, 4
    p3_start, p3_end = 4, 5
    
    # Total rank calculation
    # Strassen 2x2x2: 7
    # Schoolbook 2x2x1: 4
    # Schoolbook 1x2x2: 2
    # Schoolbook 1x2x1: 1
    
    rank_total = 0
    # n1, m1
    rank_total += 7 # p1
    rank_total += 7 # p2
    rank_total += 4 # p3
    # n1, m2
    rank_total += 7 # p1
    rank_total += 7 # p2
    rank_total += 4 # p3
    # n2, m1
    rank_total += 2 # p1
    rank_total += 2 # p2
    rank_total += 1 # p3
    # n2, m2
    rank_total += 2 # p1
    rank_total += 2 # p2
    rank_total += 1 # p3
    
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal col
        # i: i_off, i_off+1
        # j: j_off, j_off+1
        # k: k_off, k_off+1
        
        # A indices: i_off*m + j, i_off*m + j+1, (i_off+1)*m + j, (i_off+1)*m + j+1
        # B indices: j_off*p + k, j_off*p + k+1, (j_off+1)*p + k, (j_off+1)*p + k+1
        # C indices: k_off*n + i, k_off*n + i+1, (k_off+1)*n + i, (k_off+1)*n + i+1
        
        a00 = i_off * m + j_off
        a01 = i_off * m + j_off + 1
        a10 = (i_off + 1) * m + j_off
        a11 = (i_off + 1) * m + j_off + 1
        
        b00 = j_off * p + k_off
        b01 = j_off * p + k_off + 1
        b10 = (j_off + 1) * p + k_off
        b11 = (j_off + 1) * p + k_off + 1
        
        c00 = k_off * n + i_off
        c01 = k_off * n + i_off + 1
        c10 = (k_off + 1) * n + i_off
        c11 = (k_off + 1) * n + i_off + 1
        
        # M1 = (A00 + A11)(B00 + B11)
        U[a00][col] = 1; U[a11][col] = 1
        V[b00][col] = 1; V[b11][col] = 1
        W[c00][col] = 1; W[c11][col] = 1
        W[c01][col] = -1; W[c10][col] = -1 # C00 += M1, C11 -= M1? No.
        # Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Reset W for this block to be safe, though we are adding to zeros initially
        # We need to add contributions to W.
        # Let's just define the 7 terms and their contributions to C.
        
        # It's easier to just list the (U, V, W) triples for the 7 terms.
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        U[a00][col] += 1; U[a11][col] += 1
        V[b00][col] += 1; V[b11][col] += 1
        W[c00][col] += 1; W[c11][col] += 1
        col += 1
        
        # Term 2: M2 = (A10+A11)B00
        U[a10][col] += 1; U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1; W[c11][col] -= 1
        col += 1
        
        # Term 3: M3 = A00(B01-B11)
        U[a00][col] += 1
        V[b01][col] += 1; V[b11][col] -= 1
        W[c01][col] += 1; W[c11][col] += 1
        col += 1
        
        # Term 4: M4 = A11(B10-B00)
        U[a11][col] += 1
        V[b10][col] += 1; V[b00][col] -= 1
        W[c00][col] += 1; W[c10][col] += 1
        col += 1
        
        # Term 5: M5 = (A00+A01)B11
        U[a00][col] += 1; U[a01][col] += 1
        V[b11][col] += 1
        W[c00][col] -= 1; W[c01][col] += 1
        col += 1
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        U[a10][col] += 1; U[a00][col] -= 1
        V[b00][col] += 1; V[b01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # Term 7: M7 = (A01-A11)(B10+B11)
        U[a01][col] += 1; U[a11][col] -= 1
        V[b10][col] += 1; V[b11][col] += 1
        W[c00][col] += 1
        col += 1

    def add_schoolbook_221(i_off, j_off, k_off):
        nonlocal col
        # A: 2x2, B: 2x1, C: 2x1
        a00 = i_off * m + j_off
        a01 = i_off * m + j_off + 1
        a10 = (i_off + 1) * m + j_off
        a11 = (i_off + 1) * m + j_off + 1
        
        b00 = j_off * p + k_off
        b10 = (j_off + 1) * p + k_off
        
        c00 = k_off * n + i_off
        c10 = (k_off + 1) * n + i_off # Wait, C is p x n.
        # C index is k*n + i.
        # k is fixed (k_off).
        # So C indices are k_off*n + i_off and k_off*n + i_off+1.
        
        c0 = k_off * n + i_off
        c1 = k_off * n + i_off + 1
        
        # C0 = A00 B00 + A01 B10
        # C1 = A10 B00 + A11 B10
        
        U[a00][col] = 1; V[b00][col] = 1; W[c0][col] = 1
        col += 1
        U[a01][col] = 1; V[b10][col] = 1; W[c0][col] = 1
        col += 1
        U[a10][col] = 1; V[b00][col] = 1; W[c1][col] = 1
        col += 1
        U[a11][col] = 1; V[b10][col] = 1; W[c1][col] = 1
        col += 1

    def add_schoolbook_122(i_idx, j_off, k_off):
        nonlocal col
        # A: 1x2, B: 2x2, C: 1x2
        a0 = i_idx * m + j_off
        a1 = i_idx * m + j_off + 1
        
        b00 = j_off * p + k_off
        b01 = j_off * p + k_off + 1
        b10 = (j_off + 1) * p + k_off
        b11 = (j_off + 1) * p + k_off + 1
        
        # C index k*n + i
        c0 = k_off * n + i_idx
        c1 = (k_off + 1) * n + i_idx
        
        # C0 = A0 B00 + A1 B10
        # C1 = A0 B01 + A1 B11
        
        U[a0][col] = 1; V[b00][col] = 1; W[c0][col] = 1
        col += 1
        U[a1][col] = 1; V[b10][col] = 1; W[c0][col] = 1
        col += 1
        U[a0][col] = 1; V[b01][col] = 1; W[c1][col] = 1
        col += 1
        U[a1][col] = 1; V[b11][col] = 1; W[c1][col] = 1
        col += 1

    def add_schoolbook_121(i_idx, j_off, k_off):
        nonlocal col
        # A: 1x2, B: 2x1, C: 1x1
        a0 = i_idx * m + j_off
        a1 = i_idx * m + j_off + 1
        
        b0 = j_off * p + k_off
        b1 = (j_off + 1) * p + k_off
        
        c = k_off * n + i_idx
        
        # C = A0 B0 + A1 B1
        
        U[a0][col] = 1; V[b0][col] = 1; W[c][col] = 1
        col += 1
        U[a1][col] = 1; V[b1][col] = 1; W[c][col] = 1
        col += 1

    # Fill blocks
    # n1, m1
    add_strassen_block(n1_start, m1_start, p1_start)
    add_strassen_block(n1_start, m1_start, p2_start)
    add_schoolbook_221(n1_start, m1_start, p3_start)
    
    # n1, m2
    add_strassen_block(n1_start, m2_start, p1_start)
    add_strassen_block(n1_start, m2_start, p2_start)
    add_schoolbook_221(n1_start, m2_start, p3_start)
    
    # n2, m1
    add_schoolbook_122(n2_start, m1_start, p1_start)
    add_schoolbook_122(n2_start, m1_start, p2_start)
    add_schoolbook_121(n2_start, m1_start, p3_start)
    
    # n2, m2
    add_schoolbook_122(n2_start, m2_start, p1_start)
    add_schoolbook_122(n2_start, m2_start, p2_start)
    add_schoolbook_121(n2_start, m2_start, p3_start)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
