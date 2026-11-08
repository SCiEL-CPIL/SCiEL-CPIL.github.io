"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    # Matrix multiplication rank-minimal decomposition via Strassen on 2x2x2 blocks.
    # Dimensions: A is 2x5, B is 5x6, C is 2x6.
    # W index: k*n + i where k in 0..5, i in 0..1 (12 rows)
    # U index: i*m + j where i in 0..1, j in 0..4 (10 rows)
    # V index: j*p + k where j in 0..4, k in 0..5 (30 rows)
    
    n, m, p = DIMENSIONS
    rank = 54  # 6 Strassen blocks * 7 + 1 schoolbook block * 12
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2: M = A*B where A is 2x2, B is 2x2
    # M11 = (A11+A22)(B11+B22) - M5
    # M12 = A22*(B12-B22) ... wait, standard Strassen:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # But we need the tensor decomposition, so each "M" is one rank-1 term.
    # For a Strassen block computing C_block = A_block * B_block:
    # We add 7 rank terms to U, V, W.
    
    def add_strassen_block(i_offset, j_offset, k_offset, rank_offset):
        # i_offset: row offset in A (0 or 1)
        # j_offset: col offset in A = row offset in B (0..4)
        # k_offset: col offset in B (0..5)
        # rank_offset: starting column in U,V,W
        # A_block rows: i_offset, i_offset+1; cols: j_offset, j_offset+1
        # B_block rows: j_offset, j_offset+1; cols: k_offset, k_offset+1
        # C_block rows: i_offset, i_offset+1; cols: k_offset, k_offset+1
        # W index for C element (i, k): k*n + i
        
        A_idx = [(i_offset, j_offset), (i_offset, j_offset+1), 
                 (i_offset+1, j_offset), (i_offset+1, j_offset+1)]
        B_idx = [(j_offset, k_offset), (j_offset, k_offset+1),
                 (j_offset+1, k_offset), (j_offset+1, k_offset+1)]
        C_idx = [(i_offset, k_offset), (i_offset, k_offset+1),
                 (i_offset+1, k_offset), (i_offset+1, k_offset+1)]
        
        # A11, A12, A21, A22
        a11 = A_idx[0]  # (i_offset, j_offset)
        a12 = A_idx[1]  # (i_offset, j_offset+1)
        a21 = A_idx[2]  # (i_offset+1, j_offset)
        a22 = A_idx[3]  # (i_offset+1, j_offset+1)
        
        # B11, B12, B21, B22
        b11 = B_idx[0]  # (j_offset, k_offset)
        b12 = B_idx[1]  # (j_offset, k_offset+1)
        b21 = B_idx[2]  # (j_offset+1, k_offset)
        b22 = B_idx[3]  # (j_offset+1, k_offset+1)
        
        # C11, C12, C21, C22
        c11 = C_idx[0]  # (i_offset, k_offset)
        c12 = C_idx[1]  # (i_offset, k_offset+1)
        c21 = C_idx[2]  # (i_offset+1, k_offset)
        c22 = C_idx[3]  # (i_offset+1, k_offset+1)
        
        # M1 = (A11+A22)*(B11+B22)
        U[a11[0]*m + a11[1]][rank_offset] += 1
        U[a22[0]*m + a22[1]][rank_offset] += 1
        V[b11[0]*p + b11[1]][rank_offset] += 1
        V[b22[0]*p + b22[1]][rank_offset] += 1
        W[c11[1]*n + c11[0]][rank_offset] += 1
        W[c12[1]*n + c12[0]][rank_offset] += 1
        W[c21[1]*n + c21[0]][rank_offset] += 1
        W[c22[1]*n + c22[0]][rank_offset] += 1
        
        # M2 = (A21+A22)*B11
        U[a21[0]*m + a21[1]][rank_offset+1] += 1
        U[a22[0]*m + a22[1]][rank_offset+1] += 1
        V[b11[0]*p + b11[1]][rank_offset+1] += 1
        W[c21[1]*n + c21[0]][rank_offset+1] += 1
        W[c22[1]*n + c22[0]][rank_offset+1] += 1
        
        # M3 = A11*(B12-B22)
        U[a11[0]*m + a11[1]][rank_offset+2] += 1
        V[b12[0]*p + b12[1]][rank_offset+2] += 1
        V[b22[0]*p + b22[1]][rank_offset+2] -= 1
        W[c12[1]*n + c12[0]][rank_offset+2] += 1
        W[c22[1]*n + c22[0]][rank_offset+2] += 1
        
        # M4 = A22*(B21-B11)
        U[a22[0]*m + a22[1]][rank_offset+3] += 1
        V[b21[0]*p + b21[1]][rank_offset+3] += 1
        V[b11[0]*p + b11[1]][rank_offset+3] -= 1
        W[c11[1]*n + c11[0]][rank_offset+3] += 1
        W[c22[1]*n + c22[0]][rank_offset+3] += 1
        
        # M5 = (A11+A12)*B22
        U[a11[0]*m + a11[1]][rank_offset+4] += 1
        U[a12[0]*m + a12[1]][rank_offset+4] += 1
        V[b22[0]*p + b22[1]][rank_offset+4] += 1
        W[c11[1]*n + c11[0]][rank_offset+4] -= 1
        W[c12[1]*n + c12[0]][rank_offset+4] += 1
        W[c21[1]*n + c21[0]][rank_offset+4] += 1
        
        # M6 = (A21-A11)*(B11+B12)
        U[a21[0]*m + a21[1]][rank_offset+5] += 1
        U[a11[0]*m + a11[1]][rank_offset+5] -= 1
        V[b11[0]*p + b11[1]][rank_offset+5] += 1
        V[b12[0]*p + b12[1]][rank_offset+5] += 1
        W[c21[1]*n + c21[0]][rank_offset+5] += 1
        
        # M7 = (A12-A22)*(B21+B22)
        U[a12[0]*m + a12[1]][rank_offset+6] += 1
        U[a22[0]*m + a22[1]][rank_offset+6] -= 1
        V[b21[0]*p + b21[1]][rank_offset+6] += 1
        V[b22[0]*p + b22[1]][rank_offset+6] += 1
        W[c11[1]*n + c11[0]][rank_offset+6] -= 1
        W[c22[1]*n + c22[0]][rank_offset+6] += 1
    
    # 6 Strassen blocks for j=0,1 and j=2,3 with k=0,1; k=2,3; k=4,5
    # Actually, we split B's 6 columns into 3 pairs: (0,1), (2,3), (4,5)
    # And A's 5 cols into (0,1), (2,3), (4). But we can only do Strassen on 2x2.
    # So: 
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1} -> Strassen
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3} -> Strassen
    # Block 3: i in {0,1}, j in {0,1}, k in {4,5} -> Strassen
    # Block 4: i in {0,1}, j in {2,3}, k in {0,1} -> Strassen
    # Block 5: i in {0,1}, j in {2,3}, k in {2,3} -> Strassen
    # Block 6: i in {0,1}, j in {2,3}, k in {4,5} -> Strassen
    # Block 7: i in {0,1}, j in {4}, k in {0..5} -> schoolbook (12 terms)
    
    r = 0
    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start, r)
            r += 7
    
    # Schoolbook for j=4, all k
    for i in range(2):
        for k in range(6):
            U[i*m + 4][r] = 1
            V[4*p + k][r] = 1
            W[k*n + i][r] = 1
            r += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
