"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    # Schoolbook decomposition rank = n*m*p = 32
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
    return {"U": U, "V": V, "W": W}

def _strassen_2x2(a, b):
    """Return list of 7 tuples (u_idx, v_idx, w_idx) for 2x2x2 Strassen.
    a: 2x2, b: 2x2. Indices 0,1,2,3 for 2x2 blocks in row-major.
    Returns rank-1 terms (a_idx, b_idx, c_idx) where c = a * b."""
    # Standard Strassen:
    # m1 = (a00+a11)*(b00+b11)
    # m2 = (a10+a11)*b00
    # m3 = a00*(b01-b11)
    # m4 = a11*(b10-b00)
    # m5 = (a00+a01)*b11
    # m6 = (a10-a00)*(b00+b01)
    # m7 = (a01-a11)*(b10+b11)
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    # This gives 7 products but with signs/sums. For tensor decomposition we need rank-1 terms.
    # Actually Strassen gives a decomposition of the 2x2x2 tensor into 7 rank-1 terms.
    # The rank-1 terms are:
    # (a00+a11, b00+b11, c00) with +1
    # (a00+a11, b00+b11, c11) with +1
    # (a10+a11, b00, c10) with +1
    # (a10+a11, b00, c11) with -1
    # (a00, b01-b11, c01) with +1
    # (a00, b01-b11, c11) with +1
    # (a11, b10-b00, c00) with +1
    # (a11, b10-b00, c10) with +1
    # (a00+a01, b11, c00) with -1
    # (a00+a01, b11, c01) with +1
    # (a10-a00, b00+b01, c11) with +1
    # (a01-a11, b10+b11, c00) with +1
    # (a01-a11, b10+b11, c11) with +1
    # Wait, that's 14 terms, not 7. Let me reconsider.
    # Actually the 7 rank-1 terms in Strassen are the 7 bilinear forms:
    # m1 = (a00+a11)(b00+b11) -> contributes to c00 (+1) and c11 (+1)
    # m2 = (a10+a11)(b00) -> contributes to c10 (+1) and c11 (-1)
    # m3 = (a00)(b01-b11) -> contributes to c01 (+1) and c11 (+1)
    # m4 = (a11)(b10-b00) -> contributes to c00 (+1) and c10 (+1)
    # m5 = (a00+a01)(b11) -> contributes to c00 (-1) and c01 (+1)
    # m6 = (a10-a00)(b00+b01) -> contributes to c11 (+1)
    # m7 = (a01-a11)(b10+b11) -> contributes to c00 (+1) and c11 (+1)
    # Each m_i is a rank-1 term in the tensor. The U, V, W vectors for each:
    # For m1: U has 1 at positions for a00 and a11, V has 1 at b00 and b11, W has 1 at c00 and c11
    # This is a valid rank-1 decomposition of the 2x2x2 tensor into 7 terms.
    pass

def _build_strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition.
    U: shape (4, 7), V: shape (4, 7), W: shape (4, 7)
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # m1 = (a00+a11)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    # m2 = (a10+a11)(b00)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    # m3 = (a00)(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    # m4 = (a11)(b10-b00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    # m5 = (a00+a01)(b11)
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    # m6 = (a10-a00)(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    # m7 = (a01-a11)(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    # Use block decomposition: split B into two 4x2 blocks
    # C_left = A * B_left (2x2 result), C_right = A * B_right (2x2 result)
    # Use Strassen for each 2x4x2 multiplication
    # For 2x4x2: A is 2x4, B is 4x2, C is 2x2
    # Split A into A_left (2x2) and A_right (2x2)
    # C = A_left * B_left + A_right * B_right
    # Each 2x2x2 uses Strassen rank 7, so total rank 14 for 2x4x2
    # For 2x4x4: split B into B_1 (4x2) and B_2 (4x2)
    # C_1 = A * B_1 (rank 14), C_2 = A * B_2 (rank 14), total rank 28
    
    U_str, V_str, W_str = _build_strassen_2x2x2()
    
    rank = 28
    U = [[0]*rank for _ in range(n*m)]  # 8x28
    V = [[0]*rank for _ in range(m*p)]  # 16x28
    W = [[0]*rank for _ in range(p*n)]  # 8x28
    
    # For each half of B (columns 0-1 and 2-3), use Strassen
    for half in range(2):
        base_col = half * 14
        # B block: rows 0-3, columns 2*half to 2*half+1
        # A: rows 0-1, cols 0-3, split into A_left (cols 0-1) and A_right (cols 2-3)
        # For Strassen on 2x2x2, the indices are:
        # A block: 2x2, B block: 2x2, C block: 2x2
        # For the left half of B (cols 0-1 of B, which are cols 0-1 of the 4x4):
        #   B_left corresponds to B[:, 0:2] in the full B
        #   In the 2x4x2, B is 4x2, with rows 0-3
        #   We split B into B_top (rows 0-1) and B_bottom (rows 2-3)? No.
        #   
        # Actually for A (2x4) * B (4x2):
        # A = [A1 | A2] where A1, A2 are 2x2
        # B = [B1; B2] where B1, B2 are 2x2
        # C = A1*B1 + A2*B2
        # Use Strassen for A1*B1 (rank 7) and A2*B2 (rank 7)
        
        # A1 = A[:, 0:2], A2 = A[:, 2:4]
        # B1 = B[0:2, 2*half:2*half+2], B2 = B[2:4, 2*half:2*half+2]
        
        for s in range(7):
            col = base_col + s
            # U: A has 8 rows (2x4). A1 is rows 0-1, cols 0-1 -> U indices 0-1
            # A2 is rows 0-1, cols 2-3 -> U indices 2-3
            # Actually U is indexed by (i,j) where i is row of A (0-1), j is col of A (0-3)
            # So U[i*4+j] for the 2x4 matrix
            # A1: i in 0-1, j in 0-1 -> U indices 0,1,4,5
            # A2: i in 0-1, j in 2-3 -> U indices 2,3,6,7
            
            # V: B has 16 rows (4x4). For half, B block is 4x2 at cols 2*half to 2*half+1
            # B is indexed by (j,k) where j is row of B (0-3), k is col of B (0-3)
            # V[j*4+k]
            # B1: j in 0-1, k in 2*half to 2*half+1 -> V indices j*4 + 2*half, j*4+2*half+1
            # B2: j in 2-3, k in 2*half to 2*half+1 -> V indices j*4 + 2*half, j*4+2*half+1
            
            # W: C has 8 rows (4x2? no, C is 2x4). Wait, C is 2x4.
            # W is indexed by (k,i) where k is col of C (0-3), i is row of C (0-1)
            # W[k*2+i]
            # C1: k in 2*half to 2*half+1, i in 0-1 -> W indices k*2+i
            # C2: same k range, same i range
            
            # For Strassen s-th term:
            # U_str has 4 entries (2x2 A block), V_str has 4 entries (2x2 B block), W_str has 4 entries (2x2 C block)
            
            # A1*B1:
            # A1 is 2x2 with entries at U[0], U[1], U[4], U[5] (row 0: cols 0,1; row 1: cols 0,1)
            # V_str indices: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
            # For A1: local (a,b) -> global U[a*4+b]
            # For B1: local (c,d) -> global V[c*4 + 2*half + d]
            # For C1: local (e,f) -> global W[(2*half+e)*2 + f]
            
            for a in range(2):
                for b in range(2):
                    u_idx_local = a*2+b
                    if U_str[u_idx_local][s] != 0:
                        U[a*4+b][col] += U_str[u_idx_local][s]
                        # V and W for B1
            for c in range(2):
                for d in range(2):
                    v_idx_local = c*2+d
                    if V_str[v_idx_local][s] != 0:
                        V[c*4 + 2*half + d][col] += V_str[v_idx_local][s]
            for e in range(2):
                for f in range(2):
                    w_idx_local = e*2+f
                    if W_str[w_idx_local][s] != 0:
                        W[(2*half+e)*2 + f][col] += W_str[w_idx_local][s]
            
            # A2*B2:
            # A2 is 2x2 with entries at U[2], U[3], U[6], U[7] (row 0: cols 2,3; row 1: cols 2,3)
            # For A2: local (a,b) -> global U[a*4+2+b]
            # For B2: local (c,d) -> global V[(2+c)*4 + 2*half + d]
            # For C2: local (e,f) -> global W[(2*half+e)*2 + f] (same C output)
            
            for a in range(2):
                for b in range(2):
                    u_idx_local = a*2+b
                    if U_str[u_idx_local][s] != 0:
                        U[a*4+2+b][col] += U_str[u_idx_local][s]
            for c in range(2):
                for d in range(2):
                    v_idx_local = c*2+d
                    if V_str[v_idx_local][s] != 0:
                        V[(2+c)*4 + 2*half + d][col] += V_str[v_idx_local][s]
            for e in range(2):
                for f in range(2):
                    w_idx_local = e*2+f
                    if W_str[w_idx_local][s] != 0:
                        W[(2*half+e)*2 + f][col] += W_str[w_idx_local][s]
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
