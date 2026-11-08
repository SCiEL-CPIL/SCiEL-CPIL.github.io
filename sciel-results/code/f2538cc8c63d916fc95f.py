"""Matrix multiplication 4x5x5 - rank minimization via block decomposition."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

def _mul444_strassen(A, B):
    """Multiply two 4x4 matrices using Strassen's algorithm (rank 49 decomposition).
    Returns the 4x4 product."""
    # Standard Strassen for 2x2 blocks, applied to 4x4 = 2x2 blocks of 2x2
    a00, a01 = A[0:2, 0:2], A[0:2, 2:4]
    a10, a11 = A[2:4, 0:2], A[2:4, 2:4]
    b00, b01 = B[0:2, 0:2], B[0:2, 2:4]
    b10, b11 = B[2:4, 0:2], B[2:4, 2:4]
    
    M1 = (a00 + a11) @ (b00 + b11)
    M2 = (a10 + a11) @ b00
    M3 = a00 @ (b01 - b11)
    M4 = a11 @ (b10 - b00)
    M5 = (a00 + a01) @ b11
    M6 = (a10 - a00) @ (b00 + b01)
    M7 = (a01 - a11) @ (b10 + b11)
    
    C = np.zeros((4, 4))
    C[0:2, 0:2] = M1 + M4 - M5 + M7
    C[0:2, 2:4] = M3 + M5
    C[2:4, 0:2] = M2 + M4
    C[2:4, 2:4] = M1 - M2 + M3 + M6
    
    return C

def construct():
    """Construct rank-minimal tensor decomposition for 4x5 x 5x5 matrix multiplication.
    
    Uses block decomposition:
    - Top-left 4x4 block: Strassen rank 49
    - Boundary blocks: schoolbook
    
    Total rank = 49 + 16 + 16 + 4 = 85
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # We'll build U (20 x R), V (25 x R), W (20 x R)
    # Index conventions:
    # U: row index = i*m + j (i in 0..3, j in 0..4)
    # V: row index = j*p + k (j in 0..4, k in 0..4)
    # W: row index = k*n + i (k in 0..4, i in 0..3) -- trace convention W[k*n+i, r]
    
    # Block structure:
    # A is 4x5: rows i (0..3), cols j (0..4)
    # B is 5x5: rows j (0..4), cols k (0..4)
    # C is 4x5: rows i (0..3), cols k (0..4)
    
    # Split j into j0 (0..3) and j1 (4)
    # Split k into k0 (0..3) and k1 (4)
    
    # Block 1: A[0:4, 0:4] @ B[0:4, 0:4] -> C[0:4, 0:4] (4x4 @ 4x4) - Strassen rank 49
    # Block 2: A[0:4, 4:5] @ B[4:5, 0:4] -> C[0:4, 0:4] (4x1 @ 1x4) - schoolbook rank 16
    # Block 3: A[0:4, 0:4] @ B[0:4, 4:5] -> C[0:4, 4:5] (4x4 @ 4x1) - schoolbook rank 16
    # Block 4: A[0:4, 4:5] @ B[4:5, 4:5] -> C[0:4, 4:5] (4x1 @ 1x1) - schoolbook rank 4
    
    # Total rank = 49 + 16 + 16 + 4 = 85
    
    R = 85
    U = np.zeros((20, R))
    V = np.zeros((25, R))
    W = np.zeros((20, R))
    
    # --- Block 1: 4x4 @ 4x4 via Strassen ---
    # For each Strassen multiplication M = X @ Y, we need rank-7 factorizations
    # for the 2x2 @ 2x2 subproblems within Strassen.
    # 
    # Actually, let's think about this more carefully. The Strassen algorithm
    # for 4x4 uses 7 multiplications of 2x2 matrices. Each 2x2 multiplication
    # has rank 7 (Strassen). So total rank = 7 * 7 = 49.
    #
    # Let's build this explicitly.
    # 
    # For a 4x4 = 2x2 block matrix multiplication using Strassen:
    # We partition A into [[A00, A01], [A10, A11]] and B into [[B00, B01], [B10, B11]]
    # where each block is 2x2.
    #
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # Each Mx is a 2x2 matrix product, which itself can be done with rank 7.
    #
    # For a 2x2 matrix product P @ Q = R, the rank-7 Strassen decomposition is:
    # Let P = [[p00, p01], [p10, p11]], Q = [[q00, q01], [q10, q11]]
    # 
    # s1 = (p00+p11)(q00+q11)  -> but this isn't right for rank 7.
    # 
    # Actually the standard rank-7 for 2x2 is:
    # N1 = (p00+p11)(q00+q11)
    # N2 = (p10+p11)q00
    # N3 = p00(q01-q11)
    # N4 = p11(q10-q00)
    # N5 = (p00+p01)q11
    # N6 = (p10-p00)(q00+q01)
    # N7 = (p01-p11)(q10+q11)
    #
    # This gives rank 7 for 2x2 multiplication.
    
    # For each of the 7 Strassen products in the 4x4 multiplication,
    # we use rank-7 decomposition of the 2x2 multiplication.
    # Total rank = 7 * 7 = 49.
    
    # Let's define the rank-7 2x2 multiplication factors.
    # For P @ Q where P has entries indexed by (pi, pj) and Q by (qj, qk),
    # and result R by (ri, rk) where ri=pi, rk=qk.
    
    # The 7 terms for 2x2 Strassen:
    # term 1: (p00+p11) * (q00+q11)
    # term 2: (p10+p11) * q00
    # term 3: p00 * (q01-q11)
    # term 4: p11 * (q10-q00)
    # term 5: (p00+p01) * q11
    # term 6: (p10-p00) * (q00+q01)
    # term 7: (p01-p11) * (q10+q11)
    
    # For each term t, we have:
    # U_factor[pi][t] = coefficient of p_pi in the first factor
    # V_factor[qj][t] = coefficient of q_qj in the second factor
    # W_factor[ri, rk][t] = ... but we need to be careful with signs.
    
    # Actually, let's just verify by constructing the full 49-rank decomposition
    # and checking it works.
    
    # 2x2 Strassen factors (for a single 2x2 multiplication):
    # U2x2: shape (4, 7) where row = pi*2+pj, col = term index
    # V2x2: shape (4, 7) where row = qj*2+qk, col = term index  
    # W2x2: shape (4, 7) where row = ri*2+rk, col = term index
    
    U2x2 = np.zeros((4, 7))
    V2x2 = np.zeros((4, 7))
    W2x2 = np.zeros((4, 7))
    
    # Term 0: (p00+p11)(q00+q11)
    U2x2[0, 0] = 1  # p00
    U2x2[3, 0] = 1  # p11
    V2x2[0, 0] = 1  # q00
    V2x2[3, 0] = 1  # q11
    # Result contribution to r00, r01, r10, r11
    # r00 += p00*q00 + p00*q11 + p11*q00 + p11*q11... 
    # Actually the product (p00+p11)(q00+q11) contributes to all four entries of the result
    # with appropriate coefficients when combined with other terms.
    # 
    # The standard Strassen result:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # So for the 2x2 case, each Mi contributes to specific output entries with ±1 coefficients.
    
    # Let me redo this properly. For 2x2 Strassen:
    # M1 = (p00+p11)(q00+q11)
    # M2 = (p10+p11)q00
    # M3 = p00(q01-q11)
    # M4 = p11(q10-q00)
    # M5 = (p00+p01)q11
    # M6 = (p10-p00)(q00+q01)
    # M7 = (p01-p11)(q10+q11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # W2x2[row, col] where row = ri*2+rk
    W2x2[0, 0] = 1   # C00 gets +M1
    W2x2[1, 0] = 0   # C01
    W2x2[2, 0] = 0   # C10
    W2x2[3, 0] = 1   # C11 gets +M1
    
    W2x2[0, 1] = 0   # C00
    W2x2[1, 1] = 0   # C01
    W2x2[2, 1] = 1   # C10 gets +M2
    W2x2[3, 1] = -1  # C11 gets -M2
    
    W2x2[0, 2] = 0   # C00
    W2x2[1, 2] = 1   # C01 gets +M3
    W2x2[2, 2] = 0   # C10
    W2x2[3, 2] = 1   # C11 gets +M3
    
    W2x2[0, 3] = 1   # C00 gets +M4
    W2x2[1, 3] = 0   # C01
    W2x2[2, 3] = 1   # C10 gets +M4
    W2x2[3, 3] = 0   # C11
    
    W2x2[0, 4] = -1  # C00 gets -M5
    W2x2[1, 4] = 1   # C01 gets +M5
    W2x2[2, 4] = 0   # C10
    W2x2[3, 4] = 0   # C11
    
    W2x2[0, 5] = 0   # C00
    W2x2[1, 5] = 0   # C01
    W2x2[2, 5] = 0   # C10
    W2x2[3, 5] = 1   # C11 gets +M6
    
    W2x2[0, 6] = 1   # C00 gets +M7
    W2x2[1, 6] = 0   # C01
    W2x2[2, 6] = 0   # C10
    W2x2[3, 6] = 0   # C11
    
    # Verify: for each term, the U2x2 and V2x2 should correctly capture the linear combinations.
    
    # Now for the 4x4 Strassen, we have 7 terms, each being a 2x2 product.
    # The 7 terms at the 4x4 level:
    # M1 = (A00+A11)(B00+B11) - this is a 2x2 @ 2x2
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For each 4x4-level term Mi, we decompose the 2x2 product using the rank-7 2x2 Strassen.
    # So for term i, we have 7 sub-terms, each with:
    # - A coefficient for the A-block entries (from the 4x4-level linear combination)
    # - A coefficient for the B-block entries (from the 4x4-level linear combination)
    # - Then the 2x2 Strassen factors for the actual multiplication
    
    # 4x4-level coefficients for A blocks:
    # A00 is at global positions (i,j) with i in 0..1, j in 0..1
    # A01 is at (i,j) with i in 0..1, j in 2..3
    # A10 is at (i,j) with i in 2..3, j in 0..1
    # A11 is at (i,j) with i in 2..3, j in 2..3
    
    # For each 4x4 term Mi, the A-factor is a linear combination of A00, A01, A10, A11.
    # Similarly for B.
    
    # Let's define the 4x4-level linear combinations:
    # Term 0 (M1): A part = A00 + A11, B part = B00 + B11
    # Term 1 (M2): A part = A10 + A11, B part = B00
    # Term 2 (M3): A part = A00, B part = B01 - B11
    # Term 3 (M4): A part = A11, B part = B10 - B00
    # Term 4 (M5): A part = A00 + A01, B part = B11
    # Term 5 (M6): A part = A10 - A00, B part = B00 + B01
    # Term 6 (M7): A part = A01 - A11, B part = B10 + B11
    
    # For each term i and sub-term s (the 2x2 Strassen term), the overall factor is:
    # U coefficient = (4x4 A coefficient for term i) * (2x2 U factor for sub-term s)
    # V coefficient = (4x4 B coefficient for term i) * (2x2 V factor for sub-term s)
    # W coefficient = (4x4 C combination coefficient) * (2x2 W factor for sub-term s)
    
    # 4x4 C combination:
    # C00 (top-left 2x2) = M1 + M4 - M5 + M7
    # C01 (top-right 2x2) = M3 + M5
    # C10 (bottom-left 2x2) = M2 + M4
    # C11 (bottom-right 2x2) = M1 - M2 + M3 + M6
    
    # So for output block Cxy (x,y in 0..1), the coefficient for Mi is:
    # C00: [1, 0, 0, 1, -1, 0, 1] for M1..M7
    # C01: [0, 0, 1, 0, 1, 0, 0]
    # C10: [0, 1, 0, 1, 0, 0, 0]
    # C11: [1, -1, 1, 0, 0, 1, 0]
    
    c_coeff = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 0]    # C11
    ])
    
    # A block coefficients for each 4x4 term:
    # Term i: which A blocks and with what coefficients
    # A00, A01, A10, A11
    a_block_coeff = np.array([
        [1, 0, 0, 1],   # M1: A00 + A11
        [0, 0, 1, 1],   # M2: A10 + A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00 + A01
        [-1, 0, 1, 0],  # M6: A10 - A00
        [0, 1, 0, -1]   # M7: A01 - A11
    ])
    
    # B block coefficients for each 4x4 term:
    b_block_coeff = np.array([
        [1, 0, 0, 1],   # M1: B00 + B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01 - B11
        [0, 0, 1, -1],  # M4: B10 - B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00 + B01
        [0, 0, 1, 1]    # M7: B10 + B11
    ])
    
    # Now build the full U, V, W for the 4x4 @ 4x4 block.
    # Global U row: i*4 + j where i in 0..3, j in 0..3 (for the 4x4 part)
    # Global V row: j*4 + k where j in 0..3, k in 0..3
    # Global W row: k*4 + i where k in 0..3, i in 0..3
    
    # For each 4x4 term ti (0..6) and 2x2 sub-term si (0..6):
    # The sub-term si in 2x2 Strassen has:
    # - U2x2 factors for the 4 entries of the 2x2 A-matrix
    # - V2x2 factors for the 4 entries of the 2x2 B-matrix
    # - W2x2 factors for the 4 entries of the 2x2 C-matrix
    
    # The 2x2 A-matrix for term ti has blocks Axy where x,y in 0..1.
    # The global position of A[x,y] entry (local i2, j2) is at global (x*2+i2, y*2+j2).
    # So global row = (x*2+i2)*4 + (y*2+j2) = x*8 + i2*4 + y*2 + j2
    
    # Similarly for B: block Bxy, global row = (x*2+j2)*4 + (y*2+k2) where x,y in 0..1 for blocks, j2,k2 in 0..1 local.
    # Wait, B is 4x4 with blocks B00, B01, B10, B11.
    # B block xy is at rows x*2..x*2+1, cols y*2..y*2+1 in the 4x4 B.
    # Global V row for B entry (row r, col c) = r*4 + c.
    
    # For C: block Cxy is at rows x*2..x*2+1, cols y*2..y*2+1 in the 4x4 C.
    # Global W row for C entry (row r, col c) = c*4 + r (trace convention: k*n+i where k is col, i is row).
    
    col_offset = 0
    for ti in range(7):
        for si in range(7):
            r_idx = col_offset + ti * 7 + si
            
            # A coefficients: for each A block x,y and local entry (i2,j2)
            for ax in range(2):
                for ay in range(2):
                    coeff_a = a_block_coeff[ti, ax * 2 + ay]
                    if coeff_a == 0:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            # U2x2 index for A entry (i2, j2) in the 2x2 matrix
                            u2_idx = i2 * 2 + j2
                            u2_val = U2x2[u2_idx, si]
                            if u2_val == 0:
                                continue
                            # Global U row
                            gi = ax * 2 + i2
                            gj = ay * 2 + j2
                            global_u_row = gi * 4 + gj
                            U[global_u_row, r_idx] += coeff_a * u2_val
            
            # B coefficients: for each B block x,y and local entry (j2,k2)
            for bx in range(2):
                for by in range(2):
                    coeff_b = b_block_coeff[ti, bx * 2 + by]
                    if coeff_b == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(2):
                            v2_idx = j2 * 2 + k2
                            v2_val = V2x2[v2_idx, si]
                            if v2_val == 0:
                                continue
                            global_v_row = (bx * 2 + j2) * 4 + (by * 2 + k2)
                            V[global_v_row, r_idx] += coeff_b * v2_val
            
            # C coefficients: for each C block x,y and local entry (i2,k2)
            for cx in range(2):
                for cy in range(2):
                    coeff_c = c_coeff[cx * 2 + cy, ti]
                    if coeff_c == 0:
                        continue
                    for i2 in range(2):
                        for k2 in range(2):
                            w2_idx = i2 * 2 + k2
                            w2_val = W2x2[w2_idx, si]
                            if w2_val == 0:
                                continue
                            global_w_row = (cy * 2 + k2) * 4 + (cx * 2 + i2)
                            W[global_w_row, r_idx] += coeff_c * w2_val
    
    col_offset += 49
    
    # --- Block 2: A[0:4, 4:5] @ B[4:5, 0:4] -> C[0:4, 0:4] ---
    # A part: column j=4, rows i=0..3
    # B part: row j=4, cols k=0..3
    # C part: rows i=0..3, cols k=0..3
    # Schoolbook: for each i, k, one multiplication.
    for i in range(4):
        for k in range(4):
            r_idx = col_offset + i * 4 + k
            U[i * 5 + 4, r_idx] = 1   # A[i, 4]
            V[4 * 5 + k, r_idx] = 1   # B[4, k]
            W[k * 4 + i, r_idx] = 1   # C[i, k]
    
    col_offset += 16
    
    # --- Block 3: A[0:4, 0:4] @ B[0:4, 4:5] -> C[0:4, 4:5] ---
    # A part: rows i=0..3, cols j=0..3
    # B part: rows j=0..3, col k=4
    # C part: rows i=0..3, col k=4
    for i in range(4):
        for j in range(4):
            r_idx = col_offset + i * 4 + j
            U[i * 5 + j, r_idx] = 1   # A[i, j]
            V[j * 5 + 4, r_idx] = 1   # B[j, 4]
            W[4 * 4 + i, r_idx] = 1   # C[i, 4]
    
    col_offset += 16
    
    # --- Block 4: A[0:4, 4:5] @ B[4:5, 4:5] -> C[0:4, 4:5] ---
    # A part: column j=4, rows i=0..3
    # B part: row j=4, col k=4
    # C part: rows i=0..3, col k=4
    for i in range(4):
        r_idx = col_offset + i
        U[i * 5 + 4, r_idx] = 1   # A[i, 4]
        V[4 * 5 + 4, r_idx] = 1   # B[4, 4]
        W[4 * 4 + i, r_idx] = 1   # C[i, 4]
    
    col_offset += 4
    
    # Convert to lists for the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    print(f"U shape: {len(result['U'])}x{len(result['U'][0])}")
    print(f"V shape: {len(result['V'])}x{len(result['V'][0])}")
    print(f"W shape: {len(result['W'])}x{len(result['W'][0])}")
