"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin (evaluator kills at 150s)
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Strategy: Hybrid decomposition
    # Part 1: 4x4 * 4x4 (columns 0-3 of B) using Strassen -> rank 49
    # Part 2: 4x4 * 4x3 (columns 4-6 of B) using Schoolbook -> rank 48
    # Total rank = 49 + 48 = 97
    #
    # Convention: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (28, rank)
    # W: (p*n, rank) = (28, rank)
    
    rank = 97
    U = [[0] * rank for _ in range(n * m)]   # 16 x 97
    V = [[0] * rank for _ in range(m * p)]   # 28 x 97
    W = [[0] * rank for _ in range(p * n)]   # 28 x 97
    
    col = 0
    
    # --- Part 1: Strassen 4x4x4 (Rank 49) for B columns 0..3 ---
    # Split 4x4 into four 2x2 blocks.
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # Each block is 2x2.
    #
    # Strassen at block level (7 block multiplications):
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each block multiplication is a 2x2 * 2x2 = 2x2 tensor (rank 7 via Strassen,
    # but we use schoolbook rank 8 for simplicity and correctness).
    # Total: 7 * 8 = 56... wait, that's not 49.
    #
    # Actually, let me use Strassen for the 2x2x2 sub-problems too.
    # 2x2x2 has rank 7 (Strassen). So 7 * 7 = 49.
    #
    # Let me implement this properly.
    
    # Strassen 2x2x2 decomposition (rank 7):
    # For X(2x2) * Y(2x2) = Z(2x2):
    # Split X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # M1 = (x00+x11)*(y00+y11)
    # M2 = (x10+x11)*y00
    # M3 = x00*(y01-y11)
    # M4 = x11*(y10-y00)
    # M5 = (x00+x01)*y11
    # M6 = (x10-x00)*(y00+y01)
    # M7 = (x01-x11)*(y10+y11)
    #
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # U factors for 2x2x2 Strassen (7 terms):
    # Each term r has U[i,j], V[j,k], W[k,i] coefficients
    # i,j,k in {0,1}
    
    # Term 1: M1 = (x00+x11)*(y00+y11)
    # U: x00=1, x11=1; V: y00=1, y11=1; W: Z00+=1, Z11+=1
    # Term 2: M2 = (x10+x11)*y00
    # U: x10=1, x11=1; V: y00=1; W: Z10+=1, Z11-=1
    # Term 3: M3 = x00*(y01-y11)
    # U: x00=1; V: y01=1, y11=-1; W: Z01+=1, Z11+=1
    # Term 4: M4 = x11*(y10-y00)
    # U: x11=1; V: y10=1, y00=-1; W: Z00+=1, Z10+=1
    # Term 5: M5 = (x00+x01)*y11
    # U: x00=1, x01=1; V: y11=1; W: Z00-=1, Z01+=1
    # Term 6: M6 = (x10-x00)*(y00+y01)
    # U: x10=1, x00=-1; V: y00=1, y01=1; W: Z11+=1
    # Term 7: M7 = (x01-x11)*(y10+y11)
    # U: x01=1, x11=-1; V: y10=1, y11=1; W: Z00+=1, Z11+=1
    
    # For the full 4x4x4 Strassen, we nest:
    # Outer Strassen: 7 block multiplications
    # Inner: each block mult uses 2x2x2 Strassen (rank 7)
    # Total: 7*7 = 49 terms
    
    # Let me define the outer Strassen coefficients:
    # Outer U: maps 4x4 A to 2x2 blocks, then to Strassen inputs
    # Outer V: maps 4x4 B to 2x2 blocks, then to Strassen inputs
    # Outer W: maps 2x2 block results to 4x4 C
    
    # Block indices: 0=A00, 1=A01, 2=A10, 3=A11 (same for B)
    # C blocks: 0=C00, 1=C01, 2=C10, 3=C11
    
    # Outer Strassen M terms (block level):
    # M1: U blocks [0,3], V blocks [0,3]
    # M2: U blocks [2,3], V blocks [0]
    # M3: U blocks [0], V blocks [1,3] with V[3]=-1
    # M4: U blocks [3], V blocks [2,0] with V[0]=-1
    # M5: U blocks [0,1], V blocks [3]
    # M6: U blocks [2,0] with U[0]=-1, V blocks [0,1]
    # M7: U blocks [1,3] with U[3]=-1, V blocks [2,3]
    
    # W coefficients for output blocks:
    # C00: M1(+1), M4(+1), M5(-1), M7(+1)
    # C01: M3(+1), M5(+1)
    # C10: M2(+1), M4(+1)
    # C11: M1(+1), M2(-1), M3(+1), M6(+1)
    
    # For each outer term k (0..6) and inner term l (0..6),
    # we have a combined term with:
    # U_full[i,j] = sum over outer block b_u, inner (i_in,j_in) of u_outer[k][b_u] * u_inner[l][i_in,j_in]
    # where i = 2*(b_u//2) + i_in, j = 2*(b_u%2) + j_in
    # Similarly for V and W.
    
    # Let me build lookup tables.
    
    # Outer U: for each outer term k, which blocks and coefficients
    outer_u = {
        0: {0: 1, 3: 1},      # M1: A00 + A11
        1: {2: 1, 3: 1},      # M2: A10 + A11
        2: {0: 1},            # M3: A00
        3: {3: 1},            # M4: A11
        4: {0: 1, 1: 1},      # M5: A00 + A01
        5: {2: 1, 0: -1},     # M6: A10 - A00
        6: {1: 1, 3: -1},     # M7: A01 - A11
    }
    
    outer_v = {
        0: {0: 1, 3: 1},      # M1: B00 + B11
        1: {0: 1},            # M2: B00
        2: {1: 1, 3: -1},     # M3: B01 - B11
        3: {2: 1, 0: -1},     # M4: B10 - B00
        4: {3: 1},            # M5: B11
        5: {0: 1, 1: 1},      # M6: B00 + B01
        6: {2: 1, 3: 1},      # M7: B10 + B11
    }
    
    outer_w = {
        0: {0: 1, 3: 1},      # C00: M1, C11: M1
        1: {2: 1, 3: -1},     # C10: M2, C11: -M2
        2: {1: 1, 3: 1},      # C01: M3, C11: M3
        3: {0: 1, 2: 1},      # C00: M4, C10: M4
        4: {0: -1, 1: 1},     # C00: -M5, C01: M5
        5: {3: 1},            # C11: M6
        6: {0: 1, 3: 1},      # C00: M7, C11: M7
    }
    
    # Inner 2x2x2 Strassen coefficients:
    # For inner term l, u_inner[i_in, j_in], v_inner[j_in, k_in], w_inner[k_in, i_in]
    # where i_in, j_in, k_in in {0, 1}
    
    inner_u = [
        {0: 1, 3: 1},         # M1: x00 + x11
        {2: 1, 3: 1},         # M2: x10 + x11
        {0: 1},               # M3: x00
        {3: 1},               # M4: x11
        {0: 1, 1: 1},         # M5: x00 + x01
        {2: 1, 0: -1},        # M6: x10 - x00
        {1: 1, 3: -1},        # M7: x01 - x11
    ]
    
    inner_v = [
        {0: 1, 3: 1},         # M1: y00 + y11
        {0: 1},               # M2: y00
        {1: 1, 3: -1},        # M3: y01 - y11
        {2: 1, 0: -1},        # M4: y10 - y00
        {3: 1},               # M5: y11
        {0: 1, 1: 1},         # M6: y00 + y01
        {2: 1, 3: 1},         # M7: y10 + y11
    ]
    
    inner_w = [
        {0: 1, 3: 1},         # Z00: M1, Z11: M1
        {2: 1, 3: -1},        # Z10: M2, Z11: -M2
        {1: 1, 3: 1},         # Z01: M3, Z11: M3
        {0: 1, 2: 1},         # Z00: M4, Z10: M4
        {0: -1, 1: 1},        # Z00: -M5, Z01: M5
        {3: 1},               # Z11: M6
        {0: 1, 3: 1},         # Z00: M7, Z11: M7
    ]
    
    # Now construct the 49 terms for Part 1
    for k in range(7):  # outer Strassen term
        for l in range(7):  # inner Strassen term
            if time.time() > deadline:
                # Return schoolbook fallback
                return _schoolbook(n, m, p)
            
            u_vec = [0] * (n * m)  # 16
            v_vec = [0] * (m * p)  # 28
            w_vec = [0] * (p * n)  # 28
            
            # Build U vector
            for b_u, coeff_ou in outer_u[k].items():
                b_row = b_u // 2
                b_col = b_u % 2
                for i_in, j_in in range(2):
                    for j_in2 in range(2):
                        pass  # Need to iterate properly
            
            # Let me redo this more carefully
            u_vec = [0] * (n * m)
            for b_u, coeff_ou in outer_u[k].items():
                b_row = b_u // 2
                b_col = b_u % 2
                for i_in in range(2):
                    for j_in in range(2):
                        key = i_in * 2 + j_in
                        if key in inner_u[l]:
                            coeff_oi = inner_u[l][key]
                            total = coeff_ou * coeff_oi
                            if total != 0:
                                i_global = 2 * b_row + i_in
                                j_global = 2 * b_col + j_in
                                u_vec[i_global * m + j_global] += total
            
            # Build V vector
            v_vec = [0] * (m * p)
            for b_v, coeff_ov in outer_v[k].items():
                b_row = b_v // 2
                b_col = b_v % 2
                for j_in in range(2):
                    for k_in in range(2):
                        key = j_in * 2 + k_in
                        if key in inner_v[l]:
                            coeff_oi = inner_v[l][key]
                            total = coeff_ov * coeff_oi
                            if total != 0:
                                j_global = 2 * b_row + j_in
                                k_global = 2 * b_col + k_in
                                # For Part 1, k_global is 0..3 (columns 0-3 of B)
                                v_vec[j_global * p + k_global] += total
            
            # Build W vector
            w_vec = [0] * (p * n)
            for b_w, coeff_ow in outer_w[k].items():
                b_row = b_w // 2
                b_col = b_w % 2
                for k_in in range(2):
                    for i_in in range(2):
                        key = k_in * 2 + i_in
                        if key in inner_w[l]:
                            coeff_oi = inner_w[l][key]
                            total = coeff_ow * coeff_oi
                            if total != 0:
                                k_global = 2 * b_col + k_in
                                i_global = 2 * b_row + i_in
                                # For Part 1, k_global is 0..3
                                w_vec[k_global * n + i_global] += total
            
            # Store in U, V, W
            for r in range(n * m):
                U[r][col] = u_vec[r]
            for r in range(m * p):
                V[r][col] = v_vec[r]
            for r in range(p * n):
                W[r][col] = w_vec[r]
            
            col += 1
    
    # --- Part 2: Schoolbook 4x4x3 (Rank 48) for B columns 4..6 ---
    # C[i, k] = sum_j A[i,j] * B[j, k] for k in 4,5,6
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}


def _schoolbook(n, m, p):
    """Fallback schoolbook decomposition."""
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
# EVOLVE-BLOCK-END
