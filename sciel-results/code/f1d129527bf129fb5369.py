"""Matrix multiplication 4x4x5 - rank minimization via Strassen block decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]
n, m, p = DIMENSIONS

def verify_decomposition(U, V, W):
    """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_sch = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sch = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sch = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i*m+j][col] = 1
                V_sch[j*p+k][col] = 1
                W_sch[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_sch, "V": V_sch, "W": W_sch}
    best_rank = rank_schoolbook

    # Try Rank 65 construction (Strassen 4x4x4 + Schoolbook 4x4x1)
    # 4x4x4 part: Rank 49
    # 4x4x1 part: Rank 16
    total_rank = 65
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Part 1: 4x4x4 Strassen (Rank 49) ---
    # Indices for 4x4x4: i, j, k in 0..3
    
    # Define the 7 Strassen terms for the 4x4x4 level
    # Each term is defined by:
    # 1. The 2x2 matrix X (coefficients for A blocks)
    # 2. The 2x2 matrix Y (coefficients for B blocks)
    # 3. The contribution to C blocks (coefficients for M_k in C_R,C)
    
    # A blocks: A00 (0,0), A01 (0,1), A10 (1,0), A11 (1,1)
    # B blocks: B00 (0,0), B01 (0,1), B10 (1,0), B11 (1,1)
    
    # Strassen terms (M1..M7)
    # M1 = (A00+A11) * (B00+B11)
    # M2 = (A10+A11) * B00
    # M3 = A00 * (B01-B11)
    # M4 = A11 * (B10-B00)
    # M5 = (A00+A01) * B11
    # M6 = (A10-A00) * (B00+B01)
    # M7 = (A01-A11) * (B10+B11)
    
    # For each term, we need to decompose the 2x2 matrix multiplication into 7 scalar multiplications.
    # We will use the standard 2x2x2 Strassen decomposition for the sub-multiplications.
    
    # Helper: Add a rank-1 term to U, V, W
    # u_idx: dict mapping (i, j) to coefficient
    # v_idx: dict mapping (j, k) to coefficient
    # w_idx: dict mapping (k, i) to coefficient
    # col: current column index
    
    def add_term(u_dict, v_dict, w_dict, col_idx):
        for (i, j), coeff_u in u_dict.items():
            if coeff_u != 0:
                U[i*m+j][col_idx] += coeff_u
        for (j, k), coeff_v in v_dict.items():
            if coeff_v != 0:
                V[j*p+k][col_idx] += coeff_v
        for (k, i), coeff_w in w_dict.items():
            if coeff_w != 0:
                W[k*n+i][col_idx] += coeff_w

    col_idx = 0
    
    # Define the 2x2x2 Strassen sub-terms
    # For X = [[x00, x01], [x10, x11]] and Y = [[y00, y01], [y10, y11]]
    # m1 = (x00+x11)(y00+y11)
    # m2 = (x10+x11)y00
    # m3 = x00(y01-y11)
    # m4 = x11(y10-y00)
    # m5 = (x00+x01)y11
    # m6 = (x10-x00)(y00+y01)
    # m7 = (x01-x11)(y10+y11)
    
    # Output Z:
    # z00 = m1 + m4 - m5 + m7
    # z01 = m3 + m5
    # z10 = m2 + m4
    # z11 = m1 - m2 + m3 + m6
    
    # We need to map the "scalar" x_rc and y_rc to global A and B indices.
    # For a given Strassen term M_k, X and Y are 2x2 matrices of linear combinations of A and B blocks.
    # Let X[r][c] be a linear combination of A blocks.
    # Let Y[r][c] be a linear combination of B blocks.
    
    # We will iterate through the 7 main Strassen terms.
    
    main_terms = [
        # (X_coeffs, Y_coeffs, C_coeffs)
        # X_coeffs: 2x2 matrix of lists of (block_idx, coeff)
        # Y_coeffs: 2x2 matrix of lists of (block_idx, coeff)
        # C_coeffs: 2x2 matrix of coeffs for C blocks
    ]
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    X1 = [
        [(0, 1)], # A00
        [(1, 1)]  # A11
    ] # Wait, X is 2x2.
    # The formula M1 = (A00+A11)(B00+B11) implies that the "scalar" product is of the sums.
    # But A00+A11 is a 2x2 matrix. B00+B11 is a 2x2 matrix.
    # So we are multiplying two 2x2 matrices.
    # Let X = A00+A11, Y = B00+B11.
    # X[0][0] = A00[0][0] + A11[0][0]
    # ...
    
    # This is getting complicated to map "block sums" to "scalar sums" for the 2x2x2 Strassen.
    # The 2x2x2 Strassen operates on the entries of X and Y.
    # So X is a 2x2 matrix of scalars (which are themselves linear combos of A entries).
    
    # Let's define the mapping from global A indices to X entries.
    # For Term 1:
    # X[0][0] = A[0][0] + A[2][2]
    # X[0][1] = A[0][1] + A[2][3]
    # X[1][0] = A[1][0] + A[3][2]
    # X[1][1] = A[1][1] + A[3][3]
    
    # Y[0][0] = B[0][0] + B[2][2]
    # Y[0][1] = B[0][1] + B[2][3]
    # Y[1][0] = B[1][0] + B[3][2]
    # Y[1][1] = B[1][1] + B[3][3]
    
    # C contribution:
    # C[0][0] += M1
    # C[1][1] += M1
    # C[0][1] -= M1? No.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for Term 1:
    # C[0][0] block gets +1 * M1
    # C[1][1] block gets +1 * M1
    
    # We will process each of the 7 sub-terms of the 2x2x2 multiplication.
    
    def process_strassen_term(X_map, Y_map, C_contrib, current_col):
        """
        X_map: dict mapping (r, c) to list of (global_i, global_j, coeff) for A
        Y_map: dict mapping (r, c) to list of (global_j, global_k, coeff) for B
        C_contrib: dict mapping (R, C) to coeff for C block
        """
        # Define the 7 sub-terms of 2x2x2
        # Each sub-term is a product u * v
        # u is a linear combo of X entries
        # v is a linear combo of Y entries
        
        sub_terms = [
            # (u_coeffs, v_coeffs, w_coeffs)
            # u_coeffs: dict (r, c) -> coeff
            # v_coeffs: dict (r, c) -> coeff
            # w_coeffs: dict (R, C) -> coeff (contribution to Z entries)
            
            # m1 = (x00+x11)(y00+y11)
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            # m2 = (x10+x11)y00
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
            # m3 = x00(y01-y11)
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}),
            # m4 = x11(y10-y00)
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            # m5 = (x00+x01)y11
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            # m6 = (x10-x00)(y00+y01)
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            # m7 = (x01-x11)(y10+y11)
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})
        ]
        
        for u_coeffs, v_coeffs, w_coeffs in sub_terms:
            # Build global u_dict, v_dict, w_dict
            u_dict = {}
            v_dict = {}
            w_dict = {}
            
            # Expand u
            for (r, c), coeff_u in u_coeffs.items():
                for (gi, gj, coeff_a) in X_map.get((r, c), []):
                    key = (gi, gj)
                    u_dict[key] = u_dict.get(key, 0) + coeff_u * coeff_a
            
            # Expand v
            for (r, c), coeff_v in v_coeffs.items():
                for (gj, gk, coeff_b) in Y_map.get((r, c), []):
                    key = (gj, gk)
                    v_dict[key] = v_dict.get(key, 0) + coeff_v * coeff_b
            
            # Expand w
            # w_coeffs maps Z indices (r, c) to coeff
            # C_contrib maps C block indices (R, C) to coeff
            # Z[r,c] contributes to C[R,C]
            # For a 2x2 block Z, it corresponds to a 2x2 block in C.
            # If Z is the result of M_k, and M_k contributes to C[R,C] with coeff beta,
            # then Z[r,c] contributes to C[R*2+r, C*2+c] with coeff beta * w_coeff.
            
            for (R, C), beta in C_contrib.items():
                for (r, c), w_coeff in w_coeffs.items():
                    gi = R * 2 + r
                    gj = C * 2 + c
                    # W index is (k, i) -> (gj, gi)
                    key = (gj, gi)
                    w_dict[key] = w_dict.get(key, 0) + beta * w_coeff
            
            add_term(u_dict, v_dict, w_dict, current_col)
            current_col += 1
        
        return current_col

    # Define X_map and Y_map for each of the 7 main terms
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    X1_map = {
        (0,0): [(0,0,1), (2,2,1)],
        (0,1): [(0,1,1), (2,3,1)],
        (1,0): [(1,0,1), (3,2,1)],
        (1,1): [(1,1,1), (3,3,1)]
    }
    Y1_map = {
        (0,0): [(0,0,1), (2,2,1)],
        (0,1): [(0,1,1), (2,3,1)],
        (1,0): [(1,0,1), (3,2,1)],
        (1,1): [(1,1,1), (3,3,1)]
    }
    C1_contrib = {(0,0):1, (1,1):1}
    col_idx = process_strassen_term(X1_map, Y1_map, C1_contrib, col_idx)
    
    # Term 2: M2 = (A10+A11)B00
    X2_map = {
        (0,0): [(2,0,1), (0,0,0)], # A10 is block (1,0) -> rows 2-3, cols 0-1. A11 is (1,1) -> rows 2-3, cols 2-3.
        # Wait, A10 is rows 2-3, cols 0-1. A11 is rows 2-3, cols 2-3.
        # X = A10 + A11
        (0,0): [(2,0,1), (2,2,1)], # A10[0,0] + A11[0,0]
        (0,1): [(2,1,1), (2,3,1)],
        (1,0): [(3,0,1), (3,2,1)],
        (1,1): [(3,1,1), (3,3,1)]
    }
    Y2_map = {
        (0,0): [(0,0,1)],
        (0,1): [(0,1,1)],
        (1,0): [(1,0,1)],
        (1,1): [(1,1,1)]
    }
    C2_contrib = {(1,0):1}
    col_idx = process_strassen_term(X2_map, Y2_map, C2_contrib, col_idx)
    
    # Term 3: M3 = A00(B01-B11)
    X3_map = {
        (0,0): [(0,0,1)],
        (0,1): [(0,1,1)],
        (1,0): [(1,0,1)],
        (1,1): [(1,1,1)]
    }
    Y3_map = {
        (0,0): [(0,2,1), (2,2,-1)], # B01[0,0] - B11[0,0]
        (0,1): [(0,3,1), (2,3,-1)],
        (1,0): [(1,2,1), (3,2,-1)],
        (1,1): [(1,3,1), (3,3,-1)]
    }
    C3_contrib = {(0,1):1, (1,1):1}
    col_idx = process_strassen_term(X3_map, Y3_map, C3_contrib, col_idx)
    
    # Term 4: M4 = A11(B10-B00)
    X4_map = {
        (0,0): [(2,2,1)],
        (0,1): [(2,3,1)],
        (1,0): [(3,2,1)],
        (1,1): [(3,3,1)]
    }
    Y4_map = {
        (0,0): [(2,0,1), (0,0,-1)],
        (0,1): [(2,1,1), (0,1,-1)],
        (1,0): [(3,0,1), (1,0,-1)],
        (1,1): [(3,1,1), (1,1,-1)]
    }
    C4_contrib = {(0,0):1, (1,0):1}
    col_idx = process_strassen_term(X4_map, Y4_map, C4_contrib, col_idx)
    
    # Term 5: M5 = (A00+A01)B11
    X5_map = {
        (0,0): [(0,0,1), (0,2,1)],
        (0,1): [(0,1,1), (0,3,1)],
        (1,0): [(1,0,1), (1,2,1)],
        (1,1): [(1,1,1), (1,3,1)]
    }
    Y5_map = {
        (0,0): [(2,2,1)],
        (0,1): [(2,3,1)],
        (1,0): [(3,2,1)],
        (1,1): [(3,3,1)]
    }
    C5_contrib = {(0,0):-1, (0,1):1}
    col_idx = process_strassen_term(X5_map, Y5_map, C5_contrib, col_idx)
    
    # Term 6: M6 = (A10-A00)(B00+B01)
    X6_map = {
        (0,0): [(2,0,1), (0,0,-1)],
        (0,1): [(2,1,1), (0,1,-1)],
        (1,0): [(3,0,1), (1,0,-1)],
        (1,1): [(3,1,1), (1,1,-1)]
    }
    Y6_map = {
        (0,0): [(0,0,1), (0,2,1)],
        (0,1): [(0,1,1), (0,3,1)],
        (1,0): [(1,0,1), (1,2,1)],
        (1,1): [(1,1,1), (1,3,1)]
    }
    C6_contrib = {(1,1):1}
    col_idx = process_strassen_term(X6_map, Y6_map, C6_contrib, col_idx)
    
    # Term 7: M7 = (A01-A11)(B10+B11)
    X7_map = {
        (0,0): [(0,2,1), (2,2,-1)],
        (0,1): [(0,3,1), (2,3,-1)],
        (1,0): [(1,2,1), (3,2,-1)],
        (1,1): [(1,3,1), (3,3,-1)]
    }
    Y7_map = {
        (0,0): [(2,0,1), (2,2,1)],
        (0,1): [(2,1,1), (2,3,1)],
        (1,0): [(3,0,1), (3,2,1)],
        (1,1): [(3,1,1), (3,3,1)]
    }
    C7_contrib = {(0,0):1}
    col_idx = process_strassen_term(X7_map, Y7_map, C7_contrib, col_idx)
    
    # --- Part 2: 4x4x1 Schoolbook (Rank 16) ---
    # This handles the 5th column of B (k=4)
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    # For each i, j:
    for i in range(4):
        for j in range(4):
            U[i*m+j][col_idx] = 1
            V[j*p+4][col_idx] = 1
            W[4*n+i][col_idx] = 1
            col_idx += 1
    
    # Verify
    if verify_decomposition(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        return best_result

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
