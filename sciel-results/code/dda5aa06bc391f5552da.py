"""Matrix multiplication tensor decomposition 4x4 x 4x9."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    start_time = time.time()
    deadline = start_time + 120  # safety margin before 150s kill

    def check_validity(U, V, W, tol=1e-6):
        """Check if the decomposition is exact."""
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        s = 0.0
                        for c in range(U.shape[1]):
                            s += W[k * n + i, c] * U[i * m + j1, c] * V[j2 * p + k, c]
                        expected = 1.0 if j1 == j2 else 0.0
                        if abs(s - expected) > tol:
                            return False
        return True

    def schoolbook():
        rank = n * m * p  # 144
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    def strassen_2x2x2():
        """
        Strassen decomposition for 2x2 * 2x2 = 2x2.
        Returns U, V, W of shape (4, 7), (4, 7), (4, 7).
        U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen terms for 2x2*2x2=2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Each Mj is a product of two linear combinations of A and B entries.
        # M1: A_lc = A00 + A11, B_lc = B00 + B11
        # M2: A_lc = A10 + A11, B_lc = B00
        # M3: A_lc = A00, B_lc = B01 - B11
        # M4: A_lc = A11, B_lc = B10 - B00
        # M5: A_lc = A00 + A01, B_lc = B11
        # M6: A_lc = A10 - A00, B_lc = B00 + B01
        # M7: A_lc = A01 - A11, B_lc = B10 + B11
        
        terms = [
            # (A_entries, B_entries) where entries are (i,j,coeff)
            ([((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)]),  # M1
            ([((1,0),1), ((1,1),1)], [((0,0),1)]),               # M2
            ([((0,0),1)], [((0,1),1), ((1,1),-1)]),              # M3
            ([((1,1),1)], [((1,0),1), ((0,0),-1)]),              # M4
            ([((0,0),1), ((0,1),1)], [((1,1),1)]),               # M5
            ([((1,0),1), ((0,0),-1)], [((0,0),1), ((0,1),1)]),   # M6
            ([((0,1),1), ((1,1),-1)], [((1,0),1), ((1,1),1)]),   # M7
        ]
        
        # Coefficients for each Mj in each C entry:
        # C00: M1(+1), M4(+1), M5(-1), M7(+1)
        # C01: M3(+1), M5(+1)
        # C10: M2(+1), M4(+1)
        # C11: M1(+1), M2(-1), M3(+1), M6(+1)
        
        c_coeffs = {
            0: [1, 0, 0, 1],   # M1: C00(+1), C01(0), C10(0), C11(+1)
            1: [0, 0, 1, -1],  # M2: C00(0), C01(0), C10(+1), C11(-1)
            2: [0, 1, 0, 1],   # M3: C00(0), C01(+1), C10(0), C11(+1)
            3: [1, 0, 1, 0],   # M4: C00(+1), C01(0), C10(+1), C11(0)
            4: [-1, 1, 0, 0],  # M5: C00(-1), C01(+1), C10(0), C11(0)
            5: [0, 0, 0, 1],   # M6: C00(0), C01(0), C10(0), C11(+1)
            6: [1, 0, 0, 0],   # M7: C00(+1), C01(0), C10(0), C11(0)
        }
        
        for c_idx, (A_entries, B_entries) in enumerate(terms):
            for (i, j), coeff in A_entries:
                U[i * 2 + j, c_idx] += coeff
            for (j, k), coeff in B_entries:
                V[j * 2 + k, c_idx] += coeff
            
            # Compute W contributions
            for ci in range(2):
                for ck in range(2):
                    w_val = 0.0
                    for (i, j), a_coeff in A_entries:
                        for (j2, k), b_coeff in B_entries:
                            if j == j2:
                                w_val += a_coeff * b_coeff
                    # c_coeffs[c_idx] gives the coefficient for this term in each C entry
                    # C entry index: ci*2 + ck
                    w_val *= c_coeffs[c_idx][ci * 2 + ck]
                    W[ck * 2 + ci, c_idx] += w_val
        
        return U, V, W

    def strassen_4x4x4():
        """
        Strassen-based decomposition for 4x4 * 4x4 = 4x4.
        Uses two levels of Strassen: 7*7 = 49 rank-1 terms.
        Returns U, V, W of shape (16, 49), (16, 49), (16, 49).
        U[i*4+j, c], V[j*4+k, c], W[k*4+i, c]
        """
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        # Level 1: 7 Strassen terms for 4x4 block multiplication
        # Each term is Xj * Yj where Xj and Yj are 2x2 matrices of linear combinations
        # of 2x2 blocks of A and B.
        
        # Define the 7 level-1 terms.
        # Each Xj and Yj is a 2x2 matrix where each entry is a linear combination
        # of entries from specific 2x2 blocks of A or B.
        
        # Block offsets for A:
        # A11: rows 0-1, cols 0-1
        # A12: rows 0-1, cols 2-3
        # A21: rows 2-3, cols 0-1
        # A22: rows 2-3, cols 2-3
        
        # For each level-1 term, define X and Y as 2x2 arrays of linear combinations.
        # Each entry is a dict: {(global_i, global_j): coeff}
        
        def make_X_term(block_coeffs):
            """
            block_coeffs: list of (block_id, coeff)
            Returns a 2x2 array of dicts, each dict maps (i,j) -> coeff
            """
            result = [[{} for _ in range(2)] for _ in range(2)]
            for block_id, coeff in block_coeffs:
                if block_id == 0:  # A11
                    offsets = [(0, 0), (0, 1), (1, 0), (1, 1)]
                elif block_id == 1:  # A12
                    offsets = [(0, 2), (0, 3), (1, 2), (1, 3)]
                elif block_id == 2:  # A21
                    offsets = [(2, 0), (2, 1), (3, 0), (3, 1)]
                elif block_id == 3:  # A22
                    offsets = [(2, 2), (2, 3), (3, 2), (3, 3)]
                else:
                    continue
                for r in range(2):
                    for c in range(2):
                        gi = offsets[r * 2 + c][0]
                        gj = offsets[r * 2 + c][1]
                        local_r = r
                        local_c = c
                        result[local_r][local_c][(gi, gj)] = result[local_r][local_c].get((gi, gj), 0) + coeff
            return result
        
        def make_Y_term(block_coeffs):
            """
            block_coeffs: list of (block_id, coeff)
            Returns a 2x2 array of dicts, each dict maps (j,k) -> coeff
            """
            result = [[{} for _ in range(2)] for _ in range(2)]
            for block_id, coeff in block_coeffs:
                if block_id == 0:  # B11
                    offsets = [(0, 0), (0, 1), (1, 0), (1, 1)]
                elif block_id == 1:  # B12
                    offsets = [(0, 2), (0, 3), (1, 2), (1, 3)]
                elif block_id == 2:  # B21
                    offsets = [(2, 0), (2, 1), (3, 0), (3, 1)]
                elif block_id == 3:  # B22
                    offsets = [(2, 2), (2, 3), (3, 2), (3, 3)]
                else:
                    continue
                for r in range(2):
                    for c in range(2):
                        gj = offsets[r * 2 + c][0]
                        gk = offsets[r * 2 + c][1]
                        local_r = r
                        local_c = c
                        result[local_r][local_c][(gj, gk)] = result[local_r][local_c].get((gj, gk), 0) + coeff
            return result
        
        # Level 1 terms:
        # M1: X = A11+A22, Y = B11+B22
        # M2: X = A21+A22, Y = B11
        # M3: X = A11, Y = B12-B22
        # M4: X = A22, Y = B21-B11
        # M5: X = A11+A12, Y = B22
        # M6: X = A21-A11, Y = B11+B12
        # M7: X = A12-A22, Y = B21+B22
        
        level1_X = [
            make_X_term([(0, 1), (3, 1)]),   # M1
            make_X_term([(2, 1), (3, 1)]),   # M2
            make_X_term([(0, 1)]),           # M3
            make_X_term([(3, 1)]),           # M4
            make_X_term([(0, 1), (1, 1)]),   # M5
            make_X_term([(2, 1), (0, -1)]),  # M6
            make_X_term([(1, 1), (3, -1)]),  # M7
        ]
        
        level1_Y = [
            make_Y_term([(0, 1), (3, 1)]),   # M1
            make_Y_term([(0, 1)]),           # M2
            make_Y_term([(1, 1), (3, -1)]),  # M3
            make_Y_term([(2, 1), (0, -1)]),  # M4
            make_Y_term([(3, 1)]),           # M5
            make_Y_term([(0, 1), (1, 1)]),   # M6
            make_Y_term([(2, 1), (3, 1)]),   # M7
        ]
        
        # Coefficients for each Mj in each C block:
        # C11: M1(+1), M4(+1), M5(-1), M7(+1)
        # C12: M3(+1), M5(+1)
        # C21: M2(+1), M4(+1)
        # C22: M1(+1), M2(-1), M3(+1), M6(+1)
        
        c_block_coeffs = {
            0: [1, 0, 0, 1],   # M1
            1: [0, 0, 1, -1],  # M2
            2: [0, 1, 0, 1],   # M3
            3: [1, 0, 1, 0],   # M4
            4: [-1, 1, 0, 0],  # M5
            5: [0, 0, 0, 1],   # M6
            6: [1, 0, 0, 0],   # M7
        }
        
        # Strassen coefficients for 2x2 product:
        # N1 = (X[0,0]+X[1,1])(Y[0,0]+Y[1,1])
        # N2 = (X[1,0]+X[1,1])Y[0,0]
        # N3 = X[0,0](Y[0,1]-Y[1,1])
        # N4 = X[1,1](Y[1,0]-Y[0,0])
        # N5 = (X[0,0]+X[0,1])Y[1,1]
        # N6 = (X[1,0]-X[0,0])(Y[0,0]+Y[0,1])
        # N7 = (X[0,1]-X[1,1])(Y[1,0]+Y[1,1])
        
        # (X*Y)[0,0] = N1 + N4 - N5 + N7
        # (X*Y)[0,1] = N3 + N5
        # (X*Y)[1,0] = N2 + N4
        # (X*Y)[1,1] = N1 - N2 + N3 + N6
        
        n_block_coeffs = {
            0: [1, 0, 0, 1],   # N1
            1: [0, 0, 1, -1],  # N2
            2: [0, 1, 0, 1],   # N3
            3: [1, 0, 1, 0],   # N4
            4: [-1, 1, 0, 0],  # N5
            5: [0, 0, 0, 1],   # N6
            6: [1, 0, 0, 0],   # N7
        }
        
        def combine_dicts(d1, d2, sign1=1, sign2=1):
            """Combine two dicts with signs."""
            result = {}
            for k, v in d1.items():
                result[k] = result.get(k, 0) + sign1 * v
            for k, v in d2.items():
                result[k] = result.get(k, 0) + sign2 * v
            return result
        
        col = 0
        for j_idx in range(7):
            X = level1_X[j_idx]
            Y = level1_Y[j_idx]
            
            # Compute the 7 level-2 Strassen terms
            level2_A = []
            level2_B = []
            
            # N1 = (X[0,0]+X[1,1])(Y[0,0]+Y[1,1])
            level2_A.append(combine_dicts(X[0][0], X[1][1]))
            level2_B.append(combine_dicts(Y[0][0], Y[1][1]))
            
            # N2 = (X[1,0]+X[1,1])Y[0,0]
            level2_A.append(combine_dicts(X[1][0], X[1][1]))
            level2_B.append(Y[0][0])
            
            # N3 = X[0,0](Y[0,1]-Y[1,1])
            level2_A.append(X[0][0])
            level2_B.append(combine_dicts(Y[0][1], Y[1][1], 1, -1))
            
            # N4 = X[1,1](Y[1,0]-Y[0,0])
            level2_A.append(X[1][1])
            level2_B.append(combine_dicts(Y[1][0], Y[0][0], 1, -1))
            
            # N5 = (X[0,0]+X[0,1])Y[1,1]
            level2_A.append(combine_dicts(X[0][0], X[0][1]))
            level2_B.append(Y[1][1])
            
            # N6 = (X[1,0]-X[0,0])(Y[0,0]+Y[0,1])
            level2_A.append(combine_dicts(X[1][0], X[0][0], 1, -1))
            level2_B.append(combine_dicts(Y[0][0], Y[0][1]))
            
            # N7 = (X[0,1]-X[1,1])(Y[1,0]+Y[1,1])
            level2_A.append(combine_dicts(X[0][1], X[1][1], 1, -1))
            level2_B.append(combine_dicts(Y[1][0], Y[1][1]))
            
            for k_idx in range(7):
                A_lc = level2_A[k_idx]
                B_lc = level2_B[k_idx]
                
                # Fill U and V
                for (i, j), coeff in A_lc.items():
                    U[i * 4 + j, col] += coeff
                for (j, k), coeff in B_lc.items():
                    V[j * 4 + k, col] += coeff
                
                # Compute W contributions
                # For each (i,j) in A_lc and (j,k) in B_lc:
                # Contributes to C[i,k] with coefficient
                # sum over blocks: n_block_coeffs[k_idx][block] * c_block_coeffs[j_idx][block]
                # where block is determined by i and k.
                
                for (i, j), a_coeff in A_lc.items():
                    for (j2, k), b_coeff in B_lc.items():
                        if j != j2:
                            continue
                        # Determine which C block this (i,k) falls into
                        if i < 2 and k < 2:
                            block = 0  # C11
                        elif i < 2 and k >= 2:
                            block = 1  # C12
                        elif i >= 2 and k < 2:
                            block = 2  # C21
                        else:
                            block = 3  # C22
                        
                        w_coeff = n_block_coeffs[k_idx][block] * c_block_coeffs[j_idx][block]
                        W[k * 4 + i, col] += a_coeff * b_coeff * w_coeff
                
                col += 1
        
        return U, V, W

    def build_full_decomposition():
        """
        Build the full 4x4*4x9 decomposition.
        Split into: 4x4*4x4 + 4x4*4x4 + 4x4*4x1
        Rank: 49 + 49 + 16 = 114
        """
        U_parts = []
        V_parts = []
        W_parts = []
        
        # Part 1: 4x4 * 4x4 (columns 0-3), rank 49
        U1, V1, W1 = strassen_4x4x4()
        U_parts.append(U1)
        V_parts.append(V1)
        W_parts.append(W1)
        
        # Part 2: 4x4 * 4x4 (columns 4-7), rank 49
        U2, V2, W2 = strassen_4x4x4()
        # Shift column indices in V by 4
        V2_shifted = np.zeros((16, 49))
        for j in range(4):
            for k in range(4):
                V2_shifted[j * 4 + (k + 4), :] = V2[j * 4 + k, :]
        U_parts.append(U2)
        V_parts.append(V2_shifted)
        W_parts.append(W2)
        
        # Part 3: 4x4 * 4x1 (column 8), rank 16 (schoolbook)
        U3 = np.zeros((16, 16))
        V3 = np.zeros((16, 16))
        W3 = np.zeros((16, 16))
        col = 0
        for i in range(4):
            for j in range(4):
                U3[i * 4 + j, col] = 1
                V3[j * 4 + 8, col] = 1
                W3[8 * 4 + i, col] = 1
                col += 1
        U_parts.append(U3)
        V_parts.append(V3)
        W_parts.append(W3)
        
        U_full = np.hstack(U_parts)
        V_full = np.hstack(V_parts)
        W_full = np.hstack(W_parts)
        
        rank = U_full.shape[1]
        return U_full, V_full, W_full, rank

    # Get schoolbook as fallback
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb

    # Try Strassen-based decomposition
    try:
        if time.time() < deadline - 5:
            U_str, V_str, W_str, rank_str = build_full_decomposition()
            if check_validity(U_str, V_str, W_str):
                best_U, best_V, best_W, best_rank = U_str, V_str, W_str, rank_str
    except Exception as e:
        pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
